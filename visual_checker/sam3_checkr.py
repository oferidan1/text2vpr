import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional
import os
import shlex
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore[assignment]

# Prefer the installed sam3 package (with its compiled extensions).
# This avoids issues where importing directly from the source repo bypasses the
# compiled extensions and leads to runtime errors.
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ModuleNotFoundError:
    # Fallback: attempt to use a sibling sam3 repo if the package
    # is not installed in the current environment.
    _THIS_FILE = Path(__file__).resolve()
    _ROOT_DIR = _THIS_FILE.parents[2]  # /mnt/d/dan/git_projects
    _SAM3_DIR = _ROOT_DIR / "sam3"
    if not _SAM3_DIR.is_dir():
        raise

    # Add the repo itself to sys.path
    sys.path.insert(0, str(_SAM3_DIR))

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor


@dataclass
class DetectionRow:
    image_path: str
    description: str
    objects_raw: str
    stuff_raw: str = ""


def parse_input_csv(
    csv_path: Path,
    use_filtered: bool = False,
) -> Iterator[DetectionRow]:
    """
    Parse an input CSV produced by the caption-to-objects pipeline.

    Supported schemas:
    - Merged schema (LLM merged mode):
      - image_path
      - description
      - objects_and_stuff
      - optional: stuff
    - Split schema (LLM or noun-phrase non-merged mode):
      - image_path
      - description
      - objects
      - optional: stuff
    - Filtered schema (when use_filtered=True):
      - image_path
      - description
      - objects
      - filtered_by_llm  (used as the objects source)

    Args:
        csv_path: Path to the input CSV file.
        use_filtered: If True, use the 'filtered_by_llm' column as the objects source.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        fieldnames = set(reader.fieldnames)

        # Basic required columns common to all supported schemas.
        base_required = {"image_path", "description"}
        missing_base = base_required.difference(fieldnames)
        if missing_base:
            raise ValueError(
                f"Missing required columns in CSV {csv_path}: {sorted(missing_base)}. "
                f"Found columns: {reader.fieldnames}"
            )

        # Determine which object schema this CSV uses.
        if use_filtered:
            if "filtered_by_llm" not in fieldnames:
                raise ValueError(
                    f"CSV {csv_path} must contain a 'filtered_by_llm' column when "
                    f"--filtered flag is used. Found columns: {reader.fieldnames}"
                )
            schema = "filtered"
        elif "objects_and_stuff" in fieldnames:
            schema = "merged"
        elif "objects" in fieldnames:
            schema = "split"
        else:
            raise ValueError(
                f"CSV {csv_path} must contain either an 'objects_and_stuff' column "
                f"(merged schema) or an 'objects' column (split schema). "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            image_path = row.get("image_path") or ""
            description = row.get("description") or ""
            if schema == "merged":
                objects_raw = row.get("objects_and_stuff") or ""
            elif schema == "filtered":
                objects_raw = row.get("filtered_by_llm") or ""
            else:
                objects_raw = row.get("objects") or ""
            # Optional 'stuff' column is supported in both schemas.
            stuff_raw = row.get("stuff") or ""
            if not image_path:
                # Skip malformed rows instead of failing the whole job
                continue
            yield DetectionRow(
                image_path=str(image_path),
                description=str(description),
                objects_raw=str(objects_raw),
                stuff_raw=str(stuff_raw),
            )


def parse_objects_field(objects_text: str) -> List[str]:
    """
    Parse the 'objects' column into a list of object names.

    The default LLM-based pipeline uses '. ' as a separator, but we
    also accept commas and semicolons for robustness.
    """
    if not objects_text:
        return []

    normalized = objects_text.replace(";", ".").replace(",", ".")
    parts = [p.strip() for p in normalized.split(".") if p.strip()]
    return parts


def load_image(image_path: str):
    """Load an image for SAM3."""
    image_pil = Image.open(image_path).convert("RGB")
    return image_pil


def load_model(
    device: str,
    confidence_threshold: float = 0.3,
    verbose: bool = False,
) -> tuple[torch.nn.Module, Sam3Processor]:
    """Load a SAM3 model and processor."""
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(
        model=model, device=device, confidence_threshold=confidence_threshold
    )
    if verbose:
        print(f"Loaded SAM3 model on device: {device}")
    return model, processor


def run_sam3_for_object(
    processor: Sam3Processor,
    image_pil: Image.Image,
    inference_state: dict,
    object_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run SAM3 for a single object name on an image.

    Args:
        processor: SAM3 processor instance
        image_pil: PIL Image
        inference_state: State dict from set_image (will be reused)
        object_name: Name of the object to detect

    Returns:
        boxes: (N, 4) tensor in xyxy absolute coordinates.
        scores: (N,) tensor of confidence scores in [0, 1].
        masks: (N, H, W) tensor of boolean masks.
    """
    prompt = object_name.lower().strip()

    # Use the existing inference state (image already set)
    # Just update the text prompt
    output_state = processor.set_text_prompt(prompt=prompt, state=inference_state)

    boxes = output_state["boxes"]  # (N, 4) in xyxy format
    scores = output_state["scores"]  # (N,) tensor
    masks = output_state["masks"]  # (N, H, W) boolean tensor

    # If there are multiple detections for the same object above the threshold,
    # keep only the highest-confidence one (top-1) for both output and debug.
    if scores is not None and scores.numel() > 1:
        top_idx = int(torch.argmax(scores).item())
        boxes = boxes[top_idx : top_idx + 1]
        scores = scores[top_idx : top_idx + 1]
        if masks is not None:
            masks = masks[top_idx : top_idx + 1]

    return boxes, scores, masks


def create_color_map(
    labels: List[str],
    box_width: int = 300,
    item_height: int = 30,
) -> Image.Image:
    """
    Create a legend/map image showing color-to-object mappings.
    
    Args:
        labels: List of label strings (e.g., "object_name (0.85)")
        box_width: Width of the legend image
        item_height: Height of each item in the legend
    
    Returns:
        PIL Image with the color map/legend
    """
    if not labels:
        # Return a small empty image
        return Image.new("RGB", (box_width, 50), color="white")
    
    # Get unique labels (in case of duplicates)
    unique_labels = []
    seen = set()
    for label in labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)
    
    # Calculate image height
    padding = 20
    legend_height = len(unique_labels) * item_height + 2 * padding
    
    # Create legend image
    legend_img = Image.new("RGB", (box_width, legend_height), color="white")
    draw = ImageDraw.Draw(legend_img)
    
    # Try to load a font
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, 14)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # Draw title
    title = "Color Map"
    if hasattr(draw, "textbbox"):
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_w = title_bbox[2] - title_bbox[0]
    else:
        title_w, _ = draw.textsize(title, font=font)
    draw.text(((box_width - title_w) // 2, 5), title, fill="black", font=font)
    
    # Draw each item
    color_box_size = 25
    text_x = color_box_size + 15
    y_offset = padding + 25
    
    for idx, label in enumerate(unique_labels):
        y_pos = y_offset + idx * item_height
        
        # Generate the same consistent color as used in the main image
        np.random.seed(hash(label) % (2**32))
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        
        # Draw color box
        color_box = [
            padding,
            y_pos,
            padding + color_box_size,
            y_pos + color_box_size,
        ]
        draw.rectangle(color_box, fill=color, outline="black", width=2)
        
        # Draw label text
        draw.text((text_x, y_pos + 5), label, fill="black", font=font)
    
    return legend_img


def draw_segments_only(
    image_pil: Image.Image,
    boxes: torch.Tensor,
    labels: List[str],
    masks: Optional[torch.Tensor] = None,
) -> Image.Image:
    """
    Draw only the segmentation masks on the image (no labels, no boxes).
    Each segment gets a unique color that matches the color map.

    Args:
        image_pil: Original image.
        boxes: (N, 4) tensor in xyxy format (not used, but kept for compatibility).
        labels: List of N strings (used to generate consistent colors).
        masks: Optional (N, H, W) tensor of boolean masks for segmentation.

    Returns:
        PIL Image with colored segments overlaid.
    """
    if len(boxes) == 0:
        return image_pil

    # Convert to numpy for mask overlay
    img_np = np.array(image_pil).copy()
    H, W = img_np.shape[:2]

    # Draw masks only
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        if masks is not None and idx < masks.shape[0]:
            mask = masks[idx].detach().cpu().numpy()
            # Handle different mask shapes
            if mask.ndim == 3:
                mask = mask[0]
            elif mask.ndim != 2:
                continue
            
            # Ensure mask matches image dimensions
            if mask.shape != (H, W):
                from torch.nn.functional import interpolate
                mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                mask_resized = interpolate(
                    mask_tensor, size=(H, W), mode="bilinear", align_corners=False
                )
                mask = (mask_resized.squeeze().numpy() > 0.5).astype(bool)
            else:
                mask = mask.astype(bool)
            
            # Generate a consistent color for this detection (same as in color map)
            np.random.seed(hash(label) % (2**32))
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            color_np = np.array(color, dtype=np.uint8)
            
            # Overlay mask with transparency
            alpha = 0.5  # Slightly more opaque for better visibility
            img_np[mask] = (alpha * color_np + (1 - alpha) * img_np[mask]).astype(np.uint8)

    # Convert back to PIL
    result_img = Image.fromarray(img_np)
    return result_img


def prepare_clean_directory(dir_path: Path) -> None:
    """
    Ensure dir_path exists and is empty.
    If it already exists, remove all its contents (files and subdirectories).
    """
    if dir_path.exists():
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Expected a directory for debug output: {dir_path}")
        for item in dir_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        dir_path.mkdir(parents=True, exist_ok=True)


def wait_for_file(path: Path, timeout_sec: int = 300, poll_interval: float = 2.0) -> bool:
    """
    Wait for a specific file to appear on disk.

    Returns True if the file was found within the timeout, False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if path.is_file():
            return True
        time.sleep(poll_interval)
    return False


def maybe_load_cluster_index_map(csv_dir: Path) -> dict[tuple[str, str], str]:
    """
    If a 'cluster_items.csv' file exists in csv_dir, load a mapping from
    (image_path, description) -> idx_in_cluster.

    This lets us propagate idx_in_cluster into the per-object SAM3 CSV.
    """
    cluster_items_path = csv_dir / "cluster_items.csv"
    if not cluster_items_path.is_file():
        return {}

    mapping: dict[tuple[str, str], str] = {}

    with cluster_items_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}

        required = {"image_path", "description", "idx_in_cluster"}
        missing = required.difference(reader.fieldnames)
        if missing:
            # If the expected columns are not present, silently skip mapping.
            return {}

        for row in reader:
            img_path = (row.get("image_path") or "").strip()
            desc = (row.get("description") or "").strip()
            idx = row.get("idx_in_cluster")
            if not img_path or idx is None:
                continue
            mapping[(img_path, desc)] = str(idx)

    return mapping


def run(
    input_csv: Path,
    output_csv: Optional[Path],
    images_root: Optional[Path],
    box_threshold: float,
    cpu_only: bool,
    debug: bool,
    debug_dir: Optional[Path],
    no_overwrite: bool = False,
    use_filtered: bool = False,
    realtime_missing_csv: Optional[Path] = None,
    realtime_progress_csv: Optional[Path] = None,
) -> None:
    # Determine suffix based on whether we're using filtered objects.
    suffix = "_sam3_filtered" if use_filtered else "_sam3"

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + suffix + ".csv")

    # Normalize debug directory if debugging is enabled.
    if debug:
        if debug_dir is None:
            debug_suffix = "_sam3_filtered_debug" if use_filtered else "_sam3_debug"
            debug_dir = input_csv.with_name(input_csv.stem + debug_suffix)

    # If requested, skip processing when outputs already exist.
    if no_overwrite:
        output_exists = output_csv.is_file()
        debug_exists = bool(debug and debug_dir is not None and debug_dir.exists())
        if output_exists or debug_exists:
            print(
                f"[INFO] Skipping input CSV {input_csv} because "
                f"output CSV and/or debug directory already exist and --no_overwrite is set."
            )
            # Optionally log this skip to the realtime progress CSV.
            if realtime_progress_csv is not None:
                realtime_progress_fieldnames = [
                    "timestamp",
                    "input_csv",
                    "cluster_dir",
                    "image_path",
                    "idx_in_cluster",
                    "description",
                    "objects",
                    "objects_not_found",
                    "duration_sec",
                    "status",
                ]
                file_exists = realtime_progress_csv.is_file()
                with realtime_progress_csv.open(
                    "a", newline="", encoding="utf-8"
                ) as prog_f:
                    prog_writer = csv.DictWriter(
                        prog_f, fieldnames=realtime_progress_fieldnames
                    )
                    if not file_exists:
                        prog_writer.writeheader()

                    # Infer cluster directory name (e.g., 'cluster_32') from the CSV path.
                    cluster_dir = ""
                    for parent in input_csv.parents:
                        name = parent.name
                        if name.startswith("cluster_"):
                            cluster_dir = name
                            break
                    if not cluster_dir:
                        cluster_dir = input_csv.parent.name

                    prog_writer.writerow(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "input_csv": str(input_csv),
                            "cluster_dir": cluster_dir,
                            "image_path": "",
                            "idx_in_cluster": "",
                            "description": "",
                            "objects": "",
                            "objects_not_found": "",
                            "duration_sec": "",
                            "status": "skipped_existing_output",
                        }
                    )
                    prog_f.flush()
            return

    # At this point we are going to regenerate outputs; clean debug directory first.
    if debug and debug_dir is not None:
        prepare_clean_directory(debug_dir)

    device = "cpu" if cpu_only or not torch.cuda.is_available() else "cuda"

    model, processor = load_model(
        device=device,
        confidence_threshold=box_threshold,
        verbose=False,
    )

    # Optionally load idx_in_cluster mapping from a sibling cluster_items.csv.
    cluster_index_map = maybe_load_cluster_index_map(input_csv.parent)

    # One row per image with aggregated missing objects.
    fieldnames = [
        "image_path",
        "idx_in_cluster",
        "description",
        "objects",
        "objects_not_found",
        "debug_image_path",
    ]

    # Open the real-time missing CSV if provided, for writing rows where SAM3
    # did not find all objects.
    realtime_missing_f = None
    realtime_missing_writer = None
    if realtime_missing_csv is not None:
        realtime_missing_fieldnames = [
            "input_csv",
            "image_path",
            "idx_in_cluster",
            "description",
            "objects",
            "objects_not_found",
        ]
        # Open in append mode so multiple CSVs can contribute to the same file.
        file_exists = realtime_missing_csv.is_file()
        realtime_missing_f = realtime_missing_csv.open("a", newline="", encoding="utf-8")
        realtime_missing_writer = csv.DictWriter(
            realtime_missing_f, fieldnames=realtime_missing_fieldnames
        )
        if not file_exists:
            realtime_missing_writer.writeheader()

    # Open the real-time progress CSV if provided, for writing one row per
    # processed image (regardless of whether objects are missing or not).
    realtime_progress_f = None
    realtime_progress_writer = None
    if realtime_progress_csv is not None:
        realtime_progress_fieldnames = [
            "timestamp",
            "input_csv",
            "cluster_dir",
            "image_path",
            "idx_in_cluster",
            "description",
            "objects",
            "objects_not_found",
            "duration_sec",
            "status",
        ]
        file_exists = realtime_progress_csv.is_file()
        realtime_progress_f = realtime_progress_csv.open(
            "a", newline="", encoding="utf-8"
        )
        realtime_progress_writer = csv.DictWriter(
            realtime_progress_f, fieldnames=realtime_progress_fieldnames
        )
        if not file_exists:
            realtime_progress_writer.writeheader()

    try:
        # Preload all rows so we can compute a deterministic progress bar
        # over the total number of samples that SAM3 will process.
        all_rows = list(parse_input_csv(input_csv, use_filtered=use_filtered))

        # Filter down to only rows that actually have any objects or stuff,
        # which are the ones that will result in SAM3 inference.
        rows_to_process: List[DetectionRow] = []
        for row in all_rows:
            objects = parse_objects_field(row.objects_raw)
            if objects or row.stuff_raw:
                rows_to_process.append(row)

        total_samples = len(rows_to_process)

        with output_csv.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            if tqdm is not None and total_samples > 0:
                row_iter = tqdm(
                    enumerate(rows_to_process, start=1),
                    total=total_samples,
                    desc=f"SAM3 {input_csv.name}",
                    unit="image",
                )
            else:
                row_iter = enumerate(rows_to_process, start=1)

            for row_idx, row in row_iter:
                objects = parse_objects_field(row.objects_raw)

                # Measure per-image processing time (loading image + SAM3 calls).
                row_start_time = time.perf_counter()

                try:
                    image_path = (
                        str((images_root / row.image_path).resolve())
                        if images_root is not None
                        else row.image_path
                    )
                    image_pil = load_image(image_path)
                except FileNotFoundError:
                    print(
                        f"[WARN] Image not found, skipping row {row_idx}: "
                        f"{image_path}"
                    )
                    continue
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] Failed to load image {row.image_path}: {e}")
                    continue

                # Set the image once for all objects (more efficient)
                inference_state = processor.set_image(image_pil)

                # Look up idx_in_cluster for this (image_path, description) if available.
                cluster_idx = cluster_index_map.get(
                    (row.image_path.strip(), row.description.strip()), ""
                )

                # Track objects that SAM3 fails to detect.
                missing_objects: List[str] = []

                debug_boxes: List[torch.Tensor] = []
                debug_masks: List[torch.Tensor] = []
                debug_labels: List[str] = []

                # Process objects (things)
                for object_name in objects:
                    boxes, scores, masks = run_sam3_for_object(
                        processor=processor,
                        image_pil=image_pil,
                        inference_state=inference_state,
                        object_name=object_name,
                    )

                    found = len(scores) > 0
                    if not found:
                        missing_objects.append(object_name)
                    elif debug and len(boxes) > 0:
                        # For visualization, keep the (top-1) box and mask for this object
                        debug_boxes.append(boxes)
                        debug_masks.append(masks)
                        debug_labels.extend(
                            f"{object_name} ({float(s):.2f})" for s in scores
                        )

                # Process stuff (regions) if available
                stuff_list = parse_objects_field(row.stuff_raw) if row.stuff_raw else []
                for stuff_name in stuff_list:
                    boxes, scores, masks = run_sam3_for_object(
                        processor=processor,
                        image_pil=image_pil,
                        inference_state=inference_state,
                        object_name=stuff_name,
                    )

                    found = len(scores) > 0
                    if not found:
                        missing_objects.append(stuff_name)
                    elif debug and len(boxes) > 0:
                        # For visualization, keep the (top-1) box and mask for this stuff
                        debug_boxes.append(boxes)
                        debug_masks.append(masks)
                        debug_labels.extend(
                            f"{stuff_name} ({float(s):.2f})" for s in scores
                        )

                # By default, no debug image is associated.
                debug_image_path_str = ""

                # Save a debug image only if at least one object/stuff was NOT found.
                if debug and debug_dir is not None and missing_objects:
                    if debug_boxes:
                        all_boxes = torch.cat(debug_boxes, dim=0)
                        all_masks = torch.cat(debug_masks, dim=0) if debug_masks else None

                        # Create segments-only image with found detections overlaid.
                        segments_image = draw_segments_only(
                            image_pil.copy(), all_boxes, debug_labels, masks=all_masks
                        )
                    else:
                        # No detections at all; just save the original image.
                        segments_image = image_pil.copy()

                    segments_filename = (
                        f"{row_idx:06d}_{Path(row.image_path).stem}_segments.jpg"
                    )
                    segments_path = debug_dir / segments_filename
                    segments_image.save(segments_path)

                    # Optionally create a color map/legend image if we have any labels.
                    if debug_labels:
                        color_map_image = create_color_map(debug_labels)
                        color_map_filename = (
                            f"{row_idx:06d}_{Path(row.image_path).stem}_colormap.jpg"
                        )
                        color_map_path = debug_dir / color_map_filename
                        color_map_image.save(color_map_path)

                    debug_image_path_str = str(segments_path)

                # Aggregate to a single row per image.
                objects_not_found_str = (
                    ". ".join(missing_objects) if missing_objects else ""
                )
                writer.writerow(
                    {
                        "image_path": row.image_path,
                        "idx_in_cluster": cluster_idx,
                        "description": row.description,
                        "objects": row.objects_raw,
                        "objects_not_found": objects_not_found_str,
                        "debug_image_path": debug_image_path_str,
                    }
                )

                # Write to real-time missing CSV if there are missing objects.
                if missing_objects and realtime_missing_writer is not None:
                    realtime_missing_writer.writerow(
                        {
                            "input_csv": str(input_csv),
                            "image_path": row.image_path,
                            "idx_in_cluster": cluster_idx,
                            "description": row.description,
                            "objects": row.objects_raw,
                            "objects_not_found": objects_not_found_str,
                        }
                    )
                    realtime_missing_f.flush()  # Flush immediately for real-time writing

                # Write to real-time progress CSV (one row per processed image).
                if realtime_progress_writer is not None:
                    duration_sec = time.perf_counter() - row_start_time

                    # Try to infer the cluster directory name (e.g., 'cluster_32')
                    # from the input CSV path. Fall back to the immediate parent dir.
                    cluster_dir = ""
                    for parent in input_csv.parents:
                        name = parent.name
                        if name.startswith("cluster_"):
                            cluster_dir = name
                            break
                    if not cluster_dir:
                        cluster_dir = input_csv.parent.name

                    realtime_progress_writer.writerow(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "input_csv": str(input_csv),
                            "cluster_dir": cluster_dir,
                            "image_path": row.image_path,
                            "idx_in_cluster": cluster_idx,
                            "description": row.description,
                            "objects": row.objects_raw,
                            "objects_not_found": objects_not_found_str,
                            "duration_sec": f"{duration_sec:.4f}",
                            "status": "processed",
                        }
                    )
                    realtime_progress_f.flush()
    finally:
        if realtime_missing_f is not None:
            realtime_missing_f.close()
        if realtime_progress_f is not None:
            realtime_progress_f.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 on a CSV of images and object lists, "
            "or batch-process a directory of '*_objects.csv' files, "
            "and produce a CSV with per-object detection results."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_csv",
        help="Path to a single input CSV with image_path, description, and objects columns.",
    )
    group.add_argument(
        "--root_dir",
        help=(
            "Root directory to recursively search for CSV files whose names end with "
            "'_objects.csv'. For each such file, SAM3 will be run and aggregate results "
            "will be written to the results directory."
        ),
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help=(
            "Path to the output CSV. "
            "Defaults to <input_stem>_sam3.csv next to the input CSV."
        ),
    )
    parser.add_argument(
        "--images_root",
        default=None,
        help=(
            "Optional root directory to prepend to image_path values from the CSV "
            "when locating image files on disk."
        ),
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for filtering SAM3 predictions.",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force computation on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, save visualization images with detections.",
    )
    parser.add_argument(
        "--debug_dir",
        default=None,
        help=(
            "Directory to store debug images. "
            "Defaults to <input_stem>_sam3_debug next to the input CSV."
        ),
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help=(
            "Directory to store aggregated results and logs when running in batch "
            "mode with --root_dir. Defaults to <root_dir>/sam3_batch_results."
        ),
    )
    parser.add_argument(
        "--debug_root_dir",
        default=None,
        help=(
            "Root directory to store debug images when running in batch mode with "
            "--root_dir. Per-CSV subdirectories will be created inside this directory. "
            "If not provided, debug images are stored next to each input CSV."
        ),
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help=(
            "If set, do not overwrite existing per-CSV SAM3 output files or debug "
            "directories. Any input CSV whose outputs already exist will be skipped."
        ),
    )
    parser.add_argument(
        "--np",
        action="store_true",
        help=(
            "When running with --root_dir, look for caption CSVs produced by the "
            "noun-phrase pipeline, i.e. files ending with '_objects_np.csv', "
            "instead of the default '*_objects.csv' pattern."
        ),
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "When running with --root_dir, look for LLM-caption CSVs whose names end "
            "with '_objects_llm.csv' instead of the default '*_objects.csv' pattern."
        ),
    )
    parser.add_argument(
        "--filtered",
        action="store_true",
        help=(
            "When set, use the 'filtered_by_llm' column from the CSV as the objects "
            "source instead of the 'objects' column. Output files will include "
            "'_filtered' in their names (e.g., '_sam3_filtered.csv')."
        ),
    )
    parser.add_argument(
        "--realtime_missing_csv",
        default=None,
        help=(
            "Path to the real-time missing summary CSV file. This file is updated "
            "immediately whenever SAM3 fails to find all objects in a row. "
            "If not provided, defaults to 'sam3_realtime_missing[_filtered].csv' "
            "in the results directory (batch mode) or next to the input CSV (single mode)."
        ),
    )
    parser.add_argument(
        "--realtime_progress_csv",
        default=None,
        help=(
            "Path to the real-time per-image progress CSV file. This file is updated "
            "immediately after each image is processed (one row per image). "
            "If not provided, defaults to 'sam3_realtime_progress[_filtered].csv' "
            "in the results directory (batch mode) or next to the input CSV (single mode)."
        ),
    )

    args = parser.parse_args()

    # Batch mode: recursively process all '*_objects.csv' files under root_dir.
    if args.root_dir:
        root_dir = Path(args.root_dir).resolve()
        results_dir = (
            Path(args.results_dir).resolve()
            if args.results_dir
            else root_dir / "sam3_batch_results"
        )
        debug_root_dir = (
            Path(args.debug_root_dir).resolve()
            if args.debug_root_dir
            else None
        )
        images_root = (
            Path(args.images_root).resolve() if args.images_root else None
        )

        run_batch(
            root_dir=root_dir,
            results_dir=results_dir,
            images_root=images_root,
            box_threshold=args.box_threshold,
            cpu_only=args.cpu_only,
            debug=args.debug,
            debug_root_dir=debug_root_dir,
            no_overwrite=args.no_overwrite,
            use_np=args.np,
            use_llm=args.llm,
            use_filtered=args.filtered,
            realtime_missing_csv=Path(args.realtime_missing_csv).resolve()
            if args.realtime_missing_csv
            else None,
        )
    else:
        # Single-CSV mode: optionally wait for the input CSV to be created.
        input_csv_path = Path(args.input_csv).resolve()
        if not input_csv_path.is_file():
            timeout_sec = 300
            found = wait_for_file(input_csv_path, timeout_sec=timeout_sec)
            if not found:
                print(
                    f"Input CSV not found: {input_csv_path}. "
                    f"Waited {timeout_sec} seconds for it to appear; exiting."
                )
                sys.exit(1)

        # Determine realtime missing CSV path for single-CSV mode.
        if args.realtime_missing_csv:
            realtime_missing_path = Path(args.realtime_missing_csv).resolve()
        else:
            suffix = "_filtered" if args.filtered else ""
            realtime_missing_path = input_csv_path.with_name(
                f"sam3_realtime_missing{suffix}.csv"
            )

        # Determine realtime progress CSV path for single-CSV mode.
        if args.realtime_progress_csv:
            realtime_progress_path = Path(args.realtime_progress_csv).resolve()
        else:
            suffix = "_filtered" if args.filtered else ""
            realtime_progress_path = input_csv_path.with_name(
                f"sam3_realtime_progress{suffix}.csv"
            )

        run(
            input_csv=input_csv_path,
            output_csv=Path(args.output_csv).resolve() if args.output_csv else None,
            images_root=Path(args.images_root).resolve()
            if args.images_root
            else None,
            box_threshold=args.box_threshold,
            cpu_only=args.cpu_only,
            debug=args.debug,
            debug_dir=Path(args.debug_dir).resolve() if args.debug_dir else None,
            no_overwrite=args.no_overwrite,
            use_filtered=args.filtered,
            realtime_missing_csv=realtime_missing_path,
            realtime_progress_csv=realtime_progress_path,
        )


def run_batch(
    root_dir: Path,
    results_dir: Path,
    images_root: Optional[Path],
    box_threshold: float,
    cpu_only: bool,
    debug: bool,
    debug_root_dir: Optional[Path],
    no_overwrite: bool = False,
    use_np: bool = False,
    use_llm: bool = False,
    use_filtered: bool = False,
    realtime_missing_csv: Optional[Path] = None,
    realtime_progress_csv: Optional[Path] = None,
) -> None:
    """
    Recursively search root_dir for CSV files that end with '_objects.csv',
    run SAM3 on each, and aggregate cases where SAM3 could not find the
    requested object.

    The function writes:
    - Per-CSV SAM3 outputs next to each input CSV (same behavior as single mode),
      one row per image with aggregated missing objects.
    - A detailed CSV of all failed detections in results_dir.
    - A summary CSV with counts per CSV in results_dir.
    - A text log file in results_dir that includes the command line used.
    """
    root_dir = root_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if debug and debug_root_dir is not None:
        debug_root_dir.mkdir(parents=True, exist_ok=True)

    # Discover all relevant CSV files, waiting up to 5 minutes for them to appear.
    # Patterns:
    # - '*_objects.csv' (default, standard objects CSV)
    # - '*_objects_np.csv' (noun-phrase pipeline) when --np is set
    # - '*_objects_llm_v2.csv' (LLM-expanded objects, v2 naming) when --llm is set
    timeout_sec = 300
    poll_interval = 5.0
    start_time = time.time()
    csv_files: List[Path] = []
    # Resolve pattern priority: --llm > --np > default.
    if use_llm:
        # Newer LLM extraction script writes '*_objects_llm_v2.csv'.
        pattern = "*_objects_llm_v2.csv"
    elif use_np:
        pattern = "*_objects_np.csv"
    else:
        pattern = "*_objects.csv"
    while True:
        csv_files = sorted(root_dir.rglob(pattern))
        if csv_files:
            break
        if time.time() - start_time >= timeout_sec:
            print(
                f"No '{pattern}' files found under {root_dir} after "
                f"{timeout_sec} seconds; exiting."
            )
            return
        time.sleep(poll_interval)

    # Use appropriate file names based on filtered mode.
    suffix = "_filtered" if use_filtered else ""
    detailed_path = results_dir / f"sam3_missing_detections{suffix}.csv"
    summary_path = results_dir / f"sam3_missing_summary{suffix}.csv"
    log_path = results_dir / f"sam3_batch_log{suffix}.txt"

    # Determine realtime missing CSV path.
    if realtime_missing_csv is not None:
        realtime_missing_path = realtime_missing_csv
    else:
        realtime_missing_path = results_dir / f"sam3_realtime_missing{suffix}.csv"

    # Determine realtime progress CSV path.
    if realtime_progress_csv is not None:
        realtime_progress_path = realtime_progress_csv
    else:
        realtime_progress_path = results_dir / f"sam3_realtime_progress{suffix}.csv"

    # Clear the realtime CSVs to start fresh (they are opened in append mode by run()).
    if realtime_missing_path.exists():
        realtime_missing_path.unlink()
    if realtime_progress_path.exists():
        realtime_progress_path.unlink()

    print(f"Realtime missing CSV will be saved to: {realtime_missing_path}")
    print(f"Realtime progress CSV will be saved to: {realtime_progress_path}")

    # Prepare logging.
    command_line = " ".join(shlex.quote(arg) for arg in sys.argv)
    start_time = datetime.now().isoformat(timespec="seconds")

    with log_path.open("a", encoding="utf-8") as log_f, \
         detailed_path.open("w", newline="", encoding="utf-8") as det_f:
        log_f.write("\n")
        log_f.write(f"=== SAM3 batch run @ {start_time} ===\n")
        log_f.write(f"Command: {command_line}\n")
        log_f.write(f"Root dir: {root_dir}\n")
        log_f.write(f"Results dir: {results_dir}\n")
        if debug and debug_root_dir is not None:
            log_f.write(f"Debug root dir: {debug_root_dir}\n")
        if use_filtered:
            log_f.write("Mode: Using 'filtered_by_llm' column as objects source\n")
        log_f.write(f"Realtime missing CSV: {realtime_missing_path}\n")
        log_f.write(f"Realtime progress CSV: {realtime_progress_path}\n")
        log_f.write(f"Found {len(csv_files)} CSV files matching '{pattern}'.\n")

        detailed_fieldnames = [
            "input_csv",
            "output_csv",
            "image_path",
            "idx_in_cluster",
            "description",
            "objects",
            "object_name",
            "debug_image_path",
        ]
        detailed_writer = csv.DictWriter(det_f, fieldnames=detailed_fieldnames)
        detailed_writer.writeheader()

        # Track per-CSV counts for the summary.
        summary_counts: dict[str, dict[str, float]] = {}

        # Track which cluster directories we've already checked for their
        # companion cluster_items CSV to avoid redundant waits.
        checked_cluster_dirs: set[Path] = set()

        for idx, csv_path in enumerate(csv_files, start=1):
            csv_path = csv_path.resolve()
            log_f.write(f"[{idx}/{len(csv_files)}] Processing CSV: {csv_path}\n")
            log_f.flush()

            # When using LLM-based object CSVs, ensure that for each cluster_* directory
            # we also have the corresponding 'cluster_items_objects_llm_v2.csv'.
            # If it does not exist yet, wait up to 5 minutes for it to appear; if it
            # still does not show up, exit with a meaningful error.
            if use_llm:
                cluster_dir: Optional[Path] = None
                for parent in csv_path.parents:
                    if parent.name.startswith("cluster_"):
                        cluster_dir = parent
                        break
                if cluster_dir is None:
                    cluster_dir = csv_path.parent

                if cluster_dir not in checked_cluster_dirs:
                    cluster_items_name = "cluster_items_objects_llm_v2.csv"
                    cluster_items_path = cluster_dir / cluster_items_name
                    if not cluster_items_path.is_file():
                        timeout_sec = 300
                        poll_interval = 5.0
                        print(
                            f"[INFO] Waiting up to {timeout_sec} seconds for "
                            f"'{cluster_items_name}' in cluster directory "
                            f"{cluster_dir}..."
                        )
                        found = wait_for_file(
                            cluster_items_path,
                            timeout_sec=timeout_sec,
                            poll_interval=poll_interval,
                        )
                        if not found:
                            print(
                                f"[ERROR] Expected '{cluster_items_name}' not found in "
                                f"{cluster_dir} after waiting {timeout_sec} seconds. "
                                "Exiting."
                            )
                            sys.exit(1)

                    checked_cluster_dirs.add(cluster_dir)

            # Determine per-CSV debug directory in batch mode.
            per_csv_debug_dir: Optional[Path] = None
            debug_suffix_dir = "_sam3_filtered_debug" if use_filtered else "_sam3_debug"
            if debug:
                if debug_root_dir is not None:
                    try:
                        rel_parent = csv_path.parent.relative_to(root_dir)
                    except ValueError:
                        # If for some reason the CSV is not under root_dir, just use its parent name.
                        rel_parent = Path(csv_path.parent.name)
                    per_csv_debug_dir = (
                        debug_root_dir / rel_parent / f"{csv_path.stem}{debug_suffix_dir}"
                    )
                else:
                    # Default: store debug images next to each CSV, similar to single-CSV mode.
                    per_csv_debug_dir = csv_path.with_name(
                        f"{csv_path.stem}{debug_suffix_dir}"
                    )

            # Run SAM3 on this CSV (per-image results written next to the CSV).
            run(
                input_csv=csv_path,
                output_csv=None,
                images_root=images_root,
                box_threshold=box_threshold,
                cpu_only=cpu_only,
                debug=debug,
                debug_dir=per_csv_debug_dir,
                no_overwrite=no_overwrite,
                use_filtered=use_filtered,
                realtime_missing_csv=realtime_missing_path,
                realtime_progress_csv=realtime_progress_path,
            )

            output_suffix = "_sam3_filtered.csv" if use_filtered else "_sam3.csv"
            output_csv = csv_path.with_name(csv_path.stem + output_suffix)
            if not output_csv.is_file():
                log_f.write(
                    f"  [WARN] Expected output CSV not found, skipping aggregation: "
                    f"{output_csv}\n"
                )
                log_f.flush()
                continue

            # Aggregate failures from this CSV.
            total_objects = 0
            missing_objects = 0

            with output_csv.open("r", newline="", encoding="utf-8") as f_out:
                reader = csv.DictReader(f_out)
                for out_row in reader:
                    # Each row represents one image; count total and missing objects.
                    objects_list = parse_objects_field(out_row.get("objects", ""))
                    objects_not_found_list = parse_objects_field(
                        out_row.get("objects_not_found", "")
                    )
                    total_objects += len(objects_list)
                    missing_objects += len(objects_not_found_list)

                    # Emit one detailed row per missing object, if any.
                    for missing_name in objects_not_found_list:
                        detailed_writer.writerow(
                            {
                                "input_csv": str(csv_path),
                                "output_csv": str(output_csv),
                                "image_path": out_row.get("image_path", ""),
                                "idx_in_cluster": out_row.get("idx_in_cluster", ""),
                                "description": out_row.get("description", ""),
                                "objects": out_row.get("objects", ""),
                                "object_name": missing_name,
                                "debug_image_path": out_row.get(
                                    "debug_image_path", ""
                                ),
                            }
                        )

            if total_objects > 0:
                missing_rate = missing_objects / float(total_objects)
            else:
                missing_rate = 0.0

            summary_counts[str(csv_path)] = {
                "total_objects": float(total_objects),
                "missing_objects": float(missing_objects),
                "missing_rate": missing_rate,
            }

            log_f.write(
                f"  Total objects: {total_objects}, "
                f"missing: {missing_objects} "
                f"({missing_rate:.2%})\n"
            )
            log_f.flush()

    # Write per-CSV summary CSV.
    with summary_path.open("w", newline="", encoding="utf-8") as sum_f:
        summary_fieldnames = [
            "input_csv",
            "total_objects",
            "missing_objects",
            "missing_rate",
        ]
        summary_writer = csv.DictWriter(sum_f, fieldnames=summary_fieldnames)
        summary_writer.writeheader()
        for input_csv_str, stats in sorted(summary_counts.items()):
            summary_writer.writerow(
                {
                    "input_csv": input_csv_str,
                    "total_objects": int(stats["total_objects"]),
                    "missing_objects": int(stats["missing_objects"]),
                    "missing_rate": f"{stats['missing_rate']:.6f}",
                }
            )



if __name__ == "__main__":
    main()

