import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Prefer the installed GroundingDINO package (with its compiled C++ extension).
# This avoids issues where importing directly from the source repo bypasses the
# compiled `_C` extension and leads to runtime errors.
try:
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
except ModuleNotFoundError:
    # Fallback: attempt to use a sibling GroundingDINO repo if the package
    # is not installed in the current environment.
    _THIS_FILE = Path(__file__).resolve()
    _ROOT_DIR = _THIS_FILE.parents[2]  # /mnt/d/dan/git_projects
    _GROUNDING_DINO_DIR = _ROOT_DIR / "GroundingDINO"
    if not _GROUNDING_DINO_DIR.is_dir():
        raise

    # Add the repo itself to sys.path and rely on its installed build
    # (e.g. `pip install .` having produced a wheel with `_C`).
    sys.path.insert(0, str(_GROUNDING_DINO_DIR))

    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict


@dataclass
class DetectionRow:
    image_path: str
    description: str
    objects_raw: str


def parse_input_csv(csv_path: Path) -> Iterator[DetectionRow]:
    """
    Parse an input CSV with at least the following columns:
    - image_path
    - description
    - objects
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        required = {"image_path", "description", "objects"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing required columns in CSV {csv_path}: {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            image_path = row.get("image_path") or ""
            description = row.get("description") or ""
            objects_raw = row.get("objects") or ""
            if not image_path:
                # Skip malformed rows instead of failing the whole job
                continue
            yield DetectionRow(
                image_path=str(image_path),
                description=str(description),
                objects_raw=str(objects_raw),
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
    """Load and preprocess an image for GroundingDINO."""
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)  # 3, H, W
    return image_pil, image_tensor


def load_model(
    model_config_path: Path,
    model_checkpoint_path: Path,
    device: str,
) -> torch.nn.Module:
    """Load a GroundingDINO model from config and checkpoint."""
    args = SLConfig.fromfile(str(model_config_path))
    args.device = device
    model = build_model(args)

    checkpoint = torch.load(str(model_checkpoint_path), map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"Loaded GroundingDINO checkpoint with result: {load_res}")

    model.eval()
    model = model.to(device)
    return model


def run_grounding_for_object(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    object_name: str,
    box_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run GroundingDINO for a single object name on a preprocessed image.

    Returns:
        boxes: (N, 4) tensor in cxcywh normalized coordinates.
        scores: (N,) tensor of confidence scores in [0, 1].
    """
    caption = object_name.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, 256)
    boxes = outputs["pred_boxes"][0]  # (num_queries, 4)

    # Filter by maximum logit score per query
    max_per_query = logits.max(dim=1)[0]
    mask = max_per_query > box_threshold

    boxes_filt = boxes[mask].detach().cpu()
    scores_filt = max_per_query[mask].detach().cpu()
    return boxes_filt, scores_filt


def draw_boxes_on_image(
    image_pil: Image.Image,
    boxes: torch.Tensor,
    labels: List[str],
) -> Image.Image:
    """
    Draw bounding boxes and labels onto a PIL image.

    Args:
        image_pil: Original image (will be modified in-place).
        boxes: (N, 4) tensor in cxcywh format normalized to [0, 1].
        labels: List of N strings.
    """
    if len(boxes) == 0:
        return image_pil

    W, H = image_pil.size
    draw = ImageDraw.Draw(image_pil)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box, label in zip(boxes, labels):
        # cxcywh (normalized) -> xyxy (absolute)
        box = box * torch.tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = box.tolist()

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        if label and font is not None:
            text = str(label)
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((x0, y0), text, font=font)
            else:
                w, h = draw.textsize(text, font=font)
                bbox = (x0, y0, x0 + w, y0 + h)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), text, fill="white", font=font)

    return image_pil


def run(
    input_csv: Path,
    output_csv: Optional[Path],
    images_root: Optional[Path],
    model_config_path: Path,
    model_checkpoint_path: Path,
    box_threshold: float,
    cpu_only: bool,
    debug: bool,
    debug_dir: Optional[Path],
) -> None:
    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_grounding.csv")

    if debug:
        if debug_dir is None:
            debug_dir = input_csv.with_name(input_csv.stem + "_grounding_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if cpu_only or not torch.cuda.is_available() else "cuda"
    print(f"Using device: {device}")

    model = load_model(model_config_path, model_checkpoint_path, device=device)

    fieldnames = [
        "image_path",
        "description",
        "objects",
        "object_name",
        "found",
        "confidence",
        "debug_image_path",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row_idx, row in enumerate(parse_input_csv(input_csv), start=1):
            objects = parse_objects_field(row.objects_raw)
            if not objects:
                continue

            try:
                image_path = (
                    str((images_root / row.image_path).resolve())
                    if images_root is not None
                    else row.image_path
                )
                image_pil, image_tensor = load_image(image_path)
            except FileNotFoundError:
                print(
                    f"[WARN] Image not found, skipping row {row_idx}: "
                    f"{image_path}"
                )
                continue
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to load image {row.image_path}: {e}")
                continue

            per_object_results = []
            debug_boxes: List[torch.Tensor] = []
            debug_labels: List[str] = []

            for object_name in objects:
                boxes, scores = run_grounding_for_object(
                    model=model,
                    image_tensor=image_tensor,
                    object_name=object_name,
                    box_threshold=box_threshold,
                )

                found = len(scores) > 0
                confidence = float(scores.max().item()) if found else 0.0

                if debug and len(boxes) > 0:
                    # For visualization, keep all boxes for this object
                    debug_boxes.append(boxes)
                    debug_labels.extend(
                        f"{object_name} ({float(s):.2f})" for s in scores
                    )

                per_object_results.append(
                    {
                        "image_path": row.image_path,
                        "description": row.description,
                        "objects": row.objects_raw,
                        "object_name": object_name,
                        "found": int(found),
                        "confidence": confidence,
                        "debug_image_path": "",
                    }
                )

            debug_image_path_str = ""
            if debug and debug_dir is not None and debug_boxes:
                all_boxes = torch.cat(debug_boxes, dim=0)
                debug_image = draw_boxes_on_image(image_pil.copy(), all_boxes, debug_labels)
                debug_filename = f"{row_idx:06d}_{Path(row.image_path).stem}_grounding.jpg"
                debug_path = debug_dir / debug_filename
                debug_image.save(debug_path)
                debug_image_path_str = str(debug_path)

            for result in per_object_results:
                result["debug_image_path"] = debug_image_path_str
                writer.writerow(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run GroundingDINO on a CSV of images and object lists, "
            "and produce a CSV with per-object detection results."
        )
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the input CSV with image_path, description, and objects columns.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help=(
            "Path to the output CSV. "
            "Defaults to <input_stem>_grounding.csv next to the input CSV."
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
        "--config_file",
        required=True,
        help="Path to the GroundingDINO config file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to the GroundingDINO checkpoint file.",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.3,
        help="Box threshold for filtering GroundingDINO predictions.",
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
            "Defaults to <input_stem>_grounding_debug next to the input CSV."
        ),
    )

    args = parser.parse_args()

    run(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        images_root=Path(args.images_root).resolve()
        if args.images_root
        else None,
        model_config_path=Path(args.config_file),
        model_checkpoint_path=Path(args.checkpoint_path),
        box_threshold=args.box_threshold,
        cpu_only=args.cpu_only,
        debug=args.debug,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
    )


if __name__ == "__main__":
    main()


