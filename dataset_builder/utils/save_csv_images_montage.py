import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pandas as pd


def resolve_image_path(images_root: str, rel_or_abs_path: str) -> str:
    p = str(rel_or_abs_path)
    if os.path.isabs(p):
        return p
    if images_root:
        return os.path.join(images_root, p)
    return p


def read_image_paths_from_csv(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise ValueError(f"CSV missing 'image_path' column: {csv_path}")
    return df["image_path"].astype(str).tolist()


def parse_cluster_id_from_filename(csv_path: str) -> Optional[int]:
    """Extract cluster id from filename like cluster_123.csv"""
    base = os.path.splitext(os.path.basename(csv_path))[0]
    parts = base.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-1])
    except Exception:
        return None


def load_summary_mapping(summary_csv_path: str) -> Dict[int, Dict[str, str]]:
    """Load mapping from cluster_id -> {score, explanation, city_id, cluster_csv}.

    Only score and explanation are required for our overlay.
    """
    mapping: Dict[int, Dict[str, str]] = {}
    df = pd.read_csv(summary_csv_path)
    if "cluster_id" not in df.columns:
        raise ValueError("Summary CSV must contain 'cluster_id' column")
    # Optional columns; handle missing gracefully
    for _, row in df.iterrows():
        try:
            cid = int(row["cluster_id"])  # type: ignore
        except Exception:
            continue
        score_val = row["score"] if "score" in df.columns else ""
        explanation_val = row["explanation"] if "explanation" in df.columns else ""
        city_val = row["city_id"] if "city_id" in df.columns else ""
        cluster_csv_val = row["cluster_csv"] if "cluster_csv" in df.columns else ""
        mapping[cid] = {
            "score": str(score_val) if not pd.isna(score_val) else "",
            "explanation": str(explanation_val) if not pd.isna(explanation_val) else "",
            "city_id": str(city_val) if not pd.isna(city_val) else "",
            "cluster_csv": str(cluster_csv_val) if not pd.isna(cluster_csv_val) else "",
        }
    return mapping


def wrap_text_to_width(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Wrap text so that each line's rendered width <= max_width."""
    if not text:
        return []
    words = str(text).split()
    if not words:
        return []
    lines: List[str] = []
    current = words[0]
    for w in words[1:]:
        test = current + " " + w
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def normalize_score_subdir(value: str) -> Optional[str]:
    s = str(value).strip()
    if not s:
        return None
    try:
        v = float(s)
        if math.isfinite(v):
            if abs(v - round(v)) < 1e-9:
                s = str(int(round(v)))
            else:
                s = f"{v}".rstrip("0").rstrip(".")
    except Exception:
        pass
    s = s.replace(os.sep, "_")
    s = s.replace(" ", "_")
    return s or None


def make_montage(
    image_paths: List[str],
    images_root: str,
    out_path: str,
    thumb_size: Tuple[int, int] = (256, 256),
    cols: int = 5,
    margin: int = 8,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    add_index: bool = False,
    caption_title: Optional[str] = None,
    caption_body: Optional[str] = None,
    font_path: Optional[str] = None,
    font_size: int = 16,
) -> None:
    images: List[Image.Image] = []
    resolved_paths: List[str] = []
    for i, rel in enumerate(image_paths):
        full = resolve_image_path(images_root, rel)
        if not os.path.exists(full):
            continue
        try:
            img = Image.open(full).convert("RGB")
        except Exception:
            continue
        # Fit into cell while preserving aspect ratio
        img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        images.append(img)
        resolved_paths.append(full)

    if not images:
        raise SystemExit("No valid images to render in montage.")

    cell_w, cell_h = thumb_size
    rows = math.ceil(len(images) / max(1, cols))
    canvas_w = cols * cell_w + (cols + 1) * margin
    grid_h = rows * cell_h + (rows + 1) * margin

    # Prepare font(s)
    index_font: Optional[ImageFont.ImageFont] = None
    caption_font: Optional[ImageFont.ImageFont] = None

    # Try to load a TrueType font if provided, else fall back to default
    if font_path:
        try:
            caption_font = ImageFont.truetype(font_path, font_size)
            index_font = caption_font
        except Exception:
            caption_font = None
            index_font = None
    if caption_font is None or index_font is None:
        try:
            fallback = ImageFont.load_default()
            if caption_font is None:
                caption_font = fallback
            if index_font is None:
                index_font = fallback
        except Exception:
            caption_font = None
            index_font = None

    # Compute caption area height if needed
    caption_height = 0
    caption_padding_y = 8
    caption_gap = 6  # gap between title and body
    caption_bg = (240, 240, 240)
    caption_lines_title: List[str] = []
    caption_lines_body: List[str] = []

    if (caption_title or caption_body) and caption_font is not None:
        # Use a temporary image to measure text
        temp_img = Image.new("RGB", (canvas_w, 10), bg_color)
        temp_draw = ImageDraw.Draw(temp_img)
        max_text_width = canvas_w - 2 * margin
        if caption_title:
            caption_lines_title = wrap_text_to_width(caption_title, temp_draw, caption_font, max_text_width)
        if caption_body:
            caption_lines_body = wrap_text_to_width(caption_body, temp_draw, caption_font, max_text_width)

        # Accumulate height: padding + lines + gap + padding
        y = caption_padding_y
        prev_line = None
        for line in caption_lines_title:
            bbox = temp_draw.textbbox((0, 0), line, font=caption_font)
            y += (bbox[3] - bbox[1])
        if caption_lines_title and caption_lines_body:
            y += caption_gap
        for line in caption_lines_body:
            bbox = temp_draw.textbbox((0, 0), line, font=caption_font)
            y += (bbox[3] - bbox[1])
        y += caption_padding_y
        caption_height = y

    canvas_h = grid_h + caption_height
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Draw caption area if present
    offset_y = 0
    if caption_height > 0:
        # Caption background
        draw.rectangle([(0, 0), (canvas_w, caption_height)], fill=caption_bg)
        y_text = caption_padding_y
        x_text = margin
        # Title lines
        for line in caption_lines_title:
            draw.text((x_text, y_text), line, fill=(0, 0, 0), font=caption_font)
            bbox = draw.textbbox((0, 0), line, font=caption_font)
            y_text += (bbox[3] - bbox[1])
        if caption_lines_title and caption_lines_body:
            y_text += caption_gap
        # Body lines
        for line in caption_lines_body:
            draw.text((x_text, y_text), line, fill=(0, 0, 0), font=caption_font)
            bbox = draw.textbbox((0, 0), line, font=caption_font)
            y_text += (bbox[3] - bbox[1])
        offset_y = caption_height

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x0 = margin + c * (cell_w + margin)
        y0 = offset_y + margin + r * (cell_h + margin)
        # center image in cell
        paste_x = x0 + (cell_w - img.width) // 2
        paste_y = y0 + (cell_h - img.height) // 2
        canvas.paste(img, (paste_x, paste_y))
        if add_index and index_font is not None:
            draw.text((x0 + 4, y0 + 4), str(idx + 1), fill=(0, 0, 0), font=index_font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-CSV image montages from image_path lists.")
    parser.add_argument(
        "--csv-dir",
        type=str,
        required=True,
        help="Directory containing CSV files (each with 'image_path' column)",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help=(
            "Optional summary CSV with columns including 'cluster_id', 'score', and 'explanation'. "
            "If provided, the montage will include an overlaid caption with score and explanation."
        ),
    )
    parser.add_argument(
        "--images-root",
        type=str,
        required=True,
        help="Filesystem prefix to resolve relative image paths (e.g., /mnt/d/data/gsv_cities)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write output PNG montages",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns in the montage grid",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Thumbnail width height in pixels (e.g., 256 256)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=8,
        help="Margin in pixels between cells and around the canvas",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional limit per montage (0 = no limit)",
    )
    parser.add_argument(
        "--add-index",
        action="store_true",
        help="Overlay small index number on each cell",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default="",
        help="Optional path to a .ttf/.otf font file used for captions and indices",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=16,
        help="Font size for captions (used if --font-path is provided; otherwise default font size is used)",
    )

    args = parser.parse_args()

    csv_files = [
        os.path.join(args.csv_dir, f)
        for f in sorted(os.listdir(args.csv_dir))
        if f.lower().endswith(".csv")
    ]
    if not csv_files:
        raise SystemExit("No CSV files found in --csv-dir.")

    summary_map: Dict[int, Dict[str, str]] = {}
    if args.summary_csv:
        try:
            summary_map = load_summary_mapping(args.summary_csv)
        except Exception as e:
            print(f"Warning: could not load summary CSV '{args.summary_csv}': {e}")

    for csv_path in csv_files:
        try:
            rel_paths = read_image_paths_from_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
            continue

        if args.max_images and len(rel_paths) > args.max_images:
            rel_paths = rel_paths[: args.max_images]

        base = os.path.splitext(os.path.basename(csv_path))[0]

        # Prepare caption if summary is available
        caption_title: Optional[str] = None
        caption_body: Optional[str] = None
        score_for_dir: Optional[str] = None
        cluster_id = parse_cluster_id_from_filename(csv_path)
        if cluster_id is not None and summary_map:
            row = summary_map.get(cluster_id)
            if row:
                score = row.get("score", "")
                city_id = row.get("city_id", "")
                if score:
                    score_for_dir = normalize_score_subdir(score)
                    if city_id:
                        caption_title = f"{city_id} — Cluster {cluster_id} — Score: {score}"
                    else:
                        caption_title = f"Cluster {cluster_id} — Score: {score}"
                else:
                    caption_title = f"Cluster {cluster_id}"
                caption_body = row.get("explanation", "")

        # Decide output path: group by score directory if available
        if score_for_dir:
            out_png = os.path.join(args.output_dir, score_for_dir, f"{base}.png")
        else:
            out_png = os.path.join(args.output_dir, f"{base}.png")
        try:
            make_montage(
                rel_paths,
                images_root=args.images_root,
                out_path=out_png,
                thumb_size=(int(args.thumb_size[0]), int(args.thumb_size[1])),
                cols=int(args.cols),
                margin=int(args.margin),
                add_index=bool(args.add_index),
                caption_title=caption_title,
                caption_body=caption_body,
                font_path=args.font_path if getattr(args, "font_path", "") else None,
                font_size=int(args.font_size) if getattr(args, "font_size", 0) else 16,
            )
            print(f"Wrote {out_png}")
        except Exception as e:
            print(f"Failed to write montage for {csv_path}: {e}")


if __name__ == "__main__":
    main()


