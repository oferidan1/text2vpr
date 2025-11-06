import argparse
import math
import os
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def resolve_image_path(images_root: str, rel_or_abs_path: str) -> str:
    p = str(rel_or_abs_path)
    if os.path.isabs(p):
        return p
    if images_root:
        return os.path.join(images_root, p)
    return p


def wrap_text_to_width(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> List[str]:
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


def load_font(font_path: Optional[str], font_size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        # As a last resort, try a very small default bitmap
        return ImageFont.load_default()


def annotate_image_with_caption(
    image: Image.Image,
    title_lines: List[str],
    body_lines: List[str],
    font: ImageFont.ImageFont,
    margin: int = 16,
    caption_bg: Tuple[int, int, int] = (240, 240, 240),
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    extra_lines: Optional[List[str]] = None,
) -> Image.Image:
    # Compute caption height using text metrics
    temp = Image.new("RGB", (image.width, 10), bg_color)
    draw = ImageDraw.Draw(temp)
    y = margin
    line_gap = 6

    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1])
    if title_lines and body_lines:
        y += line_gap
    for line in body_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1])
    # Add a small gap and account for extra lines (e.g., original description)
    if extra_lines:
        y += line_gap
        for line in extra_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            y += (bbox[3] - bbox[1])
    y += margin
    caption_h = max(y, 0)

    # New canvas with space for caption at the bottom
    canvas_h = image.height + caption_h
    canvas_w = image.width
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    canvas.paste(image, (0, caption_h))

    draw_canvas = ImageDraw.Draw(canvas)
    # Caption background
    draw_canvas.rectangle([(0, 0), (canvas_w, caption_h)], fill=caption_bg)
    # Draw text
    x_text = margin
    y_text = margin
    for line in title_lines:
        draw_canvas.text((x_text, y_text), line, fill=(0, 0, 0), font=font)
        bbox = draw_canvas.textbbox((0, 0), line, font=font)
        y_text += (bbox[3] - bbox[1])
    if title_lines and body_lines:
        y_text += line_gap
    for line in body_lines:
        draw_canvas.text((x_text, y_text), line, fill=(0, 0, 0), font=font)
        bbox = draw_canvas.textbbox((0, 0), line, font=font)
        y_text += (bbox[3] - bbox[1])
    if extra_lines:
        y_text += line_gap
        for line in extra_lines:
            draw_canvas.text((x_text, y_text), line, fill=(0, 0, 0), font=font)
            bbox = draw_canvas.textbbox((0, 0), line, font=font)
            y_text += (bbox[3] - bbox[1])

    return canvas


def process_csv(
    csv_path: str,
    output_dir: str,
    images_root: str = "",
    font_path: Optional[str] = None,
    font_size: int = 18,
    max_rows: int = 0,
) -> None:
    # Expect columns: city_id,image_path,objects_detected,llm_score,llm_explanation,omit_list,suggested_description,detected_in_text_ratio,text_in_detected_ratio,overlay_path
    df = pd.read_csv(csv_path)
    required = ["image_path", "llm_score", "objects_detected", "llm_explanation"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"CSV missing required column '{col}': {csv_path}")

    os.makedirs(output_dir, exist_ok=True)
    font = load_font(font_path, font_size)

    # Iterate
    total = len(df)
    if max_rows and total > max_rows:
        total = max_rows

    for idx in range(total):
        row = df.iloc[idx]
        image_path_val = str(row.get("image_path", ""))
        score_val = row.get("llm_score", "")
        objects_val = row.get("objects_detected", "")
        expl_val = row.get("llm_explanation", "")
        # Prefer newly-added column from judge output; fallback to legacy names if absent
        orig_desc = row.get("original_description", "")
        if (orig_desc is None or str(orig_desc).strip() == "") and "description" in df.columns:
            orig_desc = row.get("description", "")
        if (orig_desc is None or str(orig_desc).strip() == "") and "suggested_description" in df.columns:
            orig_desc = row.get("suggested_description", "")

        # Resolve score subdirectory
        score_dir_name = normalize_score_subdir(str(score_val)) or "score_unknown"
        out_dir_for_score = os.path.join(output_dir, score_dir_name)
        os.makedirs(out_dir_for_score, exist_ok=True)

        # Resolve image path
        image_path = resolve_image_path(images_root, image_path_val)
        if not os.path.exists(image_path):
            print(f"[{idx+1}] Missing image, skipping: {image_path}")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[{idx+1}] Failed to open image '{image_path}': {e}")
            continue

        # Prepare caption text
        temp_canvas = Image.new("RGB", (img.width, 10), (255, 255, 255))
        temp_draw = ImageDraw.Draw(temp_canvas)
        max_text_width = max(64, img.width - 32)

        title_str = f"Objects: {objects_val}" if str(objects_val).strip() else "Objects: (none)"
        body_str = f"Explanation: {expl_val}" if str(expl_val).strip() else "Explanation: (none)"
        original_str = f"Original: {orig_desc}" if str(orig_desc).strip() else ""

        title_lines = wrap_text_to_width(title_str, temp_draw, font, max_text_width)
        body_lines = wrap_text_to_width(body_str, temp_draw, font, max_text_width)
        extra_lines = wrap_text_to_width(original_str, temp_draw, font, max_text_width) if original_str else []

        # Compose annotated image (caption on top, image below)
        annotated = annotate_image_with_caption(
            img,
            title_lines=title_lines,
            body_lines=body_lines,
            font=font,
            margin=16,
            extra_lines=extra_lines,
        )

        base_name = os.path.basename(image_path)
        out_name = f"{idx:06d}_{base_name}"
        out_path = os.path.join(out_dir_for_score, out_name)

        try:
            annotated.save(out_path)
            print(f"[{idx+1}] Wrote {out_path}")
        except Exception as e:
            print(f"[{idx+1}] Failed to save '{out_path}': {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a quality CSV and, for each row, save the image annotated with "
            "detected objects and LLM explanation into a subdirectory named by its llm_score."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the input CSV with columns including 'image_path', 'llm_score', 'objects_detected', 'llm_explanation'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where annotated images will be written, grouped by score",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="",
        help="Optional filesystem prefix to resolve relative image paths",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default="",
        help="Optional path to a .ttf/.otf font file for captions",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=18,
        help="Font size used for caption text",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional limit on number of rows to process (0 = no limit)",
    )

    args = parser.parse_args()
    process_csv(
        csv_path=args.csv,
        output_dir=args.output_dir,
        images_root=args.images_root,
        font_path=args.font_path if getattr(args, "font_path", "") else None,
        font_size=int(args.font_size),
        max_rows=int(args.max_rows),
    )


if __name__ == "__main__":
    main()


