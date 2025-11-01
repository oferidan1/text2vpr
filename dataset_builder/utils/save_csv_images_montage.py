import argparse
import math
import os
from typing import List, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


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


def make_montage(
    image_paths: List[str],
    images_root: str,
    out_path: str,
    thumb_size: Tuple[int, int] = (256, 256),
    cols: int = 5,
    margin: int = 8,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    add_index: bool = False,
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
    canvas_h = rows * cell_h + (rows + 1) * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    font = None
    if add_index:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x0 = margin + c * (cell_w + margin)
        y0 = margin + r * (cell_h + margin)
        # center image in cell
        paste_x = x0 + (cell_w - img.width) // 2
        paste_y = y0 + (cell_h - img.height) // 2
        canvas.paste(img, (paste_x, paste_y))
        if add_index and font is not None:
            draw.text((x0 + 4, y0 + 4), str(idx + 1), fill=(0, 0, 0), font=font)

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

    args = parser.parse_args()

    csv_files = [
        os.path.join(args.csv_dir, f)
        for f in sorted(os.listdir(args.csv_dir))
        if f.lower().endswith(".csv")
    ]
    if not csv_files:
        raise SystemExit("No CSV files found in --csv-dir.")

    for csv_path in csv_files:
        try:
            rel_paths = read_image_paths_from_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
            continue

        if args.max_images and len(rel_paths) > args.max_images:
            rel_paths = rel_paths[: args.max_images]

        base = os.path.splitext(os.path.basename(csv_path))[0]
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
            )
            print(f"Wrote {out_png}")
        except Exception as e:
            print(f"Failed to write montage for {csv_path}: {e}")


if __name__ == "__main__":
    main()


