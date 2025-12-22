import argparse
import glob
import json
import os
import textwrap
from typing import List, Tuple, Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path) if root else path


def _parse_json_list(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    # best-effort split
    parts = [p.strip() for p in s.replace("[", "").replace("]", "").split(",")]
    return [p for p in parts if p]


def _find_cluster_csv(clusters_dir: str, cluster_id: int) -> str:
    candidates = [
        os.path.join(clusters_dir, f"cluster_{int(cluster_id)}_single.csv"),
        os.path.join(clusters_dir, f"cluster_{int(cluster_id)}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback to glob any matching prefix
    pattern = os.path.join(clusters_dir, f"cluster_{int(cluster_id)}_*.csv")
    hits = sorted(glob.glob(pattern))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No cluster CSV found for cluster_id={cluster_id} under {clusters_dir}")


def _load_image_safe(path: str, target_width: int) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        # placeholder if image can't be opened
        img = Image.new("RGB", (target_width, int(target_width * 0.75)), (220, 220, 220))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "(missing)", fill=(0, 0, 0))
        return img
    # resize keeping aspect
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = target_width / float(w)
    new_h = max(1, int(h * scale))
    return img.resize((target_width, new_h), resample=Image.BILINEAR)


def _compose_collage(
    images: List[Image.Image],
    numbers: List[int],
    outlier_flags: Optional[List[int]],
    header_title: str,
    subtitle_lines: List[str],
    cols: int,
    pad: int = 8,
) -> Image.Image:
    if not images:
        return Image.new("RGB", (640, 480), (255, 255, 255))

    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    # cell dims: add room at top for number
    thumb_w, thumb_h = images[0].size
    label_h = 18
    cell_w = thumb_w
    cell_h = label_h + thumb_h

    rows = (len(images) + cols - 1) // cols
    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * cell_h + (rows + 1) * pad

    # header text height
    header_lines = [header_title] + subtitle_lines
    wrapped_lines: List[str] = []
    max_header_width = grid_w - 2 * pad
    for line in header_lines:
        wrapped_lines.extend(textwrap.wrap(line, width=100))
    line_h = title_font.getbbox("A")[3] + 6
    header_h = pad + line_h * len(wrapped_lines) + pad

    out = Image.new("RGB", (grid_w, header_h + grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)

    # draw header
    y = pad
    for line in wrapped_lines:
        draw.text((pad, y), line, font=title_font, fill=(0, 0, 0))
        y += line_h

    # paste grid
    gx0 = pad
    gy0 = header_h
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = gx0 + pad + c * (cell_w + pad)
        y = gy0 + pad + r * (cell_h + pad)

        # label bar
        draw.rectangle([x, y, x + cell_w, y + label_h], fill=(245, 245, 245))
        num_text = str(numbers[idx])
        draw.text((x + 4, y + 2), num_text, font=font, fill=(0, 0, 0))
        # image
        out.paste(img, (x, y + label_h))
        # outlier highlight (red border around the image area)
        if outlier_flags and idx < len(outlier_flags) and int(outlier_flags[idx]) == 1:
            bx0, by0 = x, y + label_h
            bx1, by1 = x + cell_w, y + label_h + thumb_h
            # draw a thicker rectangle by overdrawing
            for k in range(3):
                draw.rectangle([bx0 + k, by0 + k, bx1 - k, by1 - k], outline=(220, 0, 0))

    return out


def visualize_clusters(
    clusters_dir: str,
    summary_csv: str,
    output_dir: str,
    images_root: str,
    cols: int,
    thumb_width: int,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    summary = pd.read_csv(summary_csv)
    needed_cols = {"cluster_id", "cluster_score"}
    missing = [c for c in needed_cols if c not in summary.columns]
    if missing:
        raise SystemExit(f"Summary CSV missing columns: {missing}")

    for _, row in summary.iterrows():
        cid = int(row["cluster_id"]) if not pd.isna(row["cluster_id"]) else None
        if cid is None:
            continue
        score = int(row["cluster_score"]) if not pd.isna(row["cluster_score"]) else -1
        rationale = str(row.get("score_rationale", "")).strip()

        # read cluster members
        path = _find_cluster_csv(clusters_dir, cid)
        df = pd.read_csv(path)
        if "image_path" not in df.columns:
            # try alternate schema
            raise SystemExit(f"Cluster CSV {path} lacks 'image_path' column")
        image_paths = [
            _resolve_path(images_root, str(p))
            for p in df["image_path"].astype(str).tolist()
        ]
        outlier_flags = df["is_outlier"].astype(int).tolist() if "is_outlier" in df.columns else [0] * len(image_paths)

        # load images
        thumbs: List[Image.Image] = []
        for p in image_paths:
            thumbs.append(_load_image_safe(p, thumb_width))

        # header/subtitle
        num_outliers = sum(1 for v in outlier_flags if int(v) == 1)
        header = f"Cluster {cid}  |  Score {score}  |  N={len(thumbs)}  |  Outliers={num_outliers}"
        subtitle = []
        if rationale:
            subtitle.append(f"Why: {rationale}")
        subtitle.append("Red border = outlier (is_outlier=1)")
        # include a few outlier reasons if present
        outlier_reasons = df["outlier_reason"].astype(str).tolist() if "outlier_reason" in df.columns else ["" for _ in image_paths]
        added = 0
        for idx, (flag, reason) in enumerate(zip(outlier_flags, outlier_reasons)):
            if int(flag) == 1 and str(reason).strip():
                subtitle.append(f"#{idx}: {str(reason).strip()}")
                added += 1
                if added >= 8:
                    break

        collage = _compose_collage(
            images=thumbs,
            numbers=list(range(len(thumbs))),
            outlier_flags=outlier_flags,
            header_title=header,
            subtitle_lines=subtitle,
            cols=cols,
        )

        score_dir = os.path.join(output_dir, str(score))
        os.makedirs(score_dir, exist_ok=True)
        out_path = os.path.join(score_dir, f"cluster_{cid}.jpg")
        collage.save(out_path, quality=90)

    return output_dir


def visualize_single_cluster(
    clusters_dir: str,
    cluster_id: int,
    score: int,
    score_rationale: str,
    output_dir: str,
    images_root: str,
    cols: int,
    thumb_width: int,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = _find_cluster_csv(clusters_dir, cluster_id)
    df = pd.read_csv(path)
    if "image_path" not in df.columns:
        raise SystemExit(f"Cluster CSV {path} lacks 'image_path' column")
    image_paths = [
        _resolve_path(images_root, str(p))
        for p in df["image_path"].astype(str).tolist()
    ]
    outlier_flags = df["is_outlier"].astype(int).tolist() if "is_outlier" in df.columns else [0] * len(image_paths)

    thumbs: List[Image.Image] = []
    for p in image_paths:
        thumbs.append(_load_image_safe(p, thumb_width))

    num_outliers = sum(1 for v in outlier_flags if int(v) == 1)
    outlier_reasons = df["outlier_reason"].astype(str).tolist() if "outlier_reason" in df.columns else ["" for _ in image_paths]
    header = f"Cluster {cluster_id}  |  Score {score}  |  N={len(thumbs)}  |  Outliers={num_outliers}"
    subtitle = []
    if score_rationale:
        subtitle.append(f"Why: {score_rationale}")
    subtitle.append("Red border = outlier (is_outlier=1)")
    # Add a few outlier reasons
    added = 0
    for idx, (flag, reason) in enumerate(zip(outlier_flags, outlier_reasons)):
        if int(flag) == 1 and str(reason).strip():
            subtitle.append(f"#{idx}: {str(reason).strip()}")
            added += 1
            if added >= 8:
                break

    collage = _compose_collage(
        images=thumbs,
        numbers=list(range(len(thumbs))),
        outlier_flags=outlier_flags,
        header_title=header,
        subtitle_lines=subtitle,
        cols=cols,
    )
    score_dir = os.path.join(output_dir, str(score))
    os.makedirs(score_dir, exist_ok=True)
    out_path = os.path.join(score_dir, f"cluster_{cluster_id}.jpg")
    collage.save(out_path, quality=90)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize clusters (single method): score and outliers only")
    parser.add_argument("--clusters-dir", type=str, required=True, help="Path to city clusters directory")
    parser.add_argument("--summary-csv", type=str, required=True, help="Path to per-city cluster_consistency CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save collages")
    parser.add_argument("--images-root", type=str, default="", help="Optional root for resolving image_path")
    parser.add_argument("--cols", type=int, default=6, help="Number of images per row")
    parser.add_argument("--thumb-width", type=int, default=256, help="Thumbnail width in pixels")

    args = parser.parse_args()
    out = visualize_clusters(
        clusters_dir=args.clusters_dir,
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
        images_root=args.images_root,
        cols=args.cols,
        thumb_width=args.thumb_width,
    )
    print(f"Wrote collages under: {out}")


if __name__ == "__main__":
    main()



