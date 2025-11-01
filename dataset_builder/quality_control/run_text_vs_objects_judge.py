import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse HF judge from validation (support module and script runs)
try:  # when invoked as module: python -m dataset_builder.quality_control.run_text_vs_objects_judge
    from ..validation.judge_models import HFJudge, JudgeConfig
    from .segmentation import SegConfig, Segmenter
    from .prompts_text_vs_objects import SYSTEM_INSTRUCTIONS_TXT_OBJ, build_text_vs_objects_prompt
except Exception:  # fallback when executed directly
    import sys
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from dataset_builder.validation.judge_models import HFJudge, JudgeConfig
    from dataset_builder.quality_control.segmentation import SegConfig, Segmenter
    from dataset_builder.quality_control.prompts_text_vs_objects import (
        SYSTEM_INSTRUCTIONS_TXT_OBJ,
        build_text_vs_objects_prompt,
    )


def resolve_image_path(images_root: str, image_path: str) -> str:
    if os.path.isabs(image_path):
        return image_path
    if images_root:
        return os.path.join(images_root, image_path)
    return image_path


def extract_city_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    try:
        idx = parts.index("Images")
        return parts[idx + 1]
    except Exception:
        return ""


def append_row_to_csv(csv_path: str, row: Dict, header_order: List[str]) -> None:
    df = pd.DataFrame([{k: row.get(k, None) for k in header_order}])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def compute_simple_overlap_scores(objects: List[str], description: str) -> Dict[str, float]:
    desc = description.lower()
    detected = [o.lower() for o in objects]
    if not detected:
        return {"detected_in_text_ratio": 0.0, "text_in_detected_ratio": 0.0}

    # Count how many detected object names appear in description
    detected_mentions = sum(int(obj in desc) for obj in detected)
    detected_in_text_ratio = detected_mentions / max(1, len(detected))

    # Approximate unique text object mentions by checking substrings for detected names
    # (conservative; better noun-phrase extraction could replace this)
    text_object_matches = detected_mentions
    text_in_detected_ratio = text_object_matches / max(1, detected_mentions)

    return {
        "detected_in_text_ratio": float(detected_in_text_ratio),
        "text_in_detected_ratio": float(text_in_detected_ratio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Text vs Segmentation Objects Judge")
    parser.add_argument(
        "--predictions-csv",
        type=str,
        required=True,
        help="CSV with columns: image_path, description",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="",
        help="Optional root directory to resolve relative image_path",
    )
    parser.add_argument(
        "--seg-backend",
        type=str,
        default="hf_segformer_b2_ade",
        choices=["hf_segformer_b2_ade", "torchvision_deeplabv3"],
        help="Segmentation backend",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.4,
        help="Minimum average softmax prob for keeping a segment",
    )
    parser.add_argument(
        "--min-area-pct",
        type=float,
        default=0.002,
        help="Minimum area fraction for a segment to be considered",
    )
    parser.add_argument(
        "--debug-seg",
        action="store_true",
        help="If set, save overlay images of segmentation",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="",
        help="Optional debug dir for overlays; defaults under output dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="quality_outputs",
        help="Where to write results and overlays",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Hugging Face model for judging",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="",
        help="Optional torch dtype for LLM: float16 or bfloat16",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.predictions_csv)
    if "image_path" not in df.columns or "description" not in df.columns:
        raise SystemExit("Predictions CSV must contain 'image_path' and 'description'.")

    seg = Segmenter(SegConfig(backend=args.seg_backend, prob_threshold=args.prob_threshold, min_area_pct=args.min_area_pct))
    judge = HFJudge(JudgeConfig(model_name=args.model_name, temperature=args.temperature, dtype=args.dtype or None))

    # Prepare outputs
    global_path = os.path.join(args.output_dir, "text_vs_segments_consistency_all.csv")
    if os.path.exists(global_path):
        os.remove(global_path)
    header = [
        "city_id",
        "image_path",
        "objects_detected",
        "llm_score",
        "llm_explanation",
        "omit_list",
        "suggested_description",
        "detected_in_text_ratio",
        "text_in_detected_ratio",
        "overlay_path",
    ]

    # Group by city for tidy outputs and progress
    df["city_id"] = df["image_path"].apply(extract_city_from_path)
    for city, g in df.groupby("city_id"):
        city_dir = os.path.join(args.output_dir, str(city))
        overlays_dir = args.debug_dir or os.path.join(city_dir, "debug_overlays")
        os.makedirs(city_dir, exist_ok=True)
        if args.debug_seg:
            os.makedirs(overlays_dir, exist_ok=True)
        city_results_path = os.path.join(city_dir, f"text_vs_segments_consistency_{city}.csv")
        if os.path.exists(city_results_path):
            os.remove(city_results_path)

        for _, row in tqdm(g.iterrows(), total=g.shape[0], desc=f"{city} images", unit="img"):
            rel_path = str(row["image_path"])
            img_path = resolve_image_path(args.images_root, rel_path)
            description = str(row["description"]) if not pd.isna(row["description"]) else ""

            overlay_path = ""
            try:
                label_map, objects = seg.segment_image(img_path)
                if args.debug_seg:
                    overlay = seg.overlay_mask(img_path, label_map, seg.default_palette())
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    overlay_path = os.path.join(overlays_dir, f"{base}_overlay.jpg")
                    overlay.save(overlay_path)
            except Exception as e:
                objects = []

            prompt = build_text_vs_objects_prompt(objects, description)
            verdict = judge.judge(prompt, system_instructions=SYSTEM_INSTRUCTIONS_TXT_OBJ)

            overlap = compute_simple_overlap_scores(objects, description)

            out_row = {
                "city_id": city,
                "image_path": img_path,
                "objects_detected": ", ".join(objects),
                "llm_score": int(verdict.get("score", 3)),
                "llm_explanation": str(verdict.get("explanation", ""))[:1000],
                "omit_list": ", ".join(verdict.get("omit_list", [])) if isinstance(verdict.get("omit_list", []), list) else str(verdict.get("omit_list", "")),
                "suggested_description": str(verdict.get("suggested_description", ""))[:1000],
                "detected_in_text_ratio": overlap.get("detected_in_text_ratio", 0.0),
                "text_in_detected_ratio": overlap.get("text_in_detected_ratio", 0.0),
                "overlay_path": overlay_path,
            }

            append_row_to_csv(city_results_path, out_row, header)
            append_row_to_csv(global_path, out_row, header)

    print(f"Wrote global results to: {global_path}")


if __name__ == "__main__":
    main()


