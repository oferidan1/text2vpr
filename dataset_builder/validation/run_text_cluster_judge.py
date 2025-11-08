import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Support running as a package (relative imports) and as a script (absolute fallback)
try:  # relative imports when invoked as module
    from .judge_models import HFJudge, JudgeConfig
    from .prompts import (
        SYSTEM_INSTRUCTIONS,
        build_cluster_prompt,
        build_chunk_map_prompt,
    )
    from ..validation.parsing import load_all_assignments, load_predictions_csv
except Exception:  # fallback when executed directly
    import sys
    # We need the project root (one level ABOVE `dataset_builder`) on sys.path
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from dataset_builder.validation.judge_models import HFJudge, JudgeConfig
    from dataset_builder.validation.prompts import (
        SYSTEM_INSTRUCTIONS,
        build_cluster_prompt,
        build_chunk_map_prompt,
    )
    from dataset_builder.validation.parsing import load_all_assignments, load_predictions_csv


def append_row_to_csv(csv_path: str, row: Dict, header_order: List[str]) -> None:
    df = pd.DataFrame([{k: row.get(k, None) for k in header_order}])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def chunk_indices(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def evaluate_single_shot(
    merged: pd.DataFrame,
    output_dir: str,
    model_name: str,
    temperature: float,
    dtype: str,
    max_descriptions: int,
) -> str:
    judge = HFJudge(
        JudgeConfig(
            model_name=model_name,
            temperature=temperature,
            dtype=dtype if dtype else None,
        )
    )

    by_city = merged.groupby("city_id", dropna=False)
    global_path = os.path.join(output_dir, "cluster_consistency_all.csv")
    if os.path.exists(global_path):
        os.remove(global_path)
    header = [
        "city_id",
        "cluster_id",
        "num_images",
        "num_used_descriptions",
        "cluster_score",
        "overlap_themes",
        "critical_inconsistencies",
        "noncritical_inconsistencies",
        "outliers_csv",
    ]

    for city, city_df in by_city:
        city_out_dir = os.path.join(output_dir, str(city))
        clusters_dir = os.path.join(city_out_dir, "clusters")
        os.makedirs(clusters_dir, exist_ok=True)

        city_results_path = os.path.join(city_out_dir, f"cluster_consistency_{city}.csv")
        if os.path.exists(city_results_path):
            os.remove(city_results_path)

        n_clusters = int(city_df["cluster_id"].nunique())
        for cid, g in tqdm(
            city_df.groupby("cluster_id"), total=n_clusters, desc=f"{city} clusters", unit="cluster"
        ):
            descriptions = g["description"].astype(str).tolist()
            image_paths = g["image_path"].astype(str).tolist()
            panoids = g["panoid"].astype(str).tolist()

            # Sample if needed to fit prompt limits
            if len(descriptions) > max_descriptions:
                # uniform sampling to retain coverage
                idx = np.linspace(0, len(descriptions) - 1, num=max_descriptions)
                sel = np.unique(idx.astype(int)).tolist()
            else:
                sel = list(range(len(descriptions)))

            sampled_desc = [descriptions[i] for i in sel]
            prompt = build_cluster_prompt(sampled_desc)
            verdict = judge.judge(prompt, system_instructions=SYSTEM_INSTRUCTIONS)

            # Build an outlier CSV per cluster (marking sampled positions only)
            out_path = os.path.join(clusters_dir, f"cluster_{int(cid)}_single.csv")
            outlier_indices = verdict.get("outlier_indices", [])
            # Map sampled indices back to original positions
            outlier_global = set(sel[i] for i in outlier_indices if 0 <= i < len(sel))
            df_out = pd.DataFrame(
                {
                    "description": descriptions,
                    "image_path": image_paths,
                    "panoid": panoids,
                    "is_outlier": [1 if k in outlier_global else 0 for k in range(len(image_paths))],
                }
            )
            df_out.to_csv(out_path, index=False)

            row = {
                "city_id": city,
                "cluster_id": int(cid),
                "num_images": int(len(descriptions)),
                "num_used_descriptions": int(len(sampled_desc)),
                "cluster_score": int(verdict.get("score", 3)),
                "overlap_themes": json.dumps(verdict.get("overlap", []), ensure_ascii=False),
                "critical_inconsistencies": json.dumps(
                    verdict.get("critical_inconsistencies", []), ensure_ascii=False
                ),
                "noncritical_inconsistencies": json.dumps(
                    verdict.get("noncritical_inconsistencies", []), ensure_ascii=False
                ),
                "outliers_csv": out_path,
            }
            append_row_to_csv(city_results_path, row, header)
            append_row_to_csv(global_path, row, header)

    return global_path


def evaluate_map_reduce(
    merged: pd.DataFrame,
    output_dir: str,
    model_name: str,
    temperature: float,
    dtype: str,
    chunk_size: int,
) -> str:
    judge = HFJudge(
        JudgeConfig(
            model_name=model_name,
            temperature=temperature,
            dtype=dtype if dtype else None,
        )
    )

    by_city = merged.groupby("city_id", dropna=False)
    global_path = os.path.join(output_dir, "cluster_consistency_all.csv")
    if os.path.exists(global_path):
        os.remove(global_path)
    header = [
        "city_id",
        "cluster_id",
        "num_images",
        "num_chunks",
        "cluster_score_median",
        "cluster_score_mean",
        "overlap_themes",
        "critical_inconsistencies",
        "noncritical_inconsistencies",
        "outliers_csv",
    ]

    for city, city_df in by_city:
        city_out_dir = os.path.join(output_dir, str(city))
        clusters_dir = os.path.join(city_out_dir, "clusters")
        os.makedirs(clusters_dir, exist_ok=True)

        city_results_path = os.path.join(city_out_dir, f"cluster_consistency_{city}.csv")
        if os.path.exists(city_results_path):
            os.remove(city_results_path)

        n_clusters = int(city_df["cluster_id"].nunique())
        for cid, g in tqdm(
            city_df.groupby("cluster_id"), total=n_clusters, desc=f"{city} clusters", unit="cluster"
        ):
            descriptions = g["description"].astype(str).tolist()
            image_paths = g["image_path"].astype(str).tolist()
            panoids = g["panoid"].astype(str).tolist()

            n = len(descriptions)
            if n == 0:
                continue
            ranges = chunk_indices(n, chunk_size)

            chunk_scores: List[int] = []
            chunk_overlaps: List[str] = []
            chunk_crit: List[str] = []
            chunk_noncrit: List[str] = []
            outlier_mask = [0] * n

            for s, e in ranges:
                chunk_desc = descriptions[s:e]
                prompt = build_chunk_map_prompt(chunk_desc)
                verdict = judge.judge(prompt, system_instructions=SYSTEM_INSTRUCTIONS)
                # accumulate
                chunk_scores.append(int(verdict.get("score", 3)))
                chunk_overlaps.extend(verdict.get("overlap", []))
                chunk_crit.extend(verdict.get("critical_inconsistencies", []))
                chunk_noncrit.extend(verdict.get("noncritical_inconsistencies", []))
                out_idx = verdict.get("outlier_indices", [])
                for oi in out_idx:
                    gi = s + oi
                    if 0 <= gi < n:
                        outlier_mask[gi] = 1

            # simple programmatic reduce
            median_score = float(np.median(chunk_scores)) if chunk_scores else None
            mean_score = float(np.mean(chunk_scores)) if chunk_scores else None

            # de-duplicate themes keeping order
            def unique_order(seq: List[str]) -> List[str]:
                seen = set()
                out: List[str] = []
                for x in seq:
                    k = x.strip()
                    if k and k not in seen:
                        seen.add(k)
                        out.append(k)
                return out

            overlap_agg = unique_order(chunk_overlaps)[:30]
            crit_agg = unique_order(chunk_crit)[:30]
            noncrit_agg = unique_order(chunk_noncrit)[:30]

            # Write outlier CSV per cluster
            out_path = os.path.join(clusters_dir, f"cluster_{int(cid)}_mapreduce.csv")
            df_out = pd.DataFrame(
                {
                    "image_path": image_paths,
                    "panoid": panoids,
                    "is_outlier": outlier_mask,
                }
            )
            df_out.to_csv(out_path, index=False)

            row = {
                "city_id": city,
                "cluster_id": int(cid),
                "num_images": int(n),
                "num_chunks": int(len(ranges)),
                "cluster_score_median": median_score,
                "cluster_score_mean": mean_score,
                "overlap_themes": json.dumps(overlap_agg, ensure_ascii=False),
                "critical_inconsistencies": json.dumps(crit_agg, ensure_ascii=False),
                "noncritical_inconsistencies": json.dumps(noncrit_agg, ensure_ascii=False),
                "outliers_csv": out_path,
            }
            append_row_to_csv(city_results_path, row, header)
            append_row_to_csv(global_path, row, header)

    return global_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster/text-level VPR judge: single-shot or map-reduce.")
    parser.add_argument(
        "--predictions-csv",
        type=str,
        required=True,
        help="CSV with columns: image_path, description",
    )
    parser.add_argument(
        "--clustered-dir",
        type=str,
        default="Dataframes_clustered",
        help="Directory with per-city clustered assignments (e.g., London.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_outputs",
        help="Where to write results and per-cluster CSVs",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="",
        help="Optional single city to evaluate (matches city_id).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["single", "map_reduce"],
        default="single",
        help="Algorithm: single=single-shot cluster judge; map_reduce=minibatch map-reduce",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Hugging Face model to use for judging",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature; set 0 for deterministic",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="",
        help="Optional torch dtype: float16 or bfloat16",
    )
    parser.add_argument(
        "--max-descriptions",
        "--max-description",
        dest="max_descriptions",
        type=int,
        default=60,
        help="Max descriptions per cluster for single-shot method",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Chunk size for map-reduce method",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    preds_df = load_predictions_csv(args.predictions_csv)
    assigns_df = load_all_assignments(args.clustered_dir, args.city)
    merged = preds_df.merge(assigns_df, on="panoid", how="left", suffixes=("_pred", ""))
    merged = merged.dropna(subset=["cluster_id"])  # keep only images mapped to clusters
    if merged.empty:
        raise SystemExit("No predictions matched to clustered assignments. Check panoid parsing and inputs.")

    if args.method == "single":
        out = evaluate_single_shot(
            merged=merged,
            output_dir=args.output_dir,
            model_name=args.model_name,
            temperature=args.temperature,
            dtype=args.dtype,
            max_descriptions=args.max_descriptions,
        )
    else:
        out = evaluate_map_reduce(
            merged=merged,
            output_dir=args.output_dir,
            model_name=args.model_name,
            temperature=args.temperature,
            dtype=args.dtype,
            chunk_size=args.chunk_size,
        )
    print(f"Wrote global results to: {out}")


if __name__ == "__main__":
    main()



