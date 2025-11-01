import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Support running as a package (relative imports) and as a script (absolute fallback)
try:  # relative imports when invoked as module: python -m dataset_builder.validation.run_intra_place_judge
    from .judge_models import HFJudge, JudgeConfig
    from .prompts import SYSTEM_INSTRUCTIONS, build_consistency_prompt
    from .parsing import load_all_assignments, load_predictions_csv
except Exception:  # fallback when executed directly
    import sys
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from dataset_builder.validation.judge_models import HFJudge, JudgeConfig
    from dataset_builder.validation.prompts import SYSTEM_INSTRUCTIONS, build_consistency_prompt
    from dataset_builder.validation.parsing import load_all_assignments, load_predictions_csv


def sample_descriptions_uniform(descriptions: List[str], max_count: int) -> List[str]:
    if len(descriptions) <= max_count:
        return descriptions
    # Evenly spaced sampling to cover variety
    indices = np.linspace(0, len(descriptions) - 1, num=max_count)
    indices = np.unique(indices.astype(int)).tolist()
    return [descriptions[i] for i in indices]


def append_row_to_csv(csv_path: str, row: Dict, header_order: List[str]) -> None:
    """Append one row to CSV, writing header if file doesn't exist."""
    df = pd.DataFrame([{k: row.get(k, None) for k in header_order}])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def evaluate_clusters(
    predictions_csv: str,
    clustered_dir: str,
    output_dir: str,
    only_city: str,
    model_name: str,
    max_desc: int,
    temperature: float,
    dtype: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    preds_df = load_predictions_csv(predictions_csv)
    assigns_df = load_all_assignments(clustered_dir, only_city)

    merged = preds_df.merge(assigns_df, on="panoid", how="left", suffixes=("_pred", ""))
    merged = merged.dropna(subset=["cluster_id"])  # keep only images mapped to clusters

    if merged.empty:
        raise SystemExit("No predictions matched to clustered assignments. Check panoid parsing and inputs.")

    judge = HFJudge(
        JudgeConfig(
            model_name=model_name,
            temperature=temperature,
            dtype=dtype if dtype else None,
        )
    )

    # Prepare per-city result directories
    by_city = merged.groupby("city_id", dropna=False)

    # Global results path (append per cluster)
    global_path = os.path.join(output_dir, "intra_place_consistency_all.csv")
    if os.path.exists(global_path):
        os.remove(global_path)
    header = [
        "city_id",
        "cluster_id",
        "num_images",
        "num_used_descriptions",
        "score",
        "explanation",
        "cluster_csv",
    ]

    for city, city_df in by_city:
        city_out_dir = os.path.join(output_dir, str(city))
        clusters_dir = os.path.join(city_out_dir, "clusters")
        os.makedirs(clusters_dir, exist_ok=True)

        # Per-city results (append per cluster)
        city_results_path = os.path.join(city_out_dir, f"intra_place_consistency_{city}.csv")
        if os.path.exists(city_results_path):
            os.remove(city_results_path)

        n_clusters = int(city_df["cluster_id"].nunique())
        for cid, g in tqdm(
            city_df.groupby("cluster_id"), total=n_clusters, desc=f"{city} clusters", unit="cluster"
        ):
            descriptions = g["description"].astype(str).tolist()
            sampled = sample_descriptions_uniform(descriptions, max_desc)
            prompt = build_consistency_prompt(sampled)
            verdict = judge.judge(prompt, system_instructions=SYSTEM_INSTRUCTIONS)

            # Write per-cluster CSV of image names and descriptions
            cluster_csv_path = os.path.join(clusters_dir, f"cluster_{int(cid)}.csv")
            g_out = g[["image_path", "panoid", "description"]].copy()
            g_out.to_csv(cluster_csv_path, index=False)

            row = {
                "city_id": city,
                "cluster_id": int(cid),
                "num_images": int(g.shape[0]),
                "num_used_descriptions": int(len(sampled)),
                "score": int(verdict.get("score", 3)),
                "explanation": str(verdict.get("explanation", "")),
                "cluster_csv": cluster_csv_path,
            }

            # Append to per-city and global CSVs immediately
            append_row_to_csv(city_results_path, row, header)
            append_row_to_csv(global_path, row, header)

    return global_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge for intra-place textual consistency.")
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
        "--model-name",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Hugging Face model to use for judging",
    )
    parser.add_argument(
        "--max-descriptions",
        type=int,
        default=20,
        help="Max descriptions sampled per cluster to keep prompts manageable",
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

    args = parser.parse_args()
    out = evaluate_clusters(
        predictions_csv=args.predictions_csv,
        clustered_dir=args.clustered_dir,
        output_dir=args.output_dir,
        only_city=args.city,
        model_name=args.model_name,
        max_desc=args.max_descriptions,
        temperature=args.temperature,
        dtype=args.dtype,
    )
    print(f"Wrote global results to: {out}")


if __name__ == "__main__":
    main()


