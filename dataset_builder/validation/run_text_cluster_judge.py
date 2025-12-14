import argparse
import json
import math
import os
import sys
from datetime import datetime
import shlex
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

# Support running as a package (relative imports) and as a script (absolute fallback)
try:  # relative imports when invoked as module
    from .judge_models import HFJudge, JudgeConfig
    from .prompts import (
        SYSTEM_INSTRUCTIONS,
        RANDOM_SYSTEM_INSTRUCTIONS,
        build_cluster_prompt,
        build_chunk_map_prompt,
        build_random_cluster_prompt,
        build_random_chunk_map_prompt,
    )
    from ..validation.parsing import load_all_assignments, load_predictions_csv
    from .visualize import visualize_clusters, visualize_single_cluster
except Exception:  # fallback when executed directly
    import sys
    # We need the project root (one level ABOVE `dataset_builder`) on sys.path
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from dataset_builder.validation.judge_models import HFJudge, JudgeConfig
    from dataset_builder.validation.prompts import (
        SYSTEM_INSTRUCTIONS,
        RANDOM_SYSTEM_INSTRUCTIONS,
        build_cluster_prompt,
        build_chunk_map_prompt,
        build_random_cluster_prompt,
        build_random_chunk_map_prompt,
    )
    from dataset_builder.validation.parsing import load_all_assignments, load_predictions_csv
    from dataset_builder.validation.visualize import visualize_clusters, visualize_single_cluster


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[int] = None  # GPU index; -1 for CPU; None=auto
    max_length: int = 256
    batch_size: int = 64
    top_percent: float = 10.0  # percent of farthest points to flag as outliers


class TextEmbeddingEncoder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        # Resolve device
        if config.device is not None:
            if int(config.device) >= 0 and torch.cuda.is_available():
                self.device_str = f"cuda:{int(config.device)}"
            else:
                self.device_str = "cpu"
        else:
            if torch.cuda.is_available():
                self.device_str = "cuda:0"
            elif torch.backends.mps.is_available():
                self.device_str = "mps"
            else:
                self.device_str = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        try:
            self.model.to(self.device_str)
        except Exception:
            # fallback to CPU if device move fails
            self.device_str = "cpu"
            self.model.to(self.device_str)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        batches: List[np.ndarray] = []
        bs = max(1, int(self.config.batch_size))
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=int(self.config.max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(self.device_str) for k, v in enc.items()}
            outputs = self.model(**enc)
            # Mean pooling with attention mask
            token_embeddings = outputs.last_hidden_state  # [B, T, H]
            input_mask_expanded = enc["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            # L2 normalize
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            batches.append(sentence_embeddings.detach().cpu().numpy().astype(np.float32))
        return np.vstack(batches)


def compute_embedding_outliers(
    descriptions: List[str],
    top_percent: float,
    encoder: TextEmbeddingEncoder,
) -> Tuple[List[int], List[float]]:
    """
    Returns (outlier_mask, distances) where outlier_mask is a list of 0/1 aligned to descriptions,
    and distances are cosine distances to the centroid (higher = farther).
    """
    n = len(descriptions)
    if n == 0:
        return [], []
    # Clamp top_percent into [0, 100]
    p = max(0.0, min(100.0, float(top_percent)))
    if p <= 0.0:
        return [0] * n, [0.0] * n
    emb = encoder.encode(descriptions)  # [n, d], already L2-normalized
    if emb.shape[0] != n:
        emb = np.resize(emb, (n, emb.shape[1]))
    # Centroid and normalize
    centroid = emb.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-12
    centroid = centroid / norm
    # Cosine similarity = dot since both normalized; distance = 1 - sim
    sims = (emb * centroid).sum(axis=1)
    dists = (1.0 - sims).astype(np.float32).tolist()
    # Select top k% farthest
    k = int(math.ceil(n * (p / 100.0)))
    k = max(1, min(n, k))
    order = np.argsort(dists)[::-1]  # descending distance
    outlier_idx = set(int(i) for i in order[:k])
    mask = [1 if i in outlier_idx else 0 for i in range(n)]
    return mask, dists


def append_row_to_csv(csv_path: str, row: Dict, header_order: List[str]) -> None:
    df = pd.DataFrame([{k: row.get(k, None) for k in header_order}])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def chunk_indices(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def build_random_clusters(
    merged: pd.DataFrame,
    size_min: int,
    size_max: int,
) -> pd.DataFrame:
    """
    Construct synthetic clusters per city by sampling random-size groups of descriptions
    where each member comes from a different original cluster.
    The number of synthetic clusters equals the number of original clusters per city.
    """
    if size_min < 1 or size_max < 1:
        raise SystemExit("--random-size-min/max must be >= 1")
    if size_min > size_max:
        size_min, size_max = size_max, size_min

    rng = np.random.default_rng()
    out_frames: List[pd.DataFrame] = []

    for city, city_df in merged.groupby("city_id", dropna=False):
        # group rows by original cluster to ensure at most one sample from each per synthetic cluster
        by_orig = {cid: g.reset_index(drop=True) for cid, g in city_df.groupby("cluster_id")}
        orig_cluster_ids = list(by_orig.keys())
        n_synth = len(orig_cluster_ids)
        if n_synth == 0:
            continue

        # Precompute one random order over original clusters per synthetic cluster
        for new_cid in range(n_synth):
            # choose size and clamp to available unique original clusters
            target_size = int(rng.integers(low=size_min, high=size_max + 1))
            target_size = max(1, min(target_size, len(orig_cluster_ids)))
            perm = rng.permutation(len(orig_cluster_ids)).tolist()
            selected_rows: List[pd.Series] = []
            # take one random row from each distinct original cluster until target_size reached
            for idx in perm[:target_size]:
                ocid = orig_cluster_ids[idx]
                g = by_orig[ocid]
                ridx = int(rng.integers(low=0, high=len(g)))
                selected_rows.append(g.iloc[ridx])
            if not selected_rows:
                continue
            df_new = pd.DataFrame(selected_rows).copy()
            df_new["cluster_id"] = int(new_cid)  # overwrite with synthetic cluster id
            out_frames.append(df_new)

    if not out_frames:
        raise SystemExit("Failed to construct any random clusters.")
    randomized = pd.concat(out_frames, axis=0, ignore_index=True)
    # Keep only the necessary columns (preserve extras if present)
    needed_cols = ["city_id", "cluster_id", "description", "image_path", "panoid"]
    missing = [c for c in needed_cols if c not in randomized.columns]
    if missing:
        raise SystemExit(f"Missing required columns after randomization: {missing}")
    return randomized


def evaluate_single_shot(
    merged: pd.DataFrame,
    output_dir: str,
    model_name: str,
    temperature: float,
    dtype: str,
    max_descriptions: int,
    device: int | None = None,
    visualize: bool = False,
    viz_output_dir: str = "",
    viz_images_root: str = "",
    viz_cols: int = 6,
    viz_thumb_width: int = 256,
    system_instructions_override: Optional[str] = None,
    prompt_template_override: Optional[str] = None,
    use_random_prompts: bool = False,
    raw_output_dir: str = "",
    # Embedding outliers
    use_embedding_outliers: bool = False,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_top_percent: float = 10.0,
    embedding_device: Optional[int] = None,
    embedding_batch_size: int = 64,
) -> str:
    judge = HFJudge(
        JudgeConfig(
            model_name=model_name,
            temperature=temperature,
            dtype=dtype if dtype else None,
            device=device,
        )
    )
    # Optional embedding encoder
    if use_embedding_outliers:
        enc = TextEmbeddingEncoder(
            EmbeddingConfig(
                model_name=embedding_model_name,
                device=embedding_device if embedding_device is not None else device,
                batch_size=embedding_batch_size,
                top_percent=embedding_top_percent,
            )
        )
    else:
        enc = None

    if system_instructions_override is not None:
        sys_instr = system_instructions_override
    else:
        sys_instr = RANDOM_SYSTEM_INSTRUCTIONS if use_random_prompts else SYSTEM_INSTRUCTIONS

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
        "score_rationale",
        "outlier_rationale",
        "outliers_csv",
    ]

    for city, city_df in by_city:
        city_out_dir = os.path.join(output_dir, str(city))
        clusters_dir = os.path.join(city_out_dir, "clusters")
        os.makedirs(clusters_dir, exist_ok=True)
        # Embedding-based artifacts
        if use_embedding_outliers:
            clusters_embed_dir = os.path.join(city_out_dir, "clusters_by_embeddings")
            os.makedirs(clusters_embed_dir, exist_ok=True)
            viz_embed_city_dir = os.path.join(output_dir, "collages_embeddings", str(city))
            os.makedirs(viz_embed_city_dir, exist_ok=True)

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
            if prompt_template_override is not None:
                if "{descriptions}" in prompt_template_override:
                    numbered = "\n".join([f"{i}. {d}" for i, d in enumerate(sampled_desc)])
                    prompt = prompt_template_override.replace("{descriptions}", numbered)
                else:
                    prompt = prompt_template_override
            else:
                prompt = (
                    build_random_cluster_prompt(sampled_desc)
                    if use_random_prompts
                    else build_cluster_prompt(sampled_desc)
                )
            verdict = judge.judge(prompt, system_instructions=sys_instr)

            # Optionally persist raw prompt/response per cluster
            if raw_output_dir:
                try:
                    city_raw_dir = os.path.join(raw_output_dir, str(city))
                    os.makedirs(city_raw_dir, exist_ok=True)
                    raw_obj = {
                        "city_id": city,
                        "cluster_id": int(cid),
                        "method": "single",
                        "model": model_name,
                        "full_prompt": str(verdict.get("_full_prompt", "")),
                        "raw_text": str(verdict.get("_raw_text", "")),
                    }
                    raw_file = os.path.join(city_raw_dir, f"cluster_{int(cid)}_single.json")
                    with open(raw_file, "w", encoding="utf-8") as f:
                        json.dump(raw_obj, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            # Build an outlier CSV per cluster (marking sampled positions only)
            out_path = os.path.join(clusters_dir, f"cluster_{int(cid)}_single.csv")
            outlier_indices = verdict.get("outlier_indices", [])
            outlier_reasons_map = verdict.get("outlier_reasons_map", {}) or {}
            # Map sampled indices back to original positions
            outlier_global = set(sel[i] for i in outlier_indices if 0 <= i < len(sel))
            # Build per-image outlier reason aligned to full list
            outlier_reason: List[str] = []
            for k in range(len(image_paths)):
                if k in outlier_global:
                    # Map back from sampled pos to original index if needed
                    # Find sampled index for this original position
                    try:
                        sampled_pos = sel.index(k)
                    except ValueError:
                        sampled_pos = None
                    reason = ""
                    if sampled_pos is not None:
                        # reasons_map keys may be on sampled indices or original; try both
                        reason = str(
                            outlier_reasons_map.get(sampled_pos, outlier_reasons_map.get(k, ""))
                        ).strip()
                    outlier_reason.append(reason)
                else:
                    outlier_reason.append("")
            df_out = pd.DataFrame(
                {
                    "serial": list(range(len(image_paths))),
                    "description": descriptions,
                    "image_path": image_paths,
                    "panoid": panoids,
                    "is_outlier": [1 if k in outlier_global else 0 for k in range(len(image_paths))],
                    "outlier_reason": outlier_reason,
                }
            )
            df_out.to_csv(out_path, index=False)

            # Prepare rationale strings for summary CSV
            rationale_raw = str(verdict.get("rationale", "")).strip()
            if not rationale_raw:
                # Construct a minimal rationale if missing
                rationale_raw = f"Assigned score {int(verdict.get('score', 3))} with {int(sum(1 for v in outlier_reason if v))} outliers."
            else:
                # If the rationale string appears to contain JSON, attempt to extract the rationale field
                cleaned = rationale_raw
                try:
                    if cleaned.startswith("{") and cleaned.endswith("}"):
                        rat_obj = json.loads(cleaned)
                        inner_rationale = rat_obj.get("rationale", "")
                        if isinstance(inner_rationale, str) and inner_rationale:
                            cleaned = inner_rationale.strip()
                    else:
                        # Try to locate JSON substring and pick 'rationale'
                        import re as _re
                        m = _re.search(r"\{[\s\S]*\}", cleaned)
                        if m:
                            rat_obj = json.loads(m.group(0))
                            inner_rationale = rat_obj.get("rationale", "")
                            if isinstance(inner_rationale, str) and inner_rationale:
                                cleaned = inner_rationale.strip()
                except Exception:
                    pass
                rationale_raw = cleaned
            # Build outlier rationale mapping over full indices
            outlier_rationale_map = {int(i): outlier_reason[int(i)] for i in range(len(outlier_reason)) if outlier_reason[int(i)]}

            row = {
                "city_id": city,
                "cluster_id": int(cid),
                "num_images": int(len(descriptions)),
                "num_used_descriptions": int(len(sampled_desc)),
                "cluster_score": int(verdict.get("score", 3)),
                "score_rationale": rationale_raw[:1000],
                "outlier_rationale": json.dumps(outlier_rationale_map, ensure_ascii=False) if outlier_rationale_map else "",
                "outliers_csv": out_path,
            }
            append_row_to_csv(city_results_path, row, header)
            append_row_to_csv(global_path, row, header)

            # Optional immediate visualization per cluster
            if visualize:
                cluster_score = int(verdict.get("score", 3))
                cluster_rationale = str(verdict.get("rationale", ""))
                viz_city_dir = viz_output_dir or os.path.join(output_dir, "collages", str(city))
                os.makedirs(viz_city_dir, exist_ok=True)
                visualize_single_cluster(
                    clusters_dir=clusters_dir,
                    cluster_id=int(cid),
                    score=cluster_score,
                    score_rationale=cluster_rationale,
                    output_dir=viz_city_dir,
                    images_root=viz_images_root,
                    cols=viz_cols,
                    thumb_width=viz_thumb_width,
                )

            # Embedding-based outliers and visualization (independent of LLM)
            if use_embedding_outliers and enc is not None:
                # Use full cluster descriptions for alignment with images
                embed_mask, embed_dists = compute_embedding_outliers(descriptions, embedding_top_percent, enc)
                # Build outlier reasons (distance)
                embed_reasons = [f"cosine_dist={embed_dists[i]:.3f}" if embed_mask[i] == 1 else "" for i in range(len(embed_mask))]
                df_embed = pd.DataFrame(
                    {
                        "serial": list(range(len(image_paths))),
                        "description": descriptions,
                        "image_path": image_paths,
                        "panoid": panoids,
                        "is_outlier": embed_mask,
                        "outlier_reason": embed_reasons,
                    }
                )
                out_embed_csv = os.path.join(clusters_embed_dir, f"cluster_{int(cid)}_embed.csv")
                df_embed.to_csv(out_embed_csv, index=False)
                # Always generate embedding collages under a dedicated directory
                visualize_single_cluster(
                    clusters_dir=clusters_embed_dir,
                    cluster_id=int(cid),
                    score=int(verdict.get("score", -1)),
                    score_rationale=f"Embedding outliers (top {embedding_top_percent:.1f}%)",
                    output_dir=viz_embed_city_dir,
                    images_root=viz_images_root,
                    cols=viz_cols,
                    thumb_width=viz_thumb_width,
                )

    return global_path


def evaluate_map_reduce(
    merged: pd.DataFrame,
    output_dir: str,
    model_name: str,
    temperature: float,
    dtype: str,
    chunk_size: int,
    device: int | None = None,
    system_instructions_override: Optional[str] = None,
    prompt_template_override: Optional[str] = None,
    use_random_prompts: bool = False,
    raw_output_dir: str = "",
    # Embedding outliers
    use_embedding_outliers: bool = False,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_top_percent: float = 10.0,
    embedding_device: Optional[int] = None,
    embedding_batch_size: int = 64,
) -> str:
    judge = HFJudge(
        JudgeConfig(
            model_name=model_name,
            temperature=temperature,
            dtype=dtype if dtype else None,
            device=device,
        )
    )
    # Optional embedding encoder
    if use_embedding_outliers:
        enc = TextEmbeddingEncoder(
            EmbeddingConfig(
                model_name=embedding_model_name,
                device=embedding_device if embedding_device is not None else device,
                batch_size=embedding_batch_size,
                top_percent=embedding_top_percent,
            )
        )
    else:
        enc = None

    if system_instructions_override is not None:
        sys_instr = system_instructions_override
    else:
        sys_instr = RANDOM_SYSTEM_INSTRUCTIONS if use_random_prompts else SYSTEM_INSTRUCTIONS

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
        # Embedding artifacts
        if use_embedding_outliers:
            clusters_embed_dir = os.path.join(city_out_dir, "clusters_by_embeddings")
            os.makedirs(clusters_embed_dir, exist_ok=True)
            viz_embed_city_dir = os.path.join(output_dir, "collages_embeddings", str(city))
            os.makedirs(viz_embed_city_dir, exist_ok=True)

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
                if prompt_template_override is not None:
                    if "{descriptions}" in prompt_template_override:
                        numbered = "\n".join([f"{i}. {d}" for i, d in enumerate(chunk_desc)])
                        prompt = prompt_template_override.replace("{descriptions}", numbered)
                    else:
                        prompt = prompt_template_override
                else:
                    prompt = (
                        build_random_chunk_map_prompt(chunk_desc)
                        if use_random_prompts
                        else build_chunk_map_prompt(chunk_desc)
                    )
                verdict = judge.judge(prompt, system_instructions=sys_instr)
                # Optionally persist raw prompt/response per chunk
                if raw_output_dir:
                    try:
                        cluster_raw_dir = os.path.join(
                            raw_output_dir, str(city), f"cluster_{int(cid)}"
                        )
                        os.makedirs(cluster_raw_dir, exist_ok=True)
                        raw_obj = {
                            "city_id": city,
                            "cluster_id": int(cid),
                            "method": "map_reduce",
                            "chunk_start": int(s),
                            "chunk_end": int(e),
                            "model": model_name,
                            "full_prompt": str(verdict.get("_full_prompt", "")),
                            "raw_text": str(verdict.get("_raw_text", "")),
                        }
                        raw_file = os.path.join(cluster_raw_dir, f"chunk_{int(s)}_{int(e)}.json")
                        with open(raw_file, "w", encoding="utf-8") as f:
                            json.dump(raw_obj, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
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

            # Embedding-based outliers and visualization for map-reduce
            if use_embedding_outliers and enc is not None:
                embed_mask, embed_dists = compute_embedding_outliers(descriptions, embedding_top_percent, enc)
                embed_reasons = [f"cosine_dist={embed_dists[i]:.3f}" if embed_mask[i] == 1 else "" for i in range(len(embed_mask))]
                df_embed = pd.DataFrame(
                    {
                        "image_path": image_paths,
                        "panoid": panoids,
                        "is_outlier": embed_mask,
                        "outlier_reason": embed_reasons,
                    }
                )
                out_embed_csv = os.path.join(clusters_embed_dir, f"cluster_{int(cid)}_embed.csv")
                df_embed.to_csv(out_embed_csv, index=False)
                visualize_single_cluster(
                    clusters_dir=clusters_embed_dir,
                    cluster_id=int(cid),
                    score=-1,
                    score_rationale=f"Embedding outliers (top {embedding_top_percent:.1f}%)",
                    output_dir=viz_embed_city_dir,
                    images_root="",  # map_reduce didn't pass viz_images_root; use raw paths
                    cols=6,
                    thumb_width=256,
                )

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
        "--device",
        type=int,
        default=None,
        help="GPU device index to use (e.g., 0). Use -1 for CPU. Default: auto select",
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set (single method only), generate collages per cluster during the run",
    )
    parser.add_argument(
        "--viz-output-dir",
        type=str,
        default="",
        help="Directory to save collages (defaults to <output-dir>/collages/<city>)",
    )
    parser.add_argument(
        "--viz-images-root",
        type=str,
        default="",
        help="Optional root for resolving image_path when composing collages",
    )
    parser.add_argument(
        "--viz-cols",
        type=int,
        default=6,
        help="Number of images per row in collages",
    )
    parser.add_argument(
        "--viz-thumb-width",
        type=int,
        default=256,
        help="Thumbnail width for collages",
    )
    parser.add_argument(
        "--randomize-clusters",
        action="store_true",
        help="If set, ignore true clusters and build random-size synthetic clusters per city.",
    )
    parser.add_argument(
        "--random-size-min",
        type=int,
        default=2,
        help="Minimum size of synthetic clusters when --randomize-clusters is used.",
    )
    parser.add_argument(
        "--random-size-max",
        type=int,
        default=10,
        help="Maximum size of synthetic clusters when --randomize-clusters is used.",
    )
    parser.add_argument(
        "--random-system-file",
        type=str,
        default="",
        help="Optional path to a text file with system instructions to use only when --randomize-clusters is set. If omitted, defaults to the built-in SYSTEM_INSTRUCTIONS.",
    )
    parser.add_argument(
        "--random-prompt-file",
        type=str,
        default="",
        help=(
            "Optional path to a text template for the LLM prompt used only when --randomize-clusters is set. "
            "Include the placeholder {descriptions} where a numbered list of descriptions should be inserted. "
            "If omitted, defaults to the built-in cluster/map-reduce prompts."
        ),
    )
    parser.add_argument(
        "--raw-output-dir",
        type=str,
        default="",
        help="If set, write raw LLM prompt and response per cluster (and per chunk for map-reduce) into this directory, organized by city.",
    )
    # Embedding-based outlier detection
    parser.add_argument(
        "--embedding-outliers",
        action="store_true",
        help="If set, compute embedding-based outliers per cluster (top K%% farthest from centroid) and generate collages under collages_embeddings/<city>.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face encoder model for text embeddings (e.g., sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5).",
    )
    parser.add_argument(
        "--embedding-top-percent",
        type=float,
        default=10.0,
        help="Top percentage of farthest descriptions from centroid to mark as outliers (0-100).",
    )
    parser.add_argument(
        "--embedding-device",
        type=int,
        default=None,
        help="GPU device for embeddings (e.g., 0). Use -1 for CPU. Default: auto-select.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding inference.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Write/run log with exact command
    try:
        cmd_str = " ".join(shlex.quote(x) for x in sys.argv)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = os.path.join(args.output_dir, "run_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {cmd_str}\n")
    except Exception:
        pass

    preds_df = load_predictions_csv(args.predictions_csv)
    assigns_df = load_all_assignments(args.clustered_dir, args.city)
    merged = preds_df.merge(assigns_df, on="panoid", how="left", suffixes=("_pred", ""))
    merged = merged.dropna(subset=["cluster_id"])  # keep only images mapped to clusters
    if merged.empty:
        raise SystemExit("No predictions matched to clustered assignments. Check panoid parsing and inputs.")

    # Optional synthetic random clusters mode
    if getattr(args, "randomize_clusters", False):
        # Load optional overrides for system instructions and prompt template
        random_sys: Optional[str] = None
        random_tpl: Optional[str] = None
        use_random_prompts = True
        if getattr(args, "random_system_file", ""):
            try:
                with open(args.random_system_file, "r", encoding="utf-8") as f:
                    random_sys = f.read()
            except Exception as e:
                print(f"Warning: failed to read --random-system-file '{args.random_system_file}': {e}. Using default system instructions.")
                random_sys = None
        if getattr(args, "random_prompt_file", ""):
            try:
                with open(args.random_prompt_file, "r", encoding="utf-8") as f:
                    random_tpl = f.read()
            except Exception as e:
                print(f"Warning: failed to read --random-prompt-file '{args.random_prompt_file}': {e}. Using default prompt.")
                random_tpl = None

        merged = build_random_clusters(
            merged=merged,
            size_min=int(args.random_size_min),
            size_max=int(args.random_size_max),
        )
    else:
        random_sys = None
        random_tpl = None
        use_random_prompts = False

    if args.method == "single":
        out = evaluate_single_shot(
            merged=merged,
            output_dir=args.output_dir,
            model_name=args.model_name,
            temperature=args.temperature,
            dtype=args.dtype,
            max_descriptions=args.max_descriptions,
            device=args.device,
            visualize=args.visualize,
            viz_output_dir=args.viz_output_dir,
            viz_images_root=args.viz_images_root,
            viz_cols=args.viz_cols,
            viz_thumb_width=args.viz_thumb_width,
            system_instructions_override=random_sys,
            prompt_template_override=random_tpl,
            use_random_prompts=use_random_prompts,
            raw_output_dir=args.raw_output_dir,
            # Embedding outliers
            use_embedding_outliers=bool(getattr(args, "embedding_outliers", False)),
            embedding_model_name=str(getattr(args, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            embedding_top_percent=float(getattr(args, "embedding_top_percent", 10.0)),
            embedding_device=int(getattr(args, "embedding_device", args.device)) if getattr(args, "embedding_device", None) is not None else args.device,
            embedding_batch_size=int(getattr(args, "embedding_batch_size", 64)),
        )
    else:
        out = evaluate_map_reduce(
            merged=merged,
            output_dir=args.output_dir,
            model_name=args.model_name,
            temperature=args.temperature,
            dtype=args.dtype,
            chunk_size=args.chunk_size,
            device=args.device,
            system_instructions_override=random_sys,
            prompt_template_override=random_tpl,
            use_random_prompts=use_random_prompts,
            raw_output_dir=args.raw_output_dir,
            # Embedding outliers
            use_embedding_outliers=bool(getattr(args, "embedding_outliers", False)),
            embedding_model_name=str(getattr(args, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            embedding_top_percent=float(getattr(args, "embedding_top_percent", 10.0)),
            embedding_device=int(getattr(args, "embedding_device", args.device)) if getattr(args, "embedding_device", None) is not None else args.device,
            embedding_batch_size=int(getattr(args, "embedding_batch_size", 64)),
        )
    print(f"Wrote global results to: {out}")


if __name__ == "__main__":
    main()



