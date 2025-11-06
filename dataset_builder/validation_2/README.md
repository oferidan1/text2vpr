## Pairwise VPR Text Consistency Judge (validation_2)

Evaluates pairwise textual consistency within each clustered place to surface overlap, critical/noncritical inconsistencies, and a 1–5 VPR-consistency score per pair. Aggregates median and mean scores per cluster.

### Inputs
- Clustered assignments per city: `Dataframes_clustered/{City}.csv` with `panoid,cluster_id,city_id`.
- Predictions CSV: `image_path,description` with paths containing `Images/<City>/...` and a filename ending with `_PANOID.jpg`.

### CLI Usage
```bash
python /mnt/d/dan/git_projects/text2vpr/dataset_builder/validation_2/run_pairwise_place_judge.py \
  --predictions-csv /mnt/d/data/gsv_cities/gsv_cities_predictions_nan_fix.csv \
  --clustered-dir /mnt/d/dan/git_projects/text2vpr/Dataframes_clustered_debug \
  --output-dir /mnt/d/dan/git_projects/text2vpr/validation_outputs \
  --city London \
  --model-name microsoft/Phi-3.5-mini-instruct \
  --max-pairs-per-cluster 120
```

Flags:
- `--include-descriptions-in-pairs` to also store the raw descriptions per pair.
- `--dtype {float16,bfloat16}` to accelerate on supported GPUs.

### Outputs
- Global summary: `validation_outputs/pairwise_consistency_all.csv`
- Per-city summary: `validation_outputs/{City}/pairwise_consistency_{City}.csv`
- Per-cluster pairs: `validation_outputs/{City}/clusters/cluster_{cluster_id}_pairs.csv`

Per-pair columns:
- `image_i,image_j,panoid_i,panoid_j`
- `overlap` (JSON list), `critical_inconsistencies` (JSON list), `noncritical_inconsistencies` (JSON list)
- `score` (1–5), `rationale`
- optional: `description_i,description_j`

Per-cluster row:
- `city_id,cluster_id,num_images,num_pairs_evaluated,cluster_score_median,cluster_score_mean,cluster_pairs_csv`

## Methods
- single: single-shot cluster judge over all (or sampled) descriptions.
  - Outputs per-cluster outlier CSV: `cluster_{cluster_id}_single.csv` and per-city/global summaries in `cluster_consistency_*.csv`.
- map_reduce: minibatch map step with programmatic reduce.
  - Outputs per-cluster outlier CSV: `cluster_{cluster_id}_mapreduce.csv` and per-city/global summaries in `cluster_consistency_*.csv`.

### Cluster runner usage
```bash
python /mnt/d/dan/git_projects/text2vpr/dataset_builder/validation_2/run_text_cluster_judge.py \
  --predictions-csv /mnt/d/data/gsv_cities/gsv_cities_predictions_nan_fix.csv \
  --clustered-dir /mnt/d/dan/git_projects/text2vpr/Dataframes_clustered_debug \
  --output-dir /mnt/d/dan/git_projects/text2vpr/validation_outputs_2 \
  --city London \
  --method single \
  --max-descriptions 60
```

Map-reduce:
```bash
python /mnt/d/dan/git_projects/text2vpr/dataset_builder/validation_2/run_text_cluster_judge.py \
  --predictions-csv /mnt/d/data/gsv_cities/gsv_cities_predictions_nan_fix.csv \
  --clustered-dir /mnt/d/dan/git_projects/text2vpr/Dataframes_clustered_debug \
  --output-dir /mnt/d/dan/git_projects/text2vpr/validation_outputs_2 \
  --city London \
  --method map_reduce \
  --chunk-size 16
```

Per-city/global CSV schema:
- single: `city_id,cluster_id,num_images,num_used_descriptions,cluster_score,overlap_themes,critical_inconsistencies,noncritical_inconsistencies,outliers_csv`
- map_reduce: `city_id,cluster_id,num_images,num_chunks,cluster_score_median,cluster_score_mean,overlap_themes,critical_inconsistencies,noncritical_inconsistencies,outliers_csv`


