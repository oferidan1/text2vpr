## Intra-Place Textual Consistency Judge

This module evaluates how consistent the generated descriptions are for images within the same clustered place/viewpoint.

### Inputs
- Clustered assignments per city (from `parse_gsv_cities.py`): `Dataframes_clustered/{City}.csv` with columns including `panoid`, `cluster_id`, `city_id`.
- Predictions CSV with descriptions (example: `gsv_cities_predictions_nan_fix.csv`) with columns:
  - `image_path` — e.g., `Images/London/London_0001938_..._nZ0J9gVZM59D70Su1eG6jg.jpg`
  - `description` — generated text

The judge extracts `panoid` from `image_path` and joins with clustered assignments.

### Installation
Requires recent `transformers`, `tqdm`, and a working PyTorch install.

```bash
pip install transformers accelerate torch tqdm
```

### CLI Usage

Run the judge for all cities (shows a per-city progress bar over clusters):

```bash
python /mnt/d/dan/git_projects/text2vpr/dataset_builder/validation/run_intra_place_judge.py \
  --predictions-csv /mnt/d/dan/git_projects/text2vpr/gsv_cities_predictions_nan_fix.csv \
  --clustered-dir /mnt/d/dan/git_projects/text2vpr/Dataframes_clustered \
  --output-dir /mnt/d/dan/git_projects/text2vpr/validation_outputs \
  --model-name microsoft/Phi-3.5-mini-instruct \
  --max-descriptions 20 --temperature 0.2
```

Single city only (matches `city_id`):

```bash
python /mnt/d/dan/git_projects/text2vpr/dataset_builder/validation/run_intra_place_judge.py \
  --predictions-csv /mnt/d/dan/git_projects/text2vpr/gsv_cities_predictions_nan_fix.csv \
  --clustered-dir /mnt/d/dan/git_projects/text2vpr/Dataframes_clustered \
  --output-dir /mnt/d/dan/git_projects/text2vpr/validation_outputs \
  --city London
```

### Outputs
- Global results: `validation_outputs/intra_place_consistency_all.csv`
- Per-city results: `validation_outputs/{City}/intra_place_consistency_{City}.csv`
- Per-cluster image lists: `validation_outputs/{City}/clusters/cluster_{cluster_id}.csv` (columns: `image_path`, `panoid`, `description`)

Each result row contains an integer `score` in [1..5] and a short `explanation` of inconsistencies if any.

### Model configuration
The default is `microsoft/Phi-3.5-mini-instruct`. Any compatible HF causal LM can be used via `--model-name`. Use `--dtype float16` (or `bfloat16`) for GPU-accelerated inference if supported.


