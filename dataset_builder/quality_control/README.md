## Text vs Segmentation Objects Consistency

This pipeline validates image descriptions against objects detected by a semantic segmentation model.

### Installation
Requires PyTorch, transformers (for default SegFormer backend), and tqdm.

```bash
pip install torch torchvision transformers tqdm pillow
```

### CLI

```bash
python /mnt/d/dan/git_projects/text2vpr/dataset_builder/quality_control/run_text_vs_objects_judge.py \
  --predictions-csv /mnt/d/data/gsv_cities/gsv_cities_predictions_nan_fix.csv \
  --images-root /mnt/d/data/gsv_cities \
  --seg-backend hf_segformer_b2_ade \
  --debug-seg \
  --output-dir /mnt/d/dan/git_projects/text2vpr/quality_outputs \
  --model-name microsoft/Phi-3.5-mini-instruct
```

Options:
- `--seg-backend`: `hf_segformer_b2_ade` (default) or `torchvision_deeplabv3`
- `--debug-seg`: saves overlay images per input
- `--images-root`: prefix to resolve relative `image_path`
- `--model-name`: HF LLM used to judge alignment and suggest omissions

### Outputs
- Global CSV: `quality_outputs/text_vs_segments_consistency_all.csv`
- Per-city CSV: `quality_outputs/{City}/text_vs_segments_consistency_{City}.csv`
- Debug overlays (if enabled): `quality_outputs/{City}/debug_overlays/*.jpg`

Columns include:
- `objects_detected`: comma-separated objects from segmentation
- `llm_score`: 1–5 alignment score
- `omit_list`: items to remove from the description
- `suggested_description`: non-destructive rewrite suggestion
- `detected_in_text_ratio`, `text_in_detected_ratio`: simple overlap heuristics


