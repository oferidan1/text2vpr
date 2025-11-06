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
  --model-name microsoft/Phi-3.5-mini-instruct \
  --country London
```

Options:
- `--seg-backend`: `hf_segformer_b2_ade` (default), `torchvision_deeplabv3`, or `hf_oneformer_ade_swinl`
- `--debug-seg`: saves overlay images per input
- `--images-root`: prefix to resolve relative `image_path`
- `--model-name`: HF LLM used to judge alignment and suggest omissions
- `--city` / `--country`: filter rows by the second subdir under `Images/` (e.g., `Images/London/...`)
- `--score-only`: LLM outputs only a single integer (1..5); explanation fields left empty

### Outputs
- Global CSV: `quality_outputs/text_vs_segments_consistency_all.csv`
- Per-city CSV: `quality_outputs/{City}/text_vs_segments_consistency_{City}.csv`
- Overlays: `quality_outputs/{City}/scores/{1|2|3|4|5}/*_overlay_vpr.jpg` (grouped by LLM score)

Columns include:
- `objects_detected`: all segmented classes (unfiltered)
- `objects_detected_relevant`: segmented classes filtered to VPR-relevant
- `llm_score`: 1–5 alignment score (based only on VPR-relevant matches)
- `omit_list`: items to remove from the description (non-VPR or hallucinated)
- `suggested_description`: concise rewrite focusing on place-defining details
- `detected_in_text_ratio`, `text_in_detected_ratio`: overlap heuristics over VPR-relevant items
- `detected_relevant_not_in_text`: relevant segments detected but not described (colored red in overlay)
- `text_relevant_not_detected`: relevant described parts not found by segmentation
### Overlay semantics
- Blue: relevant segments that are also mentioned in the description (match)
- Red: relevant segments present in the image but not mentioned in the description (missing in text)
 - Each overlay is annotated with the image description, and lists for Blue (matched) and Red (detected not in text).


