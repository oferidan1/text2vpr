# LLM Filtering Guide for Object Detection

## Overview

The visual checker now supports a **hybrid approach** for extracting segmentable items:
- **Fast extraction** using spaCy noun phrase parser
- **Smart filtering** using Qwen2.5-3B-Instruct LLM

This removes non-segmentable items (abstract concepts, qualities, actions) and keeps items suitable for **SAM (Segment Anything Model)**:
- ✅ **Objects** (things): car, person, building, tree, bench
- ✅ **Regions** (stuff): sky, road, grass, pavement, water, wall

## Quick Start

### Option 1: Basic NP Parsing (Fast)
Uses noun phrase parser only:
```bash
python3 main.py \
  --input_dir /mnt/d/dan/git_projects/text2vpr/validation_embddings/outputs_manual_thr/London \
  --output_dir /tmp/visual_checker_debug_np \
  --use_noun_phrase_parser
```

### Option 2: NP + LLM Filtering (Combined)
Parse and filter in one step:
```bash
python3 main.py \
  --input_dir /mnt/d/dan/git_projects/text2vpr/validation_embddings/outputs_manual_thr/London \
  --output_dir /tmp/visual_checker_debug_np_filtered \
  --use_noun_phrase_parser \
  --filter_with_llm
```

### Option 3: Two-Step Workflow (Flexible) ⭐
First parse (fast), then filter later (when needed):

```bash
# Step 1: Fast NP parsing (run once)
python3 main.py \
  --input_dir /path/to/data \
  --use_noun_phrase_parser

# Step 2: Apply filtering to existing CSVs (run when needed)
python3 main.py \
  --input_dir /path/to/data \
  --filter_existing_np
```

**Benefits of two-step:**
- Run expensive parsing only once
- Apply filtering only where needed
- Try different filter models easily
- Compare filtered vs unfiltered side-by-side

See [FILTER_EXISTING_GUIDE.md](FILTER_EXISTING_GUIDE.md) for details.

## How It Works

### Without Filtering
```
Caption: "A beautiful street with tall buildings and bright sky"
↓ (spaCy NP Parser)
Output: "street. building-tall. sky. beauty-bright"
❌ Includes "beauty-bright" (not segmentable)
```

### With Filtering
```
Caption: "A beautiful street with tall buildings and bright sky"
↓ (spaCy NP Parser)
Intermediate: "street. building-tall. sky. beauty-bright"
↓ (Qwen2.5-3B LLM Filter)
Output: "street. building. sky"
✅ Only concrete, segmentable items (objects + regions)
   • "street", "building" = objects (things)
   • "sky" = region (stuff)
   • "beauty-bright" = removed (abstract quality)
```

## Command Options

### Core Options
- `--use_noun_phrase_parser` - Use spaCy for fast extraction (required for filtering)
- `--filter_with_llm` - Enable LLM filtering (only works with noun phrase parser)
- `--filter_model` - Specify filter model (default: `Qwen/Qwen2.5-3B-Instruct`)

### Advanced Options
- `--filter_prompt_template` - Custom filtering prompt (use `{object_list}` placeholder)

## Recommended Models

### Default (Best Choice)
- **Qwen/Qwen2.5-3B-Instruct** ⭐
  - Excellent quality-to-speed ratio
  - 3B parameters
  - Best for most use cases

### Alternatives
- **microsoft/Phi-3.5-mini-instruct**
  - 3.8B parameters
  - Already used in your project
  - Good quality
  ```bash
  --filter_model microsoft/Phi-3.5-mini-instruct
  ```

- **google/gemma-2-2b-it**
  - 2B parameters
  - Fastest inference
  - Good for simple filtering
  ```bash
  --filter_model google/gemma-2-2b-it
  ```

## Output Files

### File Naming
- Without filtering: `cluster_items_objects_np.csv`
- With filtering: `cluster_items_objects_np_filtered.csv`

### CSV Structure
Same as before, but with cleaner object lists:
```csv
image_path,description,objects
/path/to/img.jpg,"A beautiful street...",street. building. sky
```

## Performance Considerations

- **Noun phrase parsing**: ~0.01s per caption (very fast)
- **LLM filtering**: ~0.5-2s per caption (depends on model and hardware)
- **Recommended**: Use filtering when quality matters more than speed

## Examples

### Example 1: Single CSV
```bash
python3 main.py \
  --input_csv /path/to/cluster_items.csv \
  --output_dir /tmp/filtered_objects \
  --use_noun_phrase_parser \
  --filter_with_llm
```

### Example 2: Directory Scan with Custom Model
```bash
python3 main.py \
  --input_dir /mnt/d/data/captions \
  --output_dir /tmp/filtered_objects \
  --use_noun_phrase_parser \
  --filter_with_llm \
  --filter_model google/gemma-2-2b-it
```

### Example 3: Without Filtering (Faster)
```bash
python3 main.py \
  --input_dir /mnt/d/data/captions \
  --output_dir /tmp/np_objects \
  --use_noun_phrase_parser
```

## Troubleshooting

### Issue: "spaCy is required..."
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Issue: "Model not found"
The first time you use a filter model, it will be downloaded from Hugging Face.
Ensure you have:
- Internet connection
- Sufficient disk space (~6GB for Qwen2.5-3B)

### Issue: Out of memory
Try a smaller model:
```bash
--filter_model google/gemma-2-2b-it
```

## Best Practices

1. **Use filtering for final production runs** where quality matters
2. **Skip filtering during development/testing** for faster iterations
3. **Experiment with different models** to find the best speed/quality trade-off
4. **Check output files** to verify filtering is working as expected

## Integration with Segmentation Models

The filtered items can be directly fed to:
- **SAM (Segment Anything Model)** ⭐ - segments both objects and regions
- **SAM2** - improved segmentation model
- **OWL-ViT** (open-vocabulary object detection)
- **CLIPSeg** (open-vocabulary segmentation)
- **GroundingDINO** (zero-shot object detection)
- Any other open-vocabulary segmentation model

Example workflow with SAM:
```bash
# Step 1: Extract and filter items
python3 main.py --input_dir /data/captions --use_noun_phrase_parser --filter_with_llm

# Step 2: Use filtered items with SAM
# The CSV now contains clean, segmentable items (objects + regions) ready for SAM
# Use the 'filtered_by_llm' column as input prompts to SAM
```

