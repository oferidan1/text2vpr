# Filter Existing NP CSV Files Guide

## Overview

This feature allows you to apply LLM filtering to **already-processed** CSV files created with `--use_noun_phrase_parser`, without re-running the noun phrase extraction.

The filtered list is optimized for **SAM (Segment Anything Model)** and includes:
- ✅ **Objects** (things): car, person, building, tree, etc.
- ✅ **Regions** (stuff): sky, road, grass, pavement, water, etc.
- ❌ **Abstract concepts**: beauty, happiness, importance
- ❌ **Actions/qualities**: running, brightness, moving

## Why This is Useful

**Two-Step Workflow:**

1. **Step 1: Fast NP Parsing** (run once)
   ```bash
   python3 main.py \
     --input_dir /path/to/data \
     --use_noun_phrase_parser
   ```
   Creates: `cluster_items_objects_np.csv` files (fast!)

2. **Step 2: LLM Filtering** (run when needed)
   ```bash
   python3 main.py \
     --input_dir /path/to/data \
     --filter_existing_np
   ```
   Adds `filtered_by_llm` column to existing CSVs

**Benefits:**
- ✅ Run expensive NP parsing only once
- ✅ Apply filtering later (or multiple times with different models)
- ✅ Compare filtered vs unfiltered side-by-side
- ✅ No need to reprocess if you want to try different filter models

## Usage

### Single CSV File

Filter a specific `*_objects_np.csv` file:

```bash
python3 main.py \
  --input_csv /path/to/cluster_items_objects_np.csv \
  --filter_existing_np
```

### Directory Mode (Recommended)

Find and filter all `*_objects_np.csv` files recursively:

```bash
python3 main.py \
  --input_dir /path/to/data \
  --filter_existing_np
```

This will:
1. Search recursively for all `*_objects_np.csv` files
2. For each file, read the `objects` column
3. Apply LLM filtering to each row
4. Add/update `filtered_by_llm` column
5. Save back to the same CSV

### Custom Filter Model

Use a different model for filtering:

```bash
python3 main.py \
  --input_dir /path/to/data \
  --filter_existing_np \
  --filter_model google/gemma-2-2b-it
```

## How It Works

### Before Running `--filter_existing_np`

Your CSV looks like this:

```csv
image_path,description,objects
/path/img1.jpg,"A street scene","street. building. car. beauty. sky"
/path/img2.jpg,"A park view","tree. grass. person. happiness. bench"
```

### After Running `--filter_existing_np`

A new column `filtered_by_llm` is added:

```csv
image_path,description,objects,filtered_by_llm
/path/img1.jpg,"A street scene","street. building. car. beauty. sky","street. building. car. sky"
/path/img2.jpg,"A park view","tree. grass. person. happiness. bench","tree. grass. person. bench"
```

**Notice:**
- `objects` column remains unchanged (unfiltered NP output)
- `filtered_by_llm` column contains cleaned, detectable objects only
- Abstract concepts removed: "beauty", "happiness"

## Complete Workflow Example

### Step 1: Initial NP Parsing

```bash
cd /mnt/d/dan/git_projects/text2vpr/visual_checker

python3 main.py \
  --input_dir /mnt/d/dan/git_projects/text2vpr/validation_embddings/outputs_manual_thr/London \
  --output_dir /tmp/visual_checker_np \
  --use_noun_phrase_parser
```

**Output:** Creates `cluster_items_objects_np.csv` files with NP-extracted objects

### Step 2: Apply LLM Filtering

```bash
python3 main.py \
  --input_dir /mnt/d/dan/git_projects/text2vpr/validation_embddings/outputs_manual_thr/London \
  --filter_existing_np
```

**Output:** Updates all `*_objects_np.csv` files with `filtered_by_llm` column

### Step 3: Use Filtered Results

Now your CSVs have both columns, and you can:
- Use `filtered_by_llm` for your object detection model
- Compare `objects` vs `filtered_by_llm` to see what was filtered
- Re-run with different filter models if needed

## Command Options

| Option | Required | Description |
|--------|----------|-------------|
| `--input_csv` | One of csv/dir | Path to single `*_objects_np.csv` file |
| `--input_dir` | One of csv/dir | Directory to search for `*_objects_np.csv` files |
| `--filter_existing_np` | Yes | Enable this mode |
| `--filter_model` | No | Filter model (default: `Qwen/Qwen2.5-3B-Instruct`) |
| `--filter_prompt_template` | No | Custom filter prompt |

## Important Notes

### File Requirements

- ✅ CSV must end with `_objects_np.csv`
- ✅ CSV must have an `objects` column
- ❌ Cannot use this with regular CSV files (not `*_objects_np.csv`)

### Column Behavior

- If `filtered_by_llm` column doesn't exist → it will be created
- If `filtered_by_llm` column exists → it will be **overwritten**

This allows you to re-run with different filter models.

### Safety

- Original file is updated atomically (using temp file + rename)
- If filtering fails mid-process, original file remains unchanged
- `objects` column is **never modified**

## Error Messages

### "No '*_objects_np.csv' files found"

You need to run NP parsing first:

```bash
python3 main.py \
  --input_dir /path/to/data \
  --use_noun_phrase_parser
```

### "Input CSV must end with '_objects_np.csv'"

You specified a regular CSV file. This mode only works with NP-generated CSVs:

```bash
# Wrong:
--input_csv cluster_items.csv

# Correct:
--input_csv cluster_items_objects_np.csv
```

### "CSV must have an 'objects' column"

The CSV is malformed or not from NP parsing. Check the file structure.

## Comparison with Other Modes

### `--use_noun_phrase_parser --filter_with_llm`

**Creates new CSV with filtering already applied:**
```bash
python3 main.py --use_noun_phrase_parser --filter_with_llm --input_dir /data
```

**Result:** `cluster_items_objects_np_filtered.csv` with filtered objects only

**Use when:** You want to filter during initial processing

### `--filter_existing_np`

**Updates existing CSV by adding column:**
```bash
python3 main.py --filter_existing_np --input_dir /data
```

**Result:** `cluster_items_objects_np.csv` with both `objects` and `filtered_by_llm` columns

**Use when:** 
- You already ran NP parsing
- You want to compare filtered vs unfiltered
- You want to try different filter models

## Advanced Usage

### Different Filter Models for Different Directories

```bash
# Fast model for development data
python3 main.py \
  --input_dir /data/dev \
  --filter_existing_np \
  --filter_model google/gemma-2-2b-it

# Best model for production data
python3 main.py \
  --input_dir /data/prod \
  --filter_existing_np \
  --filter_model Qwen/Qwen2.5-3B-Instruct
```

### Re-filter with Different Model

Already filtered but want to try a different model?

```bash
# First run (creates filtered_by_llm column)
python3 main.py --input_dir /data --filter_existing_np

# Change your mind, use different model (overwrites filtered_by_llm column)
python3 main.py \
  --input_dir /data \
  --filter_existing_np \
  --filter_model microsoft/Phi-3.5-mini-instruct
```

### Custom Filter Prompt

```bash
python3 main.py \
  --input_dir /data \
  --filter_existing_np \
  --filter_prompt_template "Keep only outdoor objects from: {object_list}"
```

## Performance

### Speed Comparison

| Mode | Speed | Use Case |
|------|-------|----------|
| NP Parser Only | Very Fast (~0.01s/caption) | Initial extraction |
| NP + Filter (combined) | Slow (~0.5-2s/caption) | One-step workflow |
| Filter Existing | Slow (~0.5-2s/caption) | Re-filter after NP |

**Recommendation:**
1. Run NP parser on all data (fast, do once)
2. Use `--filter_existing_np` on subset you need (flexible)

### Example Timing

For 1000 captions:

```
NP parsing:          ~10 seconds
LLM filtering:       ~500-2000 seconds (8-33 minutes)
Total (combined):    ~510-2010 seconds
Total (two-step):    ~510-2010 seconds

Advantage of two-step: Can run NP once, filter multiple times or on subsets
```

## Troubleshooting

### Issue: Column keeps getting overwritten

**Expected behavior!** The `filtered_by_llm` column will be overwritten each time you run `--filter_existing_np`.

If you want to preserve previous results:
1. Copy the CSV before re-running
2. Rename the column manually before re-running

### Issue: Too slow

Try a smaller/faster model:

```bash
--filter_model google/gemma-2-2b-it
```

Or filter only specific directories:

```bash
python3 main.py \
  --input_csv /path/to/specific/cluster_items_objects_np.csv \
  --filter_existing_np
```

### Issue: Some rows have empty filtered_by_llm

This can happen if:
- Original `objects` column was empty
- LLM filtered out all objects (all were non-detectable)

Check the original `objects` column to understand why.

## Summary

✅ **Use `--filter_existing_np` when you want to:**
- Apply filtering to already-processed NP CSV files
- Compare filtered vs unfiltered side-by-side
- Try different filter models
- Separate fast NP parsing from slower LLM filtering

✅ **Workflow:**
```
Step 1: python3 main.py --use_noun_phrase_parser --input_dir /data
        → Creates cluster_items_objects_np.csv files

Step 2: python3 main.py --filter_existing_np --input_dir /data
        → Adds filtered_by_llm column to existing files

Step 3: Use the filtered_by_llm column for your object detection!
```

🎯 **Maximum flexibility: Fast initial processing, filter when/where needed!**

