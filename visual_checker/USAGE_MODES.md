# Visual Checker - Usage Modes Summary

Quick reference for all available modes and when to use them.

## Mode Comparison

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| **LLM Only** | Slow | High | Need best quality, have GPU/time |
| **NP Only** | Very Fast | Medium | Quick extraction, speed matters |
| **NP + Filter** | Slow | High | One-step quality extraction |
| **Two-Step** | Fast+Slow | High | **Recommended: Flexible workflow** |

## Mode 1: LLM Only (Original)

**Extract objects using LLM directly:**

```bash
# High-quality (default preset, Phi-3.5)
python3 main.py \
  --input_dir /path/to/data \
  --output_dir /tmp/llm_output

# Faster, smaller model (Gemma 2B)
python3 main.py \
  --input_dir /path/to/data \
  --output_dir /tmp/llm_output_fast \
  --model_preset fast
```

**Creates:** `cluster_items_objects_llm.csv`

**Pros:**
- High quality extraction
- Understands context well

**Cons:**
- Slow (~1-2s per caption)
- Requires GPU for reasonable speed

**Use when:**
- You need the absolute best quality
- Speed is not a concern
- You have GPU available

---

## Mode 2: Noun Phrase Parser Only (Fast)

**Extract objects using spaCy NP parser:**

```bash
python3 main.py \
  --input_dir /path/to/data \
  --output_dir /tmp/np_output \
  --use_noun_phrase_parser
```

**Creates:** `cluster_items_objects_np.csv`

**Pros:**
- Very fast (~0.01s per caption)
- No GPU needed
- Good for quick iterations

**Cons:**
- May include non-detectable terms
- Less context-aware than LLM

**Use when:**
- You need speed over perfect quality
- Prototyping/development
- CPU-only environment

---

## Mode 3: NP + LLM Filtering (Combined)

**Extract with NP, filter with LLM in one step:**

```bash
python3 main.py \
  --input_dir /path/to/data \
  --output_dir /tmp/filtered_output \
  --use_noun_phrase_parser \
  --filter_with_llm
```

**Creates:** `cluster_items_objects_np_filtered.csv`

**Pros:**
- High quality (LLM filtering)
- Single command
- Clean output

**Cons:**
- Slow (combines both steps)
- Must rerun both steps to change filtering

**Use when:**
- You want quality in a single command
- Final production run
- Don't need to iterate on filtering

---

## Mode 4: Two-Step Workflow ⭐ **RECOMMENDED**

**Step 1: Extract with NP (fast, run once):**

```bash
python3 main.py \
  --input_dir /path/to/data \
  --output_dir /tmp/np_output \
  --use_noun_phrase_parser
```

**Step 2: Filter existing CSVs (when needed):**

```bash
python3 main.py \
  --input_dir /path/to/data \
  --filter_existing_np
```

**Creates:** Updates `cluster_items_objects_np.csv` with `filtered_by_llm` column

**Pros:**
- ✅ Maximum flexibility
- ✅ Fast NP parsing runs only once
- ✅ Can filter selectively (only some directories)
- ✅ Can try different filter models easily
- ✅ Can compare filtered vs unfiltered
- ✅ Can re-filter without re-parsing

**Cons:**
- Two commands instead of one

**Use when:**
- **Recommended for most workflows!**
- You have lots of data to process
- You want to try different filter models
- You want flexibility

---

## Command Options Reference

### Core Options

| Flag | Description | Modes |
|------|-------------|-------|
| `--input_dir` | Directory to search for CSVs | All |
| `--input_csv` | Single CSV file | All |
| `--output_dir` | Where to save debug files | 1, 2, 3 |
| `--model` | Main LLM model (overrides preset) | 1, 3 |
| `--model_preset` | Quality/speed preset for main LLM (`quality`, `fast`) | 1, 3 |
| `--use_noun_phrase_parser` | Use spaCy NP parser | 2, 3 |
| `--filter_with_llm` | Filter NP output with LLM | 3 |
| `--filter_existing_np` | Filter existing `*_np.csv` files | 4 |
| `--filter_model` | Model for filtering | 3, 4 |

### Output Files

| Mode | Output File | Contains |
|------|-------------|----------|
| 1. LLM Only | `cluster_items_objects_llm.csv` | LLM-extracted objects |
| 2. NP Only | `cluster_items_objects_np.csv` | NP-extracted objects (unfiltered) |
| 3. NP + Filter | `cluster_items_objects_np_filtered.csv` | Filtered objects only |
| 4. Two-Step | `cluster_items_objects_np.csv` | Both `objects` and `filtered_by_llm` columns |

---

## Workflow Recommendations

### For Development/Testing
```bash
# Use fast NP parser only
python3 main.py --input_dir /data --use_noun_phrase_parser
```

### For Production (One-time)
```bash
# Use combined NP + filtering
python3 main.py --input_dir /data --use_noun_phrase_parser --filter_with_llm
```

### For Production (Iterative) ⭐
```bash
# Step 1: Parse once
python3 main.py --input_dir /data --use_noun_phrase_parser

# Step 2: Filter as needed
python3 main.py --input_dir /data --filter_existing_np

# Optional: Try different model
python3 main.py --input_dir /data --filter_existing_np --filter_model google/gemma-2-2b-it
```

### For Large Datasets
```bash
# Parse all data once (fast)
python3 main.py --input_dir /all_data --use_noun_phrase_parser

# Filter only specific subsets (saves time)
python3 main.py --input_dir /all_data/subset1 --filter_existing_np
python3 main.py --input_dir /all_data/subset2 --filter_existing_np
```

---

## Filter Model Options

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **Qwen/Qwen2.5-3B-Instruct** ⭐ | 3B | Fast | Excellent | **Default, best choice** |
| microsoft/Phi-3.5-mini-instruct | 3.8B | Fast | Very Good | Alternative |
| google/gemma-2-2b-it | 2B | Faster | Good | Speed priority |

**Usage:**
```bash
--filter_model Qwen/Qwen2.5-3B-Instruct  # Default
--filter_model google/gemma-2-2b-it      # Faster
```

---

## Examples

### Example 1: Quick Development Iteration

```bash
# Fast extraction for testing
python3 main.py \
  --input_csv /data/test/cluster_items.csv \
  --output_dir /tmp/test \
  --use_noun_phrase_parser
```

### Example 2: Production with Quality Filtering

```bash
# Extract + filter in one go
python3 main.py \
  --input_dir /data/production \
  --output_dir /tmp/prod_filtered \
  --use_noun_phrase_parser \
  --filter_with_llm
```

### Example 3: Large Dataset with Selective Filtering

```bash
# Parse everything once (fast)
python3 main.py \
  --input_dir /data/all_cities \
  --use_noun_phrase_parser

# Filter only the cities you need (saves time)
python3 main.py \
  --input_dir /data/all_cities/London \
  --filter_existing_np

python3 main.py \
  --input_dir /data/all_cities/Paris \
  --filter_existing_np
```

### Example 4: Compare Different Filter Models

```bash
# Parse once
python3 main.py --input_dir /data --use_noun_phrase_parser

# Try Qwen (best quality)
python3 main.py --input_dir /data --filter_existing_np --filter_model Qwen/Qwen2.5-3B-Instruct

# Compare with Gemma (faster)
python3 main.py --input_dir /data --filter_existing_np --filter_model google/gemma-2-2b-it

# Now compare the results and choose the best model for your use case
```

---

## Error Messages and Solutions

### "No 'cluster_items.csv' files found"
**Solution:** Check your `--input_dir` path

### "No '*_objects_np.csv' files found"
**Solution:** Run NP parsing first:
```bash
python3 main.py --input_dir /data --use_noun_phrase_parser
```

### "Input CSV must end with '_objects_np.csv'"
**Solution:** You're using `--filter_existing_np` with the wrong CSV type. It only works with files created by NP parsing.

### "spaCy is required..."
**Solution:** Install spaCy:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## Performance Comparison

For 1000 captions on typical hardware:

| Mode | Time | GPU Required |
|------|------|--------------|
| LLM Only | ~30-60 min | Recommended |
| NP Only | ~10 sec | No |
| NP + Filter (combined) | ~10-30 min | Recommended |
| Two-Step (NP) | ~10 sec | No |
| Two-Step (Filter) | ~10-30 min | Recommended |

**Total time is similar for combined vs two-step, but two-step offers flexibility!**

---

## Documentation

- **[LLM_FILTERING_GUIDE.md](LLM_FILTERING_GUIDE.md)** - Complete filtering guide
- **[FILTER_EXISTING_GUIDE.md](FILTER_EXISTING_GUIDE.md)** - Two-step workflow details
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Smart features (dynamic tokens, END marker)
- **[SAFEGUARDS_SUMMARY.md](SAFEGUARDS_SUMMARY.md)** - How we prevent LLM chattiness
- **[example_two_step_workflow.sh](example_two_step_workflow.sh)** - Runnable example

---

## Quick Decision Tree

```
Do you need filtered objects?
├─ No → Use Mode 2 (NP Only)
└─ Yes → Are you iterating/experimenting?
    ├─ Yes → Use Mode 4 (Two-Step) ⭐
    └─ No → Use Mode 3 (NP + Filter combined)

Do you have a GPU?
├─ No → Don't use Mode 1 (too slow on CPU)
└─ Yes → Any mode works

Do you have lots of data?
├─ Yes → Use Mode 4 (Two-Step) for flexibility ⭐
└─ No → Any mode works
```

---

## Summary

🏆 **Recommended for most users:** Mode 4 (Two-Step Workflow)

✅ Fast initial parsing  
✅ Flexible filtering  
✅ Compare filtered vs unfiltered  
✅ Try different models easily  
✅ Maximum control  

```bash
# Parse once, filter when needed
python3 main.py --input_dir /data --use_noun_phrase_parser
python3 main.py --input_dir /data --filter_existing_np
```

