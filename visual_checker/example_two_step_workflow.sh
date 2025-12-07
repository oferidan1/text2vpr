#!/bin/bash
#
# Example: Two-Step Workflow for NP Parsing + LLM Filtering
# 
# This demonstrates the recommended workflow:
# 1. Run fast NP parsing once
# 2. Apply LLM filtering later (flexible, can re-run with different models)
#

set -e  # Exit on error

DATA_DIR="/mnt/d/dan/git_projects/text2vpr/validation_embddings/outputs_manual_thr/London"
OUTPUT_DIR="/tmp/visual_checker_example"

echo "============================================"
echo "Two-Step Workflow Example"
echo "============================================"
echo ""

# ============================================
# Step 1: Fast NP Parsing (run once)
# ============================================
echo "Step 1: Running fast noun phrase parsing..."
echo "This extracts objects quickly without LLM"
echo ""

python3 main.py \
  --input_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --use_noun_phrase_parser

echo ""
echo "✓ Step 1 complete!"
echo "  Created: cluster_items_objects_np.csv files"
echo "  These contain unfiltered object lists in the 'objects' column"
echo ""
sleep 2

# ============================================
# Step 2a: Apply LLM Filtering (default model)
# ============================================
echo "Step 2a: Applying LLM filtering with Qwen2.5-3B..."
echo "This adds a 'filtered_by_llm' column to existing CSVs"
echo ""

python3 main.py \
  --input_dir "$DATA_DIR" \
  --filter_existing_np

echo ""
echo "✓ Step 2a complete!"
echo "  Updated: cluster_items_objects_np.csv files"
echo "  Added: 'filtered_by_llm' column with clean, detectable objects"
echo ""
sleep 2

# ============================================
# Optional: Try different filter model
# ============================================
echo "Step 2b (optional): Re-filtering with a different model..."
echo "This demonstrates trying a faster model"
echo ""

python3 main.py \
  --input_dir "$DATA_DIR" \
  --filter_existing_np \
  --filter_model google/gemma-2-2b-it

echo ""
echo "✓ Step 2b complete!"
echo "  Updated: 'filtered_by_llm' column with results from gemma-2-2b"
echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "Workflow Complete!"
echo "============================================"
echo ""
echo "Your CSV files now contain:"
echo "  • 'objects' column: Original NP-extracted objects (unfiltered)"
echo "  • 'filtered_by_llm' column: Clean, detectable objects only"
echo ""
echo "Benefits of this two-step approach:"
echo "  ✓ Fast NP parsing runs only once"
echo "  ✓ Can filter different datasets separately"
echo "  ✓ Can try multiple filter models easily"
echo "  ✓ Can compare filtered vs unfiltered"
echo ""
echo "Next steps:"
echo "  • Use the 'filtered_by_llm' column for your object detection model"
echo "  • Compare columns to see what was filtered out"
echo "  • Re-run Step 2 anytime with different models or prompts"
echo ""

