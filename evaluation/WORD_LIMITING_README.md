# Enhanced Dataset Creator with Smart Word Limiting

## 🆕 New Features

The `dataset_creator.py` has been enhanced with intelligent word limiting that goes beyond simple text truncation. Instead of just cutting off responses, it uses **smart prompt engineering** to encourage more concise and focused descriptions.

## 🧠 Smart Prompt Engineering

### How It Works

Instead of just truncating the response, the system:

1. **Intelligently modifies the prompt** based on the desired word count
2. **Adjusts generation parameters** (e.g., max tokens for InternVL)
3. **Applies post-processing** only if needed
4. **Provides feedback** about truncation when it occurs

### Prompt Modifications by Word Count

#### For Description Tasks (InternVL):
- **≤10 words**: "Give a very brief, 10-word maximum description."
- **≤25 words**: "Be concise. Limit to 25 words maximum."
- **≤50 words**: "Keep it brief, under 50 words."
- **>50 words**: "Be concise and descriptive."

#### For VPR Tasks (Gemini):
- **≤15 words**: "List only the most distinctive features in 15 words max."
- **≤30 words**: "Focus on key identifying features. 30 words maximum."
- **>30 words**: "Emphasize distinctive elements for place recognition."

## 🚀 Usage

### Command Line Interface

```bash
# Basic usage with word limit
python dataset_creator.py --image_path "path/to/image.jpg" --max_words 20

# Use InternVL model with 15 word limit
python dataset_creator.py --image_path "path/to/image.jpg" --model internvl --max_words 15

# No word limit (original behavior)
python dataset_creator.py --image_path "path/to/image.jpg" --max_words 0

# Save output to file
python dataset_creator.py --image_path "path/to/image.jpg" --max_words 25 --output_file "description.txt"
```

### Programmatic Usage

```python
# For Gemini
run_gemini("path/to/image.jpg", max_response_words=20)

# For InternVL
run_intern_vl("path/to/image.jpg", max_response_words=15)
```

## 🎯 Benefits Over Simple Truncation

### 1. **Better Quality Descriptions**
- Models are guided to be concise from the start
- More coherent and complete descriptions within limits
- Better preservation of important information

### 2. **Intelligent Token Management**
- InternVL: Adjusts `max_new_tokens` based on word count
- Gemini: Optimizes prompt for desired response length
- Prevents unnecessary computation

### 3. **Task-Specific Optimization**
- Different prompt strategies for description vs. VPR tasks
- Context-aware word limit suggestions
- Better alignment with use case requirements

### 4. **Transparency**
- Clear feedback when truncation occurs
- Shows original vs. truncated response lengths
- Helps users understand the system's behavior

## 📊 Example Output

```
Processing image: path/to/image.jpg
Model: gemini
Max words: 15
--------------------------------------------------
Inference time: 2.34 seconds

⚠️  Response truncated from 23 to 15 words:
Truncated: modern glass building with distinctive facade, traffic intersection, street signs, urban architecture
```

## 🔧 Technical Details

### InternVL Integration
- Automatically adjusts `max_new_tokens` (1 word ≈ 1.5 tokens)
- Minimum token limit of 10 to ensure basic responses
- Smart prompt engineering for description tasks

### Gemini Integration
- Optimized prompts for VPR-specific tasks
- Post-processing with user feedback
- Maintains response quality within limits

### Fallback Handling
- If smart prompting doesn't achieve desired length, post-processing truncates
- Clear indication when truncation occurs
- Preserves most important information (left-to-right order)

## 🧪 Testing

Run the test script to see the word limiting in action:

```bash
python test_dataset_creator.py
```

This will test different word limits and show how the system adapts prompts and responses.

## 💡 Best Practices

1. **Start with higher limits** (25-30 words) and reduce as needed
2. **Use task-appropriate models**: Gemini for VPR, InternVL for general description
3. **Monitor truncation warnings** to understand when limits are too restrictive
4. **Combine with output files** for batch processing and analysis

## 🔄 Backward Compatibility

The original functionality is preserved:
- No command line arguments = original behavior
- `max_words=0` = no limit
- All existing function calls work unchanged
