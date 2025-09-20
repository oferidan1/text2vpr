# BLIP Integration for Text-to-Image Retrieval

This document describes the integration of BLIP (Bootstrapping Language-Image Pre-training) models as an alternative to CLIP for handling longer text descriptions in the text-to-image retrieval system.

## Overview

BLIP models are particularly useful when dealing with longer text descriptions that exceed CLIP's token limits (77 tokens). BLIP can handle much longer text inputs (up to 512 tokens typically) while maintaining good performance for image-text retrieval tasks.

## Key Features

- **Long Text Support**: Handle text descriptions up to 2000 characters (vs CLIP's ~300 characters)
- **Better Context Understanding**: BLIP's architecture is designed for more complex text-image relationships
- **Seamless Integration**: Drop-in replacement for CLIP with the same API
- **GPU Acceleration**: Optimized batch processing for large image databases
- **Model Flexibility**: Support for different BLIP model sizes

## Installation

### 1. Install BLIP Dependencies

```bash
# Install additional requirements for BLIP
pip install -r requirements_blip.txt

# Or install individually
pip install transformers>=4.21.0 torch>=1.12.0 torchvision>=0.13.0
```

### 2. Verify Installation

```python
from transformers import BlipProcessor, BlipForImageTextRetrieval
print("BLIP installation successful!")
```

## Usage

### Command Line Interface

#### Basic BLIP Usage
```bash
# Use BLIP for text-to-image retrieval
python text_to_image_retriever.py \
    --input queries.csv \
    --database ./images \
    --output results.csv \
    --model_type blip
```

#### Using BLIP Presets
```bash
# Use BLIP preset configuration
python text_to_image_retriever.py \
    --input queries.csv \
    --database ./images \
    --output results.csv \
    --preset blip

# Use larger BLIP model for better performance
python text_to_image_retriever.py \
    --input queries.csv \
    --database ./images \
    --output results.csv \
    --preset blip_large
```

#### Custom BLIP Model
```bash
# Specify custom BLIP model
python text_to_image_retriever.py \
    --input queries.csv \
    --database ./images \
    --output results.csv \
    --model_type blip \
    --config custom_blip_config.json
```

### Python API

```python
from clip_baseline.config import CLIPConfig
from clip_baseline.text_to_image_retriever import TextToImageRetriever

# Create BLIP configuration
config = CLIPConfig(
    model_type="blip",
    blip_model_name="Salesforce/blip-image-captioning-base",
    similarity_metric="cosine",
    normalize_features=True,
    default_top_k=10
)

# Initialize retriever with BLIP
retriever = TextToImageRetriever(config=config)

# Load database
retriever.load_database("./images")

# Process queries with long text descriptions
results = retriever.process_csv_queries("queries.csv", "results.csv")
```

## Configuration Options

### BLIP-Specific Settings

```python
config = CLIPConfig(
    model_type="blip",                    # Use BLIP instead of CLIP
    blip_model_name="Salesforce/blip-image-captioning-base",  # BLIP model variant
    similarity_metric="cosine",           # Similarity metric
    normalize_features=True,              # Normalize feature vectors
    default_top_k=10,                     # Number of results to return
    min_similarity_threshold=0.1,         # Minimum similarity threshold
    verbose=True                          # Enable verbose output
)
```

### Available BLIP Models

1. **blip-image-captioning-base**: Fast, good for most use cases
2. **blip-image-captioning-large**: Better performance, more memory intensive
3. **blip2-opt-2.7b**: Even larger model with better performance (requires more GPU memory)

### Preset Configurations

- `blip`: Optimized for BLIP with base model
- `blip_large`: Uses larger BLIP model for better performance

## Performance Considerations

### Memory Usage
- BLIP models are more memory-intensive than CLIP
- Recommended GPU memory: 8GB+ for base model, 16GB+ for large model
- Batch sizes are automatically adjusted based on available memory

### Speed
- BLIP is generally slower than CLIP due to more complex architecture
- Batch processing is optimized for GPU acceleration
- Consider using smaller batch sizes if memory is limited

### Text Length Limits
- CLIP: ~300 characters (77 tokens)
- BLIP: ~2000 characters (512 tokens)
- Automatic truncation with word boundary preservation

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Reduce batch size
   python text_to_image_retriever.py --model_type blip --batch_size 2
   ```

2. **Model Download Issues**
   ```bash
   # Clear Hugging Face cache and retry
   rm -rf ~/.cache/huggingface/
   ```

3. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python text_to_image_retriever.py --model_type blip --device cpu
   ```

### Performance Tips

1. **Use GPU**: BLIP benefits significantly from GPU acceleration
2. **Batch Processing**: Larger batches improve throughput
3. **Model Selection**: Choose appropriate model size for your hardware
4. **Memory Management**: Monitor GPU memory usage during processing

## Comparison: CLIP vs BLIP

| Feature | CLIP | BLIP |
|---------|------|------|
| Text Length Limit | ~300 chars | ~2000 chars |
| Speed | Faster | Slower |
| Memory Usage | Lower | Higher |
| Accuracy | Good | Better for complex text |
| Model Size | Smaller | Larger |
| Best For | Short descriptions | Long, complex descriptions |

## Examples

### Long Text Description Example

```python
# This would fail with CLIP due to length
long_description = """
A beautiful sunset over the mountains with golden light reflecting off a calm lake. 
The scene includes tall pine trees silhouetted against the colorful sky, 
with mist rising from the water surface. The composition shows a peaceful 
wilderness setting perfect for photography, with dramatic lighting and 
natural beauty that would make for an excellent landscape photograph.
"""

# Works perfectly with BLIP
results = retriever.search(long_description, top_k=5)
```

### Batch Processing Example

```python
# Process multiple long descriptions
queries = [
    "A detailed description of a complex architectural building...",
    "A long narrative about a nature scene with specific details...",
    "An elaborate description of a cityscape at night..."
]

for query in queries:
    results = retriever.search(query, top_k=3)
    print(f"Found {len(results)} results for query")
```

## Migration from CLIP

To migrate from CLIP to BLIP:

1. **Update Configuration**: Change `model_type` from `"clip"` to `"blip"`
2. **Install Dependencies**: Add BLIP requirements
3. **Test Performance**: Verify performance on your specific use case
4. **Adjust Parameters**: Fine-tune batch sizes and thresholds

The API remains the same, making migration straightforward.

## Support

For issues specific to BLIP integration:

1. Check GPU memory usage
2. Verify model download
3. Test with shorter text first
4. Check batch size settings

The system will automatically fall back to CLIP if BLIP fails to load, ensuring robustness.
