# Dataset Analysis Scripts

This directory contains scripts for analyzing text descriptions in CSV datasets, particularly for determining optimal token lengths for BLIP models.

## Files

- `main.py` - Main script that runs analysis on multiple datasets
- `config.yaml` - Configuration file specifying datasets and analysis options
- `explore_descriptions.py` - Analyzes description lengths and creates visualizations
- `calculate_optimal_token_length.py` - Analyzes actual token lengths using BLIP's tokenizer
- `example_config.yaml` - Example configuration file

## Quick Start

1. **Create your configuration file:**
   ```bash
   cp example_config.yaml config.yaml
   ```

2. **Edit `config.yaml`** to specify your CSV files and output directories:
   ```yaml
   datasets:
     - name: "my_dataset"
       csv_path: "/path/to/your/dataset.csv"
       output_base_dir: "/path/to/output/directory"
   ```

3. **Run the analysis:**
   ```bash
   python main.py
   ```

## Configuration Options

### Datasets
Each dataset entry should have:
- `name`: A descriptive name for the dataset
- `csv_path`: Path to the CSV file containing descriptions
- `output_base_dir`: Base directory where results will be saved
- `image_prefix`: Prefix path to prepend to image paths from CSV

### Analysis Options
- `run_description_analysis`: Whether to run word-based description analysis (default: true)
- `run_token_analysis`: Whether to run token-based analysis (default: true)
- `run_image_examples`: Whether to generate image examples with descriptions (default: true)
- `whole_dataset_analysis`: When true, creates individual dataset analyses AND a summary analysis combining results from ALL datasets into consolidated CSV files and logs (default: false)
- `summary_only_mode`: When true, skips individual analyses and only creates summary analysis from existing individual results (default: false)
- `nan_detection_only`: When true, only detects and reports NaN descriptions in datasets, skipping all other analyses (default: false)

### Global Settings
- `overwrite_existing`: Overwrite existing output directories (default: false)

## Usage Examples

### List available datasets:
```bash
python main.py --list-datasets
```

### Run analysis for specific dataset:
```bash
python main.py --dataset "sf_xl_small_train"
```

### Use custom config file:
```bash
python main.py --config my_config.yaml
```

### Run only NaN detection:
```bash
# Set nan_detection_only: true in config.yaml
python main.py --config config.yaml
```

## CSV File Format

The CSV files should contain at least:
- A column with "description" in the name (case-insensitive)
- A column with "image" in the name (case-insensitive)

Example:
```csv
image_path,description
queries_night/@0543030.55@4180984.98@10@S@37.775128@-122.51158@31760538811@@@@@@20161226@@.jpg,"Stepped concrete sidewalk, STOP painted on asphalt road, pedestrian crossing with white zebra stripes, red octagonal stop sign on a pole, multi-story buildings, multiple street light poles receding into the distance."
queries_night/@0543102.49@4180346.93@10@S@37.76941@-122.510462@28026809684@@@@@@20160730@@.jpg,"A large, rectangular outdoor business sign with a thick, dark wooden frame and an overhead linear light fixture; the sign's light beige face features The Beach Chalet in large, stylized"
```

The `image_prefix` in the config will be prepended to the `image_path` from the CSV to create the full path to the image file.

## Output

For each dataset, the analysis creates:

### Description Analysis (`description_analysis/` subdirectory):
- `analysis_log.txt` - Detailed log of the analysis
- `description_statistics.csv` - Summary statistics
- `description_length_analysis.png` - Visualizations (histogram, box plot, etc.)
- `min_description.txt` - Shortest description found
- `max_description.txt` - Longest description found

### Token Analysis (`token_analysis/` subdirectory):
- `token_analysis_log.txt` - Detailed analysis with recommendations
- `token_length_distribution.csv` - Token length statistics
- `truncated_at_*.csv` - Samples that would be truncated at common max_lengths

### Image Examples (`image_examples/` subdirectory):
- `example_01.png`, `example_02.png`, `example_03.png` - Individual images with descriptions
- `examples_details.txt` - Detailed information about the selected examples
- `selected_examples.csv` - CSV file with the selected examples and metadata

### Summary Analysis (`summary_analysis/` subdirectory):
When `whole_dataset_analysis: true` is set, creates BOTH individual dataset analyses AND a consolidated summary:
- Individual dataset directories (same as regular mode)
- `summary_analysis/` directory containing:
  - `description_summary.csv` - Combined description statistics from all datasets
  - `token_summary.csv` - Combined token statistics from all datasets  
  - `combined_analysis_log.txt` - Concatenated logs from all individual dataset analyses
  - `overall_statistics.txt` - Overall statistics with weighted averages across all datasets
  - `summary_analysis_plots.png` - 4-panel summary visualization comparing all datasets
  - `description_distribution_comparison.png` - Violin plot comparing description length distributions

### NaN Detection (`{dataset_name}_nan_detection/` subdirectory):
When `nan_detection_only: true` is set, creates NaN detection reports for each dataset:
- `nan_descriptions.csv` - CSV file containing all rows with NaN descriptions
- `nan_detection_summary.txt` - Summary report with counts and percentages

## Recommendations

The token analysis provides recommendations for optimal `max_length` values:
- **Very Aggressive (95% coverage)**: Fastest training, some data loss
- **Recommended (99% coverage)**: Good balance of speed and coverage
- **Conservative (99.9% coverage)**: Minimal data loss, slower training
- **Full Coverage (100%)**: No data loss, slowest training

## Dependencies

Make sure you have the required packages installed:
```bash
pip install pandas matplotlib numpy pyyaml transformers
```

## Troubleshooting

- **CSV file not found**: Check the paths in your config file
- **Column not found**: Ensure your CSV has columns with "description" and "image" in the name
- **Memory issues**: For very large datasets, consider processing in smaller batches
- **Tokenizer errors**: Make sure you have internet access for downloading the BERT tokenizer
