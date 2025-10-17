import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for WSL/headless environments
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_descriptions(csv_path, output_dir=None):
    """
    Analyze text descriptions from a CSV file.
    
    Args:
        csv_path: Path to CSV file with 'Image_path' and 'Description' columns
        output_dir: Base directory to save outputs. A timestamped subdirectory will be created.
                   If None, saves to './output'
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path('./output')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}")
    # Read the CSV file with better error handling
    print(f"Reading CSV from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, quoting=1, escapechar='\\')  # quoting=1 means QUOTE_ALL
    except pd.errors.ParserError as e:
        print(f"⚠️  CSV parsing failed with default settings. Trying alternative parsing...")
        print(f"   Error: {e}")
        try:
            # Try with different quoting options
            df = pd.read_csv(csv_path, quoting=3, escapechar='\\')  # quoting=3 means QUOTE_NONE
        except pd.errors.ParserError as e2:
            print(f"⚠️  Alternative parsing also failed. Trying with error handling...")
            print(f"   Error: {e2}")
            # Try with error handling - skip bad lines
            df = pd.read_csv(csv_path, on_bad_lines='skip', quoting=1, escapechar='\\')
            print(f"   ⚠️  Some lines were skipped due to parsing errors")
    
    # Check columns
    print(f"\nCSV columns found: {df.columns.tolist()}")
    print(f"Column names (with repr to show hidden chars): {[repr(col) for col in df.columns]}")
    print(f"Total number of samples: {len(df)}")
    
    # Strip whitespace and quotes from column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.strip("'\"")  # Remove quotes if present
    
    print(f"Cleaned column names: {df.columns.tolist()}")
    
    # Try to find the description column (case-insensitive)
    description_col = None
    image_col = None
    
    for col in df.columns:
        if 'description' in col.lower():
            description_col = col
            print(f"\nFound description column: '{description_col}'")
        if 'image' in col.lower():
            image_col = col
            print(f"Found image column: '{image_col}'")
    
    if description_col is None:
        raise ValueError(f"Could not find description column! Available columns: {df.columns.tolist()}")
    if image_col is None:
        raise ValueError(f"Could not find image column! Available columns: {df.columns.tolist()}")
    
    # Calculate description lengths in words
    df['description_length'] = df[description_col].astype(str).apply(lambda x: len(x.split()))
    
    # Create log file
    log_file = output_dir / 'analysis_log.txt'
    with open(log_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DATASET ANALYSIS LOG\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset CSV: {Path(csv_path).absolute()}\n")
        f.write(f"Output Directory: {output_dir.absolute()}\n")
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Columns: {df.columns.tolist()}\n")
        f.write(f"Description Column: {description_col}\n")
        f.write(f"Image Column: {image_col}\n")
        f.write("="*60 + "\n\n")
    
    print(f"Created log file: {log_file}")
    
    # Calculate statistics
    mean_length = df['description_length'].mean()
    median_length = df['description_length'].median()
    min_length = df['description_length'].min()
    max_length = df['description_length'].max()
    std_length = df['description_length'].std()
    
    print("\n" + "="*60)
    print("DESCRIPTION LENGTH STATISTICS")
    print("="*60)
    print(f"Mean length:   {mean_length:.2f} words")
    print(f"Median length: {median_length:.2f} words")
    print(f"Min length:    {min_length} words")
    print(f"Max length:    {max_length} words")
    print(f"Std deviation: {std_length:.2f} words")
    print("="*60)
    
    # Also write statistics to log file
    with open(log_file, 'a') as f:
        f.write("DESCRIPTION LENGTH STATISTICS\n")
        f.write("="*60 + "\n")
        f.write(f"Mean length:   {mean_length:.2f} words\n")
        f.write(f"Median length: {median_length:.2f} words\n")
        f.write(f"Min length:    {min_length} words\n")
        f.write(f"Max length:    {max_length} words\n")
        f.write(f"Std deviation: {std_length:.2f} words\n")
        f.write("="*60 + "\n\n")
    
    # Find min and max descriptions
    min_idx = df['description_length'].idxmin()
    max_idx = df['description_length'].idxmax()
    
    min_description = df[description_col].loc[min_idx]
    max_description = df[description_col].loc[max_idx]
    min_image_path = df[image_col].loc[min_idx]
    max_image_path = df[image_col].loc[max_idx]
    
    print("\n" + "="*60)
    print("SHORTEST DESCRIPTION")
    print("="*60)
    print(f"Image: {min_image_path}")
    print(f"Length: {min_length} words")
    print(f"Text: {min_description}")
    print("="*60)
    
    print("\n" + "="*60)
    print("LONGEST DESCRIPTION")
    print("="*60)
    print(f"Image: {max_image_path}")
    print(f"Length: {max_length} words")
    print(f"Text: {max_description}")
    print("="*60)
    
    # Save min and max descriptions to files
    
    with open(output_dir / 'min_description.txt', 'w') as f:
        f.write(f"Image: {min_image_path}\n")
        f.write(f"Length: {min_length} words\n\n")
        f.write(f"{min_description}\n")
    
    with open(output_dir / 'max_description.txt', 'w') as f:
        f.write(f"Image: {max_image_path}\n")
        f.write(f"Length: {max_length} words\n\n")
        f.write(f"{max_description}\n")
    
    # Also add to log file
    with open(log_file, 'a') as f:
        f.write("\nSHORTEST DESCRIPTION\n")
        f.write("="*60 + "\n")
        f.write(f"Image: {min_image_path}\n")
        f.write(f"Length: {min_length} words\n")
        f.write(f"Text: {min_description}\n")
        f.write("="*60 + "\n\n")
        
        f.write("LONGEST DESCRIPTION\n")
        f.write("="*60 + "\n")
        f.write(f"Image: {max_image_path}\n")
        f.write(f"Length: {max_length} words\n")
        f.write(f"Text: {max_description}\n")
        f.write("="*60 + "\n")
    
    print(f"\nSaved min description to: {output_dir / 'min_description.txt'}")
    print(f"Saved max description to: {output_dir / 'max_description.txt'}")
    
    # Create visualization - only distribution histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Distribution of Description Lengths', fontsize=16, fontweight='bold')
    
    # Histogram of description lengths
    ax.hist(df['description_length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
    ax.axvline(median_length, color='green', linestyle='--', linewidth=2, label=f'Median: {median_length:.1f}')
    ax.set_xlabel('Description Length (words)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Description Lengths', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / 'description_length_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    plt.close()  # Close the figure to free memory
    
    # Save statistics to CSV
    stats_df = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Total Samples'],
        'Value': [mean_length, median_length, min_length, max_length, std_length, len(df)]
    })
    stats_path = output_dir / 'description_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to: {stats_path}")
    
    # Final summary in log file
    with open(log_file, 'a') as f:
        f.write("\n" + "="*60 + "\n")
        f.write("OUTPUT FILES GENERATED\n")
        f.write("="*60 + "\n")
        f.write(f"1. {output_dir / 'analysis_log.txt'} - This log file\n")
        f.write(f"2. {output_dir / 'description_statistics.csv'} - Statistics summary\n")
        f.write(f"3. {output_dir / 'description_length_analysis.png'} - Visualizations\n")
        f.write(f"4. {output_dir / 'min_description.txt'} - Shortest description\n")
        f.write(f"5. {output_dir / 'max_description.txt'} - Longest description\n")
        f.write("="*60 + "\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n{'='*60}")
    print("Analysis complete! All outputs saved to:")
    print(f"{output_dir.absolute()}")
    print(f"{'='*60}")
    
    return df

if __name__ == "__main__":
    csv_path = "/mnt/d/data/sf_xl/small/sf_xl_small_train_descriptions.csv"
    output_dir = "/mnt/d/dan/git_projects/text2vpr_updated/BLIP/caption_summarization/stats/dates"
    df = analyze_descriptions(csv_path, output_dir)

