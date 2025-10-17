import pandas as pd
from transformers import BertTokenizer
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_token_lengths(csv_path, output_dir=None):
    """
    Analyze actual token lengths for BLIP's tokenizer to determine optimal max_length.
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path('./output')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Read CSV with better error handling
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
    
    # Clean column names
    df.columns = df.columns.str.strip().str.strip("'\"")
    
    # Validate CSV format: must have exactly 2 columns
    if len(df.columns) != 2:
        error_msg = f"Invalid CSV format: Expected exactly 2 columns, found {len(df.columns)}. Columns: {list(df.columns)}"
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # Find description column
    description_col = None
    for col in df.columns:
        if 'description' in col.lower():
            description_col = col
            break
    
    if description_col is None:
        raise ValueError(f"Could not find description column! Available: {df.columns.tolist()}")
    
    print(f"Found description column: '{description_col}'")
    print(f"Total samples: {len(df)}\n")
    
    # Initialize BLIP tokenizer (same as in the model)
    print("Loading BLIP tokenizer (bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize all descriptions
    print("Tokenizing all descriptions (this may take a minute)...")
    token_lengths = []
    
    for i, desc in enumerate(df[description_col]):
        if i % 10000 == 0:
            print(f"  Processed {i}/{len(df)} descriptions...")
        tokens = tokenizer.encode(str(desc), add_special_tokens=True, truncation=False)
        token_lengths.append(len(tokens))
    
    token_lengths = np.array(token_lengths)
    df['token_length'] = token_lengths
    
    # Calculate statistics
    mean_tokens = token_lengths.mean()
    median_tokens = np.median(token_lengths)
    max_tokens = token_lengths.max()
    min_tokens = token_lengths.min()
    std_tokens = token_lengths.std()
    
    # Calculate percentiles
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    percentile_values = {p: np.percentile(token_lengths, p) for p in percentiles}
    
    # Print statistics
    print("\n" + "="*70)
    print("TOKEN LENGTH STATISTICS (Actual BLIP Tokenizer)")
    print("="*70)
    print(f"Mean:           {mean_tokens:.2f} tokens")
    print(f"Median:         {median_tokens:.0f} tokens")
    print(f"Min:            {min_tokens} tokens")
    print(f"Max:            {max_tokens} tokens")
    print(f"Std Dev:        {std_tokens:.2f} tokens")
    print("="*70)
    print("\nPERCENTILE COVERAGE:")
    print("-"*70)
    for p, v in percentile_values.items():
        samples_covered = len(df)
        samples_truncated = np.sum(token_lengths > v)
        pct_truncated = (samples_truncated / len(df)) * 100
        print(f"  {p:5.1f}th percentile: {v:6.0f} tokens  "
              f"({100-pct_truncated:.2f}% covered, {samples_truncated} samples truncated)")
    print("="*70)
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR max_length")
    print("="*70)
    
    recommendations = [
        ("Very Aggressive (95% coverage)", int(np.ceil(percentile_values[95]))),
        ("Recommended (99% coverage)", int(np.ceil(percentile_values[99]))),
        ("Conservative (99.9% coverage)", int(np.ceil(percentile_values[99.9]))),
        ("Full Coverage (100%)", int(np.ceil(max_tokens))),
    ]
    
    for desc, length in recommendations:
        # Round up to nearest power of 2 or common value
        rounded = 2 ** int(np.ceil(np.log2(length)))
        if rounded > length and rounded // 2 >= length:
            rounded = rounded // 2
        # Common values: 32, 64, 77, 128, 256, 512
        common_values = [32, 64, 77, 128, 192, 256, 384, 512]
        for cv in common_values:
            if cv >= length:
                rounded = cv
                break
        
        samples_lost = np.sum(token_lengths > rounded)
        pct_lost = (samples_lost / len(df)) * 100
        
        speedup = 512 / rounded
        memory_saving = 1 - (rounded**2 / 512**2)
        
        print(f"\n{desc}:")
        print(f"  max_length = {rounded}")
        print(f"  Samples truncated: {samples_lost} ({pct_lost:.3f}%)")
        print(f"  Speed improvement: {speedup:.2f}x faster")
        print(f"  Memory savings: {memory_saving*100:.1f}% less attention memory")
    
    print("="*70)
    
    # Compare with current setting
    current_max = 512
    wasted_tokens = current_max - token_lengths
    avg_waste_pct = (wasted_tokens.mean() / current_max) * 100
    
    print(f"\nCURRENT SETTING (max_length=512):")
    print(f"  Average padding: {wasted_tokens.mean():.1f} tokens ({avg_waste_pct:.1f}% waste)")
    print(f"  Median padding:  {np.median(wasted_tokens):.0f} tokens")
    print("="*70)
    
    # Save detailed analysis
    log_file = output_dir / 'token_analysis_log.txt'
    with open(log_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TOKEN LENGTH ANALYSIS\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {Path(csv_path).absolute()}\n")
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Tokenizer: bert-base-uncased (BLIP's tokenizer)\n\n")
        
        f.write("STATISTICS:\n")
        f.write(f"  Mean:   {mean_tokens:.2f} tokens\n")
        f.write(f"  Median: {median_tokens:.0f} tokens\n")
        f.write(f"  Min:    {min_tokens} tokens\n")
        f.write(f"  Max:    {max_tokens} tokens\n")
        f.write(f"  Std:    {std_tokens:.2f} tokens\n\n")
        
        f.write("PERCENTILES:\n")
        for p, v in percentile_values.items():
            f.write(f"  {p}th: {v:.0f} tokens\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATIONS:\n")
        for desc, length in recommendations:
            f.write(f"\n{desc}: {length} tokens\n")
    
    # Save token length distribution
    df[['token_length']].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_csv(
        output_dir / 'token_length_distribution.csv'
    )
    
    
    print(f"\n✓ Analysis complete! Logs saved to: {output_dir.absolute()}")
    
    return df

if __name__ == "__main__":
    csv_path = "/mnt/d/data/sf_xl/small/sf_xl_small_train_descriptions.csv"
    output_dir = "/mnt/d/dan/git_projects/text2vpr_updated/BLIP/caption_summarization/stats/token_analysis"
    df = analyze_token_lengths(csv_path, output_dir)

