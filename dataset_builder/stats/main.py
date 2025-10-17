#!/usr/bin/env python3
"""
Main script to run dataset analysis on multiple CSV files.
Reads configuration from config.yaml and runs both description and token analysis.
"""

import yaml
import sys
from pathlib import Path
from datetime import datetime
import argparse
import traceback
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import pandas as pd

# Import the analysis functions
from explore_descriptions import analyze_descriptions
from calculate_optimal_token_length import analyze_token_lengths
from generate_image_examples import generate_image_examples


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        sys.exit(1)


def validate_config(config):
    """Validate the configuration file."""
    required_keys = ['datasets', 'analysis_options']
    
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required key '{key}' in config file")
            sys.exit(1)
    
    # Validate datasets
    if not isinstance(config['datasets'], list) or len(config['datasets']) == 0:
        print("Error: 'datasets' must be a non-empty list")
        sys.exit(1)
    
    for i, dataset in enumerate(config['datasets']):
        required_dataset_keys = ['name', 'csv_path', 'output_base_dir', 'image_prefix']
        for key in required_dataset_keys:
            if key not in dataset:
                print(f"Error: Dataset {i} missing required key '{key}'")
                sys.exit(1)
        
        # Check if CSV file exists
        csv_path = Path(dataset['csv_path'])
        if not csv_path.exists():
            print(f"Warning: CSV file does not exist: {csv_path}")
            print("  This dataset will be skipped.")
    
    # Validate analysis options
    analysis_opts = config['analysis_options']
    if not isinstance(analysis_opts.get('run_description_analysis', True), bool):
        print("Error: 'run_description_analysis' must be a boolean")
        sys.exit(1)
    if not isinstance(analysis_opts.get('run_token_analysis', True), bool):
        print("Error: 'run_token_analysis' must be a boolean")
        sys.exit(1)
    if not isinstance(analysis_opts.get('run_image_examples', True), bool):
        print("Error: 'run_image_examples' must be a boolean")
        sys.exit(1)
    if not isinstance(analysis_opts.get('whole_dataset_analysis', False), bool):
        print("Error: 'whole_dataset_analysis' must be a boolean")
        sys.exit(1)
    if not isinstance(analysis_opts.get('summary_only_mode', False), bool):
        print("Error: 'summary_only_mode' must be a boolean")
        sys.exit(1)
    if not isinstance(analysis_opts.get('nan_detection_only', False), bool):
        print("Error: 'nan_detection_only' must be a boolean")
        sys.exit(1)


def check_individual_analyses_exist(datasets_to_run, analysis_options):
    """Check if all required individual analyses exist for the datasets."""
    missing_analyses = []
    
    for dataset_config in datasets_to_run:
        name = dataset_config['name']
        output_base_dir = Path(dataset_config['output_base_dir'])
        dataset_output_dir = output_base_dir / name
        
        if not dataset_output_dir.exists():
            missing_analyses.append(f"{name}: No output directory")
            continue
        
        # Check for description analysis
        if analysis_options.get('run_description_analysis', True):
            desc_dir = dataset_output_dir / "description_analysis"
            if not desc_dir.exists():
                missing_analyses.append(f"{name}: Missing description_analysis directory")
            else:
                stats_file = desc_dir / 'description_statistics.csv'
                if not stats_file.exists():
                    missing_analyses.append(f"{name}: Missing description_statistics.csv")
        
        # Check for token analysis
        if analysis_options.get('run_token_analysis', True):
            token_dir = dataset_output_dir / "token_analysis"
            if not token_dir.exists():
                missing_analyses.append(f"{name}: Missing token_analysis directory")
            else:
                dist_file = token_dir / 'token_length_distribution.csv'
                if not dist_file.exists():
                    missing_analyses.append(f"{name}: Missing token_length_distribution.csv")
    
    return len(missing_analyses) == 0, missing_analyses


def detect_nan_descriptions(datasets_to_run, analysis_options, global_settings):
    """Detect and report NaN descriptions in datasets."""
    print(f"\n{'='*80}")
    print("NaN DESCRIPTION DETECTION")
    print(f"{'='*80}")
    
    total_nan_count = 0
    total_samples = 0
    
    for dataset_config in datasets_to_run:
        name = dataset_config['name']
        csv_path = dataset_config['csv_path']
        
        print(f"\n📊 Analyzing {name}...")
        print(f"   CSV: {csv_path}")
        
        # Check if CSV exists
        if not Path(csv_path).exists():
            print(f"   ❌ CSV file not found, skipping...")
            continue
        
        try:
            # Read CSV with better error handling
            try:
                df = pd.read_csv(csv_path, quoting=1, escapechar='\\')
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(csv_path, quoting=3, escapechar='\\')
                except pd.errors.ParserError:
                    df = pd.read_csv(csv_path, on_bad_lines='skip', quoting=1, escapechar='\\')
            
            # Clean column names
            df.columns = df.columns.str.strip().str.strip("'\"")
            
            # Validate CSV format: must have exactly 2 columns
            if len(df.columns) != 2:
                print(f"   ❌ Invalid CSV format: Expected exactly 2 columns, found {len(df.columns)}")
                print(f"   📋 Found columns: {list(df.columns)}")
                
                # Create error log
                output_base_dir = Path(dataset_config['output_base_dir'])
                error_output_dir = output_base_dir / f"{name}_csv_format_error"
                error_output_dir.mkdir(parents=True, exist_ok=True)
                
                error_log_path = error_output_dir / 'csv_format_error.txt'
                with open(error_log_path, 'w') as f:
                    f.write("="*60 + "\n")
                    f.write(f"CSV FORMAT ERROR - {name}\n")
                    f.write("="*60 + "\n")
                    f.write(f"Dataset: {name}\n")
                    f.write(f"CSV file: {csv_path}\n")
                    f.write(f"Error date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Expected: Exactly 2 columns\n")
                    f.write(f"Found: {len(df.columns)} columns\n")
                    f.write(f"Columns found: {list(df.columns)}\n")
                    f.write("="*60 + "\n")
                    f.write("REQUIRED FORMAT:\n")
                    f.write("Column 1: Image_path (or any column with 'image' in name)\n")
                    f.write("Column 2: Description (or any column with 'description' in name)\n")
                    f.write("="*60 + "\n")
                    f.write("EXAMPLE:\n")
                    f.write("Image_path,Description\n")
                    f.write("/path/to/image1.jpg,\"A beautiful sunset\"\n")
                    f.write("/path/to/image2.jpg,\"A cat sitting\"\n")
                
                print(f"   📄 Error log saved to: {error_log_path}")
                continue
            
            # Find description and image columns
            description_col = None
            image_col = None
            
            for col in df.columns:
                if 'description' in col.lower():
                    description_col = col
                if 'image' in col.lower():
                    image_col = col
            
            if description_col is None or image_col is None:
                print(f"   ❌ Invalid column names: Expected columns with 'image' and 'description' in names")
                print(f"   📋 Found columns: {list(df.columns)}")
                
                # Create error log
                output_base_dir = Path(dataset_config['output_base_dir'])
                error_output_dir = output_base_dir / f"{name}_column_name_error"
                error_output_dir.mkdir(parents=True, exist_ok=True)
                
                error_log_path = error_output_dir / 'column_name_error.txt'
                with open(error_log_path, 'w') as f:
                    f.write("="*60 + "\n")
                    f.write(f"COLUMN NAME ERROR - {name}\n")
                    f.write("="*60 + "\n")
                    f.write(f"Dataset: {name}\n")
                    f.write(f"CSV file: {csv_path}\n")
                    f.write(f"Error date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Found columns: {list(df.columns)}\n")
                    f.write("="*60 + "\n")
                    f.write("REQUIRED COLUMN NAMES:\n")
                    f.write("• One column must contain 'image' in the name (case-insensitive)\n")
                    f.write("• One column must contain 'description' in the name (case-insensitive)\n")
                    f.write("="*60 + "\n")
                    f.write("VALID EXAMPLES:\n")
                    f.write("• Image_path, Description\n")
                    f.write("• image_file, text_description\n")
                    f.write("• IMAGE_PATH, DESCRIPTION\n")
                    f.write("• image, description\n")
                
                print(f"   📄 Error log saved to: {error_log_path}")
                continue
            
            print(f"   Found {len(df)} samples")
            
            # Detect NaN descriptions
            nan_mask = df[description_col].isna()
            nan_count = nan_mask.sum()
            
            print(f"   📋 NaN descriptions found: {nan_count}")
            print(f"   📊 Percentage: {(nan_count/len(df)*100):.2f}%")
            
            total_nan_count += nan_count
            total_samples += len(df)
            
            # Save NaN samples to CSV if any found
            if nan_count > 0:
                nan_samples = df[nan_mask].copy()
                
                # Create output directory
                output_base_dir = Path(dataset_config['output_base_dir'])
                nan_output_dir = output_base_dir / f"{name}_nan_detection"
                nan_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save NaN samples
                nan_csv_path = nan_output_dir / 'nan_descriptions.csv'
                nan_samples.to_csv(nan_csv_path, index=False)
                print(f"   💾 Saved {nan_count} NaN samples to: {nan_csv_path}")
                
                # Create summary report
                summary_path = nan_output_dir / 'nan_detection_summary.txt'
                with open(summary_path, 'w') as f:
                    f.write("="*60 + "\n")
                    f.write(f"NaN DETECTION SUMMARY - {name}\n")
                    f.write("="*60 + "\n")
                    f.write(f"Dataset: {name}\n")
                    f.write(f"CSV file: {csv_path}\n")
                    f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total samples: {len(df):,}\n")
                    f.write(f"NaN descriptions: {nan_count:,}\n")
                    f.write(f"Percentage: {(nan_count/len(df)*100):.2f}%\n")
                    f.write(f"Valid descriptions: {len(df) - nan_count:,}\n")
                    f.write("="*60 + "\n")
                    f.write(f"NaN samples saved to: {nan_csv_path}\n")
                
                print(f"   📄 Summary report saved to: {summary_path}")
            else:
                print(f"   ✅ No NaN descriptions found!")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {name}: {e}")
            continue
    
    # Create overall summary
    print(f"\n{'='*80}")
    print("OVERALL NaN DETECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets analyzed: {len(datasets_to_run)}")
    print(f"Total samples across all datasets: {total_samples:,}")
    print(f"Total NaN descriptions found: {total_nan_count:,}")
    if total_samples > 0:
        print(f"Overall NaN percentage: {(total_nan_count/total_samples*100):.2f}%")
    print(f"{'='*80}")
    
    return True


def create_summary_analysis(datasets_to_run, analysis_options, global_settings):
    """Create a summary analysis combining results from all datasets."""
    if not datasets_to_run:
        print("❌ No datasets to summarize")
        return False
    
    # Create summary output directory
    summary_output_dir = Path(datasets_to_run[0]['output_base_dir']) / "summary_analysis"
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CREATING SUMMARY ANALYSIS")
    print(f"{'='*80}")
    print(f"Summary directory: {summary_output_dir}")
    print(f"Combining {len(datasets_to_run)} datasets...")
    
    all_description_stats = []
    all_token_stats = []
    all_logs = []
    
    for dataset_config in datasets_to_run:
        name = dataset_config['name']
        csv_path = dataset_config['csv_path']
        
        print(f"\n📊 Processing {name}...")
        
        # Check if CSV exists
        if not Path(csv_path).exists():
            print(f"  ⚠️  Skipping {name}: CSV file not found")
            continue
        
        try:
            # Run description analysis
            if analysis_options.get('run_description_analysis', True):
                print(f"  📊 Running description analysis...")
                temp_dir = summary_output_dir / "temp" / name
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                from explore_descriptions import analyze_descriptions
                df_desc = analyze_descriptions(csv_path, str(temp_dir))
                
                # Collect statistics
                stats = {
                    'dataset': name,
                    'total_samples': len(df_desc),
                    'mean_length': df_desc['description_length'].mean(),
                    'median_length': df_desc['description_length'].median(),
                    'min_length': df_desc['description_length'].min(),
                    'max_length': df_desc['description_length'].max(),
                    'std_length': df_desc['description_length'].std()
                }
                all_description_stats.append(stats)
                
                # Read the log file
                log_file = temp_dir / 'analysis_log.txt'
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    all_logs.append(f"\n{'='*60}\nDATASET: {name}\n{'='*60}\n{log_content}")
            
            # Run token analysis
            if analysis_options.get('run_token_analysis', True):
                print(f"  🔤 Running token analysis...")
                temp_dir = summary_output_dir / "temp" / name
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                from calculate_optimal_token_length import analyze_token_lengths
                df_tokens = analyze_token_lengths(csv_path, str(temp_dir))
                
                # Collect token statistics
                token_stats = {
                    'dataset': name,
                    'total_samples': len(df_tokens),
                    'mean_tokens': df_tokens['token_length'].mean(),
                    'median_tokens': df_tokens['token_length'].median(),
                    'min_tokens': df_tokens['token_length'].min(),
                    'max_tokens': df_tokens['token_length'].max(),
                    'std_tokens': df_tokens['token_length'].std()
                }
                all_token_stats.append(token_stats)
            
            print(f"  ✅ {name} processed successfully")
            
        except Exception as e:
            print(f"  ❌ Error processing {name}: {e}")
            continue
    
    # Create consolidated summaries
    print(f"\n📝 Creating consolidated summaries...")
    
    # Description summary
    if all_description_stats:
        desc_summary_df = pd.DataFrame(all_description_stats)
        desc_summary_file = summary_output_dir / 'description_summary.csv'
        desc_summary_df.to_csv(desc_summary_file, index=False)
        print(f"  ✅ Description summary saved to: {desc_summary_file}")
    
    # Token summary
    if all_token_stats:
        token_summary_df = pd.DataFrame(all_token_stats)
        token_summary_file = summary_output_dir / 'token_summary.csv'
        token_summary_df.to_csv(token_summary_file, index=False)
        print(f"  ✅ Token summary saved to: {token_summary_file}")
    
    # Combined log file
    if all_logs:
        combined_log_file = summary_output_dir / 'combined_analysis_log.txt'
        with open(combined_log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMBINED DATASET ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total datasets analyzed: {len(datasets_to_run)}\n")
            f.write("="*80 + "\n")
            f.write("\n".join(all_logs))
        print(f"  ✅ Combined log saved to: {combined_log_file}")
    
    # Create overall statistics
    if all_description_stats and all_token_stats:
        overall_stats_file = summary_output_dir / 'overall_statistics.txt'
        with open(overall_stats_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OVERALL STATISTICS ACROSS ALL DATASETS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total datasets: {len(all_description_stats)}\n\n")
            
            # Description statistics
            f.write("DESCRIPTION LENGTH STATISTICS:\n")
            f.write("-" * 40 + "\n")
            total_samples = sum(s['total_samples'] for s in all_description_stats)
            weighted_mean_desc = sum(s['mean_length'] * s['total_samples'] for s in all_description_stats) / total_samples
            f.write(f"Total samples: {total_samples:,}\n")
            f.write(f"Weighted mean length: {weighted_mean_desc:.2f} words\n")
            f.write(f"Min length (across datasets): {min(s['min_length'] for s in all_description_stats)} words\n")
            f.write(f"Max length (across datasets): {max(s['max_length'] for s in all_description_stats)} words\n\n")
            
            # Token statistics
            f.write("TOKEN LENGTH STATISTICS:\n")
            f.write("-" * 40 + "\n")
            weighted_mean_tokens = sum(s['mean_tokens'] * s['total_samples'] for s in all_token_stats) / total_samples
            f.write(f"Weighted mean tokens: {weighted_mean_tokens:.2f} tokens\n")
            f.write(f"Min tokens (across datasets): {min(s['min_tokens'] for s in all_token_stats)} tokens\n")
            f.write(f"Max tokens (across datasets): {max(s['max_tokens'] for s in all_token_stats)} tokens\n")
        
        print(f"  ✅ Overall statistics saved to: {overall_stats_file}")
    
    # Create summary visualizations
    if all_description_stats and all_token_stats:
        print(f"\n📊 Creating summary visualizations...")
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Summary Analysis Across All Datasets', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        datasets = [s['dataset'] for s in all_description_stats]
        desc_means = [s['mean_length'] for s in all_description_stats]
        desc_medians = [s['median_length'] for s in all_description_stats]
        desc_stds = [s['std_length'] for s in all_description_stats]
        sample_counts = [s['total_samples'] for s in all_description_stats]
        
        token_means = [s['mean_tokens'] for s in all_token_stats]
        token_medians = [s['median_tokens'] for s in all_token_stats]
        token_stds = [s['std_tokens'] for s in all_token_stats]
        
        # 1. Description length comparison
        x_pos = np.arange(len(datasets))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, desc_means, width, label='Mean', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x_pos + width/2, desc_medians, width, label='Median', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Datasets')
        axes[0, 0].set_ylabel('Description Length (words)')
        axes[0, 0].set_title('Description Length Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Token length comparison
        axes[0, 1].bar(x_pos - width/2, token_means, width, label='Mean', alpha=0.8, color='lightgreen')
        axes[0, 1].bar(x_pos + width/2, token_medians, width, label='Median', alpha=0.8, color='orange')
        axes[0, 1].set_xlabel('Datasets')
        axes[0, 1].set_ylabel('Token Length (tokens)')
        axes[0, 1].set_title('Token Length Comparison')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Sample size comparison
        axes[1, 0].bar(x_pos, sample_counts, alpha=0.8, color='purple')
        axes[1, 0].set_xlabel('Datasets')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Dataset Size Comparison')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add sample count labels on bars
        for i, v in enumerate(sample_counts):
            axes[1, 0].text(i, v + max(sample_counts)*0.01, f'{v:,}', ha='center', va='bottom', fontsize=8)
        
        # 4. Description vs Token length scatter plot
        axes[1, 1].scatter(desc_means, token_means, s=100, alpha=0.7, c='red')
        
        # Add dataset labels to scatter points
        for i, dataset in enumerate(datasets):
            axes[1, 1].annotate(dataset.replace('_', '\n'), 
                               (desc_means[i], token_means[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, ha='left')
        
        axes[1, 1].set_xlabel('Mean Description Length (words)')
        axes[1, 1].set_ylabel('Mean Token Length (tokens)')
        axes[1, 1].set_title('Description vs Token Length Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation line
        z = np.polyfit(desc_means, token_means, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(desc_means, p(desc_means), "r--", alpha=0.8, linewidth=2)
        
        # Calculate and display correlation coefficient
        correlation = np.corrcoef(desc_means, token_means)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=axes[1, 1].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=10)
        
        plt.tight_layout()
        
        # Save the summary plot
        summary_plot_path = summary_output_dir / 'summary_analysis_plots.png'
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Summary plots saved to: {summary_plot_path}")
        
        # Create individual comparison plots
        # Description length distribution comparison
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create violin plot for description lengths
        desc_data = []
        labels = []
        for stats in all_description_stats:
            # For violin plot, we'll use the mean and std to create a normal distribution approximation
            # This is a simplified representation since we don't have the raw data
            mean = stats['mean_length']
            std = stats['std_length']
            # Generate synthetic data based on normal distribution
            synthetic_data = np.random.normal(mean, std, min(1000, stats['total_samples']))
            desc_data.append(synthetic_data)
            labels.append(stats['dataset'].replace('_', '\n'))
        
        parts = ax.violinplot(desc_data, positions=range(len(datasets)), showmeans=True, showmedians=True)
        
        # Customize violin plot colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Description Length (words)')
        ax.set_title('Description Length Distribution Comparison')
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save the violin plot
        violin_plot_path = summary_output_dir / 'description_distribution_comparison.png'
        plt.savefig(violin_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Description distribution plot saved to: {violin_plot_path}")
    
    # Clean up temp directories
    temp_dir = summary_output_dir / "temp"
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
    
    print(f"\n🎉 Summary analysis completed!")
    print(f"   Results saved to: {summary_output_dir}")
    
    return True


def create_summary_from_existing_results(datasets_to_run, analysis_options, global_settings):
    """Create summary analysis from existing individual dataset results."""
    if not datasets_to_run:
        print("❌ No datasets to summarize")
        return False
    
    # Create summary output directory
    summary_output_dir = Path(datasets_to_run[0]['output_base_dir']) / "summary_analysis"
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Summary directory: {summary_output_dir}")
    print(f"Combining results from {len(datasets_to_run)} datasets...")
    
    all_description_stats = []
    all_token_stats = []
    all_logs = []
    
    for dataset_config in datasets_to_run:
        name = dataset_config['name']
        output_base_dir = Path(dataset_config['output_base_dir'])
        dataset_output_dir = output_base_dir / name
        
        print(f"\n📊 Processing results from {name}...")
        
        # Check if individual results exist
        if not dataset_output_dir.exists():
            print(f"  ⚠️  Skipping {name}: No individual results found")
            continue
        
        try:
            # Read description analysis results
            if analysis_options.get('run_description_analysis', True):
                desc_dir = dataset_output_dir / "description_analysis"
                if desc_dir.exists():
                    # Read the statistics CSV
                    stats_file = desc_dir / 'description_statistics.csv'
                    if stats_file.exists():
                        stats_df = pd.read_csv(stats_file)
                        # Convert to dictionary format
                        stats_dict = {}
                        for _, row in stats_df.iterrows():
                            stats_dict[row['Statistic']] = row['Value']
                        
                        desc_stats = {
                            'dataset': name,
                            'total_samples': int(stats_dict.get('Total Samples', 0)),
                            'mean_length': stats_dict.get('Mean', 0),
                            'median_length': stats_dict.get('Median', 0),
                            'min_length': stats_dict.get('Min', 0),
                            'max_length': stats_dict.get('Max', 0),
                            'std_length': stats_dict.get('Std Dev', 0)
                        }
                        all_description_stats.append(desc_stats)
                        
                        # Read the log file
                        log_file = desc_dir / 'analysis_log.txt'
                        if log_file.exists():
                            with open(log_file, 'r') as f:
                                log_content = f.read()
                            all_logs.append(f"\n{'='*60}\nDATASET: {name}\n{'='*60}\n{log_content}")
            
            # Read token analysis results
            if analysis_options.get('run_token_analysis', True):
                token_dir = dataset_output_dir / "token_analysis"
                if token_dir.exists():
                    # Read the token length distribution CSV
                    dist_file = token_dir / 'token_length_distribution.csv'
                    if dist_file.exists():
                        dist_df = pd.read_csv(dist_file)
                        # Get statistics from the describe() output
                        token_stats = {
                            'dataset': name,
                            'total_samples': int(dist_df.loc[dist_df['Unnamed: 0'] == 'count', 'token_length'].iloc[0]) if 'count' in dist_df['Unnamed: 0'].values else 0,
                            'mean_tokens': dist_df.loc[dist_df['Unnamed: 0'] == 'mean', 'token_length'].iloc[0] if 'mean' in dist_df['Unnamed: 0'].values else 0,
                            'median_tokens': dist_df.loc[dist_df['Unnamed: 0'] == '50%', 'token_length'].iloc[0] if '50%' in dist_df['Unnamed: 0'].values else 0,
                            'min_tokens': dist_df.loc[dist_df['Unnamed: 0'] == 'min', 'token_length'].iloc[0] if 'min' in dist_df['Unnamed: 0'].values else 0,
                            'max_tokens': dist_df.loc[dist_df['Unnamed: 0'] == 'max', 'token_length'].iloc[0] if 'max' in dist_df['Unnamed: 0'].values else 0,
                            'std_tokens': dist_df.loc[dist_df['Unnamed: 0'] == 'std', 'token_length'].iloc[0] if 'std' in dist_df['Unnamed: 0'].values else 0
                        }
                        all_token_stats.append(token_stats)
            
            print(f"  ✅ {name} results processed successfully")
            
        except Exception as e:
            print(f"  ❌ Error processing results from {name}: {e}")
            continue
    
    # Create consolidated summaries (same as before)
    print(f"\n📝 Creating consolidated summaries...")
    
    # Description summary
    if all_description_stats:
        desc_summary_df = pd.DataFrame(all_description_stats)
        desc_summary_file = summary_output_dir / 'description_summary.csv'
        desc_summary_df.to_csv(desc_summary_file, index=False)
        print(f"  ✅ Description summary saved to: {desc_summary_file}")
    
    # Token summary
    if all_token_stats:
        token_summary_df = pd.DataFrame(all_token_stats)
        token_summary_file = summary_output_dir / 'token_summary.csv'
        token_summary_df.to_csv(token_summary_file, index=False)
        print(f"  ✅ Token summary saved to: {token_summary_file}")
    
    # Combined log file
    if all_logs:
        combined_log_file = summary_output_dir / 'combined_analysis_log.txt'
        with open(combined_log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMBINED DATASET ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total datasets analyzed: {len(datasets_to_run)}\n")
            f.write("="*80 + "\n")
            f.write("\n".join(all_logs))
        print(f"  ✅ Combined log saved to: {combined_log_file}")
    
    # Create overall statistics
    if all_description_stats and all_token_stats:
        overall_stats_file = summary_output_dir / 'overall_statistics.txt'
        with open(overall_stats_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OVERALL STATISTICS ACROSS ALL DATASETS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total datasets: {len(all_description_stats)}\n\n")
            
            # Description statistics
            f.write("DESCRIPTION LENGTH STATISTICS:\n")
            f.write("-" * 40 + "\n")
            total_samples = sum(s['total_samples'] for s in all_description_stats)
            weighted_mean_desc = sum(s['mean_length'] * s['total_samples'] for s in all_description_stats) / total_samples
            f.write(f"Total samples: {total_samples:,}\n")
            f.write(f"Weighted mean length: {weighted_mean_desc:.2f} words\n")
            f.write(f"Min length (across datasets): {min(s['min_length'] for s in all_description_stats)} words\n")
            f.write(f"Max length (across datasets): {max(s['max_length'] for s in all_description_stats)} words\n\n")
            
            # Token statistics
            f.write("TOKEN LENGTH STATISTICS:\n")
            f.write("-" * 40 + "\n")
            weighted_mean_tokens = sum(s['mean_tokens'] * s['total_samples'] for s in all_token_stats) / total_samples
            f.write(f"Weighted mean tokens: {weighted_mean_tokens:.2f} tokens\n")
            f.write(f"Min tokens (across datasets): {min(s['min_tokens'] for s in all_token_stats)} tokens\n")
            f.write(f"Max tokens (across datasets): {max(s['max_tokens'] for s in all_token_stats)} tokens\n")
        
        print(f"  ✅ Overall statistics saved to: {overall_stats_file}")
    
    # Create summary visualizations (same as before)
    if all_description_stats and all_token_stats:
        print(f"\n📊 Creating summary visualizations...")
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Summary Analysis Across All Datasets', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        datasets = [s['dataset'] for s in all_description_stats]
        desc_means = [s['mean_length'] for s in all_description_stats]
        desc_medians = [s['median_length'] for s in all_description_stats]
        desc_stds = [s['std_length'] for s in all_description_stats]
        sample_counts = [s['total_samples'] for s in all_description_stats]
        
        token_means = [s['mean_tokens'] for s in all_token_stats]
        token_medians = [s['median_tokens'] for s in all_token_stats]
        token_stds = [s['std_tokens'] for s in all_token_stats]
        
        # 1. Description length comparison
        x_pos = np.arange(len(datasets))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, desc_means, width, label='Mean', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x_pos + width/2, desc_medians, width, label='Median', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Datasets')
        axes[0, 0].set_ylabel('Description Length (words)')
        axes[0, 0].set_title('Description Length Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Token length comparison
        axes[0, 1].bar(x_pos - width/2, token_means, width, label='Mean', alpha=0.8, color='lightgreen')
        axes[0, 1].bar(x_pos + width/2, token_medians, width, label='Median', alpha=0.8, color='orange')
        axes[0, 1].set_xlabel('Datasets')
        axes[0, 1].set_ylabel('Token Length (tokens)')
        axes[0, 1].set_title('Token Length Comparison')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Sample size comparison
        axes[1, 0].bar(x_pos, sample_counts, alpha=0.8, color='purple')
        axes[1, 0].set_xlabel('Datasets')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Dataset Size Comparison')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add sample count labels on bars
        for i, v in enumerate(sample_counts):
            axes[1, 0].text(i, v + max(sample_counts)*0.01, f'{v:,}', ha='center', va='bottom', fontsize=8)
        
        # 4. Description vs Token length scatter plot
        axes[1, 1].scatter(desc_means, token_means, s=100, alpha=0.7, c='red')
        
        # Add dataset labels to scatter points
        for i, dataset in enumerate(datasets):
            axes[1, 1].annotate(dataset.replace('_', '\n'), 
                               (desc_means[i], token_means[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, ha='left')
        
        axes[1, 1].set_xlabel('Mean Description Length (words)')
        axes[1, 1].set_ylabel('Mean Token Length (tokens)')
        axes[1, 1].set_title('Description vs Token Length Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation line
        z = np.polyfit(desc_means, token_means, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(desc_means, p(desc_means), "r--", alpha=0.8, linewidth=2)
        
        # Calculate and display correlation coefficient
        correlation = np.corrcoef(desc_means, token_means)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=axes[1, 1].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=10)
        
        plt.tight_layout()
        
        # Save the summary plot
        summary_plot_path = summary_output_dir / 'summary_analysis_plots.png'
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Summary plots saved to: {summary_plot_path}")
        
        # Create individual comparison plots
        # Description length distribution comparison
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create violin plot for description lengths
        desc_data = []
        labels = []
        for stats in all_description_stats:
            # For violin plot, we'll use the mean and std to create a normal distribution approximation
            # This is a simplified representation since we don't have the raw data
            mean = stats['mean_length']
            std = stats['std_length']
            # Generate synthetic data based on normal distribution
            synthetic_data = np.random.normal(mean, std, min(1000, stats['total_samples']))
            desc_data.append(synthetic_data)
            labels.append(stats['dataset'].replace('_', '\n'))
        
        parts = ax.violinplot(desc_data, positions=range(len(datasets)), showmeans=True, showmedians=True)
        
        # Customize violin plot colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Description Length (words)')
        ax.set_title('Description Length Distribution Comparison')
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save the violin plot
        violin_plot_path = summary_output_dir / 'description_distribution_comparison.png'
        plt.savefig(violin_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Description distribution plot saved to: {violin_plot_path}")
    
    print(f"\n🎉 Summary analysis completed!")
    print(f"   Results saved to: {summary_output_dir}")
    
    return True


def run_analysis_for_dataset(dataset_config, analysis_options, global_settings):
    """Run analysis for a single dataset."""
    name = dataset_config['name']
    csv_path = dataset_config['csv_path']
    output_base_dir = Path(dataset_config['output_base_dir'])
    image_prefix = dataset_config['image_prefix']
    
    print(f"\n{'='*80}")
    print(f"ANALYZING DATASET: {name}")
    print(f"{'='*80}")
    print(f"CSV Path: {csv_path}")
    print(f"Output Base Dir: {output_base_dir}")
    print(f"Image Prefix: {image_prefix}")
    
    # Check if CSV exists
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        print(f"❌ Skipping {name}: CSV file not found")
        return False
    
    # Create dataset-specific output directory
    dataset_output_dir = output_base_dir / name
    
    # Check if output directory exists and handle overwrite
    if dataset_output_dir.exists():
        if global_settings.get('overwrite_existing', False):
            print(f"⚠️  Output directory exists, overwriting: {dataset_output_dir}")
        else:
            print(f"❌ Output directory already exists: {dataset_output_dir}")
            print("   Set 'overwrite_existing: true' in config to overwrite")
            return False
    
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Run individual analyses (always run these)
    # Run description analysis
    if analysis_options.get('run_description_analysis', True):
        try:
            print(f"\n📊 Running description analysis...")
            desc_output_dir = dataset_output_dir / "description_analysis"
            analyze_descriptions(csv_path, str(desc_output_dir))
            print(f"✅ Description analysis completed")
        except Exception as e:
            print(f"❌ Description analysis failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            success = False
    
    # Run token analysis
    if analysis_options.get('run_token_analysis', True):
        try:
            print(f"\n🔤 Running token analysis...")
            token_output_dir = dataset_output_dir / "token_analysis"
            analyze_token_lengths(csv_path, str(token_output_dir))
            print(f"✅ Token analysis completed")
        except Exception as e:
            print(f"❌ Token analysis failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            success = False
    
    # Run image examples generation
    if analysis_options.get('run_image_examples', True):
        try:
            print(f"\n🖼️  Running image examples generation...")
            examples_output_dir = dataset_output_dir / "image_examples"
            generate_image_examples(csv_path, image_prefix, str(examples_output_dir))
            print(f"✅ Image examples generation completed")
        except Exception as e:
            print(f"❌ Image examples generation failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            success = False
    
    if success:
        print(f"\n🎉 All analyses completed successfully for {name}")
        print(f"   Results saved to: {dataset_output_dir}")
    else:
        print(f"\n⚠️  Some analyses failed for {name}")
    
    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run dataset analysis on multiple CSV files')
    parser.add_argument('--config', '-c', 
                       default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--dataset', '-d',
                       help='Run analysis only for specific dataset name')
    parser.add_argument('--list-datasets', '-l',
                       action='store_true',
                       help='List available datasets and exit')
    
    args = parser.parse_args()
    
    # Load and validate config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please create a config.yaml file or specify a different path with --config")
        sys.exit(1)
    
    config = load_config(config_path)
    validate_config(config)
    
    # List datasets if requested
    if args.list_datasets:
        print("Available datasets:")
        for i, dataset in enumerate(config['datasets']):
            csv_exists = "✅" if Path(dataset['csv_path']).exists() else "❌"
            print(f"  {i+1}. {dataset['name']} {csv_exists}")
            print(f"     CSV: {dataset['csv_path']}")
            print(f"     Output: {dataset['output_base_dir']}")
        sys.exit(0)
    
    # Filter datasets if specific one requested
    datasets_to_run = config['datasets']
    if args.dataset:
        datasets_to_run = [d for d in config['datasets'] if d['name'] == args.dataset]
        if not datasets_to_run:
            print(f"Error: Dataset '{args.dataset}' not found in config")
            print("Available datasets:", [d['name'] for d in config['datasets']])
            sys.exit(1)
    
    # Print configuration summary
    print("="*80)
    print("DATASET ANALYSIS RUNNER")
    print("="*80)
    print(f"Config file: {config_path.absolute()}")
    print(f"Datasets to analyze: {len(datasets_to_run)}")
    print(f"Description analysis: {'✅' if config['analysis_options'].get('run_description_analysis', True) else '❌'}")
    print(f"Token analysis: {'✅' if config['analysis_options'].get('run_token_analysis', True) else '❌'}")
    print(f"Image examples: {'✅' if config['analysis_options'].get('run_image_examples', True) else '❌'}")
    print(f"Whole dataset analysis: {'✅' if config['analysis_options'].get('whole_dataset_analysis', False) else '❌'}")
    print(f"Summary-only mode: {'✅' if config['analysis_options'].get('summary_only_mode', False) else '❌'}")
    print(f"NaN detection only: {'✅' if config['analysis_options'].get('nan_detection_only', False) else '❌'}")
    print("="*80)
    
    # Check if we should run NaN detection only
    if config['analysis_options'].get('nan_detection_only', False):
        print(f"\n{'='*80}")
        print("NaN DETECTION ONLY MODE")
        print(f"{'='*80}")
        
        try:
            success = detect_nan_descriptions(
                datasets_to_run,
                config['analysis_options'],
                config.get('global_settings', {})
            )
            if success:
                successful_runs = len(datasets_to_run)
                failed_runs = 0
                print(f"✅ NaN detection completed successfully")
            else:
                successful_runs = 0
                failed_runs = len(datasets_to_run)
                print(f"⚠️  NaN detection had some issues")
        except KeyboardInterrupt:
            print(f"\n⚠️  NaN detection interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error in NaN detection: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            successful_runs = 0
            failed_runs = len(datasets_to_run)
    
    # Check if we should run in summary-only mode
    elif config['analysis_options'].get('summary_only_mode', False):
        print(f"\n{'='*80}")
        print("SUMMARY-ONLY MODE: Checking existing individual analyses")
        print(f"{'='*80}")
        
        # Check if all individual analyses exist
        all_exist, missing = check_individual_analyses_exist(datasets_to_run, config['analysis_options'])
        
        if not all_exist:
            print("❌ Cannot run summary-only mode. Missing individual analyses:")
            for item in missing:
                print(f"   • {item}")
            print("\n💡 To fix this, either:")
            print("   1. Set 'summary_only_mode: false' to run individual analyses first")
            print("   2. Run individual analyses manually")
            sys.exit(1)
        
        print("✅ All individual analyses found!")
        
        # Run summary analysis only
        try:
            success = create_summary_from_existing_results(
                datasets_to_run,
                config['analysis_options'],
                config.get('global_settings', {})
            )
            if success:
                successful_runs = len(datasets_to_run)
                failed_runs = 0
                print(f"✅ Summary analysis completed successfully")
            else:
                successful_runs = 0
                failed_runs = len(datasets_to_run)
                print(f"⚠️  Summary analysis had some issues")
        except KeyboardInterrupt:
            print(f"\n⚠️  Summary analysis interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error in summary analysis: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            successful_runs = 0
            failed_runs = len(datasets_to_run)
    
    else:
        # Run individual dataset analysis first
        successful_runs = 0
        failed_runs = 0
        
        for dataset_config in datasets_to_run:
            try:
                success = run_analysis_for_dataset(
                    dataset_config, 
                    config['analysis_options'], 
                    config.get('global_settings', {})
                )
                if success:
                    successful_runs += 1
                else:
                    failed_runs += 1
            except KeyboardInterrupt:
                print(f"\n⚠️  Analysis interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error analyzing {dataset_config['name']}: {e}")
                print(f"   Traceback: {traceback.format_exc()}")
                failed_runs += 1
        
        # If summary analysis is enabled, create summary after individual analyses
        if config['analysis_options'].get('whole_dataset_analysis', False):
            print(f"\n{'='*80}")
            print("CREATING SUMMARY ANALYSIS FROM INDIVIDUAL RESULTS")
            print(f"{'='*80}")
            
            try:
                success = create_summary_from_existing_results(
                    datasets_to_run,
                    config['analysis_options'],
                    config.get('global_settings', {})
                )
                if success:
                    print(f"✅ Summary analysis completed successfully")
                else:
                    print(f"⚠️  Summary analysis had some issues")
            except KeyboardInterrupt:
                print(f"\n⚠️  Summary analysis interrupted by user")
            except Exception as e:
                print(f"\n❌ Unexpected error in summary analysis: {e}")
                print(f"   Traceback: {traceback.format_exc()}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets: {len(datasets_to_run)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    if failed_runs > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
