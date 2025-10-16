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
    if global_settings.get('use_timestamps', True):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dataset_output_dir = output_base_dir / f"{name}_{timestamp}"
    else:
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
    print("="*80)
    
    # Run analysis for each dataset
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
