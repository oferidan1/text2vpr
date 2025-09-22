#!/usr/bin/env python3
"""
Script to change image paths in CSV files while preserving base filenames

This script reads a CSV file and updates all image path columns to use a new base path
while keeping the original filenames intact.

Usage:
    python change_image_paths.py --input output.csv --output new_output.csv --new_path /new/base/path/
"""

import os
import csv
import argparse
import pandas as pd
from pathlib import Path
import sys


def change_image_paths(input_csv: str, output_csv: str, new_base_path: str, 
                      query_path_col: str = 'query_image_path',
                      change_references: bool = False,
                      reference_path_prefix: str = 'reference_',
                      reference_path_suffix: str = '_path') -> None:
    """
    Change image paths in CSV file while preserving base filenames
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        new_base_path: New base path for images
        query_path_col: Name of the query image path column
        change_references: Whether to also change reference image paths (default: False)
        reference_path_prefix: Prefix for reference path columns (e.g., 'reference_')
        reference_path_suffix: Suffix for reference path columns (e.g., '_path')
    """
    print(f"📖 Reading input CSV: {input_csv}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_csv)
        print(f"✅ Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Ensure new_base_path ends with a separator
    new_base_path = new_base_path.rstrip('/\\') + os.sep
    
    print(f"🔄 Changing image paths to: {new_base_path}")
    
    # Find path columns to process
    path_columns = []
    
    # Add query path column if it exists
    if query_path_col in df.columns:
        path_columns.append(query_path_col)
        print(f"   Found query path column: {query_path_col}")
    
    # Find reference path columns only if change_references is True
    if change_references:
        ref_path_columns = []
        for col in df.columns:
            if col.startswith(reference_path_prefix) and col.endswith(reference_path_suffix):
                ref_path_columns.append(col)
        
        if ref_path_columns:
            print(f"   Found {len(ref_path_columns)} reference path columns: {ref_path_columns}")
        
        path_columns.extend(ref_path_columns)
    else:
        print("   Skipping reference path columns (use --change_references to include them)")
    
    if not path_columns:
        print("⚠️  No image path columns found!")
        return
    
    # Process each path column
    changes_made = 0
    for col in path_columns:
        print(f"   Processing column: {col}")
        
        for idx, path in enumerate(df[col]):
            if pd.isna(path) or path == '' or str(path).lower() == 'nan':
                continue
            
            try:
                # Extract filename from the path
                # Handle both Windows and Unix paths properly
                path_str = str(path)
                
                # Split by both forward and backward slashes to handle mixed paths
                path_parts = path_str.replace('\\', '/').split('/')
                
                # Get the last non-empty part as filename
                filename = None
                for part in reversed(path_parts):
                    if part.strip():  # Skip empty parts
                        filename = part
                        break
                
                if not filename:
                    print(f"     ⚠️  Could not extract filename from: {path}")
                    continue
                
                # Create new path
                new_path = new_base_path + filename
                
                # Update the dataframe
                df.at[idx, col] = new_path
                changes_made += 1
                
                # Show first few changes as examples
                if changes_made <= 3:
                    print(f"     {path} -> {new_path}")
                
            except Exception as e:
                print(f"     ⚠️  Error processing path '{path}': {e}")
                continue
    
    print(f"✅ Made {changes_made} path changes")
    
    # Save the updated CSV
    print(f"💾 Saving updated CSV to: {output_csv}")
    try:
        df.to_csv(output_csv, index=False)
        print(f"✅ Successfully saved updated CSV")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        return
    
    # Show summary
    print(f"\n📊 Summary:")
    print(f"   Input file: {input_csv}")
    print(f"   Output file: {output_csv}")
    print(f"   New base path: {new_base_path}")
    print(f"   Path columns processed: {len(path_columns)}")
    print(f"   Total changes made: {changes_made}")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Change image paths in CSV files while preserving filenames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (only changes query paths)
  python change_image_paths.py --input output.csv --output new_output.csv --new_path /new/base/path/
  
  # Change both query and reference paths
  python change_image_paths.py --input output.csv --output new_output.csv --new_path /new/base/path/ --change_references
  
  # With custom column names
  python change_image_paths.py --input output.csv --output new_output.csv --new_path /new/base/path/ --query_col image_path --ref_prefix ref_ --ref_suffix _image_path
        """
    )
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                       help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True,
                       help="Output CSV file path")
    parser.add_argument("--new_path", type=str, required=True,
                       help="New base path for images (will be combined with original filenames)")
    
    # Optional arguments
    parser.add_argument("--query_col", type=str, default="query_image_path",
                       help="Name of the query image path column (default: query_image_path)")
    parser.add_argument("--change_references", action="store_true",
                       help="Also change reference image paths (default: only change query paths)")
    parser.add_argument("--ref_prefix", type=str, default="reference_",
                       help="Prefix for reference path columns (default: reference_)")
    parser.add_argument("--ref_suffix", type=str, default="_path",
                       help="Suffix for reference path columns (default: _path)")
    parser.add_argument("--preview", action="store_true",
                       help="Preview changes without saving (shows first 5 changes)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    # Validate new path (create directory if it doesn't exist)
    if not os.path.exists(args.new_path):
        try:
            os.makedirs(args.new_path, exist_ok=True)
            print(f"📁 Created directory: {args.new_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create directory {args.new_path}: {e}")
    
    # Preview mode
    if args.preview:
        print("🔍 Preview mode - showing first 5 changes:")
        # This would require implementing a preview function
        print("Preview mode not yet implemented. Use without --preview to make changes.")
        return 0
    
    # Process the CSV
    try:
        change_image_paths(
            input_csv=args.input,
            output_csv=args.output,
            new_base_path=args.new_path,
            query_path_col=args.query_col,
            change_references=args.change_references,
            reference_path_prefix=args.ref_prefix,
            reference_path_suffix=args.ref_suffix
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
