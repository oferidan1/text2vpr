#!/usr/bin/env python3
"""
Text to CSV Converter Script

This script converts a text file to a properly formatted CSV file.
It handles empty values, validates the data, and ensures proper CSV formatting.

Usage:
    python txt_to_csv_converter.py input_file.txt output_file.csv
    python txt_to_csv_converter.py input_file.txt  # Uses input_file.csv as output
"""

import csv
import sys
import os
import argparse
from pathlib import Path


def convert_txt_to_csv(input_file, output_file=None):
    """
    Convert a text file to CSV format.
    
    Args:
        input_file (str): Path to the input text file
        output_file (str, optional): Path to the output CSV file. 
                                   If None, uses input_file with .csv extension
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Validate input file
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist.")
            return False
        
        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.with_suffix('.csv')
        
        print(f"Converting '{input_file}' to '{output_file}'...")
        
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        if not lines:
            print("Error: Input file is empty.")
            return False
        
        # Process the data
        processed_rows = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Split by comma and clean up the data
            row = [field.strip() for field in line.split(',')]
            
            # Handle empty fields (multiple consecutive commas)
            cleaned_row = []
            for field in row:
                if field == '':
                    cleaned_row.append('')  # Keep empty fields as empty strings
                else:
                    cleaned_row.append(field)
            
            processed_rows.append(cleaned_row)
        
        if not processed_rows:
            print("Error: No valid data found in input file.")
            return False
        
        # Write to CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(processed_rows)
        
        print(f"Successfully converted {len(processed_rows)} rows to '{output_file}'")
        
        # Display some statistics
        if processed_rows:
            header = processed_rows[0]
            print(f"CSV has {len(header)} columns: {', '.join(header[:5])}{'...' if len(header) > 5 else ''}")
            print(f"Data rows: {len(processed_rows) - 1}")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description='Convert a text file to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python txt_to_csv_converter.py data.txt
  python txt_to_csv_converter.py data.txt output.csv
  python txt_to_csv_converter.py /path/to/input.txt /path/to/output.csv
        """
    )
    
    parser.add_argument('input_file', 
                       help='Path to the input text file')
    parser.add_argument('output_file', 
                       nargs='?', 
                       help='Path to the output CSV file (optional)')
    parser.add_argument('-v', '--verbose', 
                       action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Input file: {args.input_file}")
        print(f"Output file: {args.output_file or 'auto-generated'}")
    
    success = convert_txt_to_csv(args.input_file, args.output_file)
    
    if success:
        print("Conversion completed successfully!")
        sys.exit(0)
    else:
        print("Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
