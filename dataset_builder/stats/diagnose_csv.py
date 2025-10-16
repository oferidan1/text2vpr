#!/usr/bin/env python3
"""
Diagnose CSV parsing issues by examining the problematic lines.
"""

import pandas as pd
import argparse
from pathlib import Path


def diagnose_csv(csv_path, problem_line=206):
    """
    Diagnose CSV parsing issues by examining specific lines.
    
    Args:
        csv_path: Path to the CSV file
        problem_line: Line number that's causing issues (1-indexed)
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"🔍 Diagnosing CSV file: {csv_path}")
    print(f"📋 Examining line {problem_line} and surrounding lines...")
    print("="*80)
    
    # Read the file line by line to examine the problematic area
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines in file: {total_lines}")
    
    # Show header
    if total_lines > 0:
        print(f"\n📄 Header (line 1):")
        print(f"   {repr(lines[0])}")
    
    # Show lines around the problem
    start_line = max(1, problem_line - 3)
    end_line = min(total_lines, problem_line + 3)
    
    print(f"\n🔍 Lines around problem area ({start_line}-{end_line}):")
    for i in range(start_line - 1, end_line):
        line_num = i + 1
        line = lines[i]
        marker = " ⚠️  PROBLEM" if line_num == problem_line else ""
        print(f"   {line_num:3d}: {repr(line)}{marker}")
    
    # Count commas in each line
    print(f"\n📊 Comma count analysis:")
    for i in range(start_line - 1, end_line):
        line_num = i + 1
        line = lines[i]
        comma_count = line.count(',')
        marker = " ⚠️  PROBLEM" if line_num == problem_line else ""
        print(f"   Line {line_num:3d}: {comma_count} commas{marker}")
    
    # Try different parsing approaches
    print(f"\n🧪 Testing different parsing approaches:")
    
    # Test 1: Default parsing
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ Default parsing: SUCCESS ({len(df)} rows)")
    except Exception as e:
        print(f"   ❌ Default parsing: FAILED - {e}")
    
    # Test 2: With quoting=1 (QUOTE_ALL)
    try:
        df = pd.read_csv(csv_path, quoting=1, escapechar='\\')
        print(f"   ✅ QUOTE_ALL parsing: SUCCESS ({len(df)} rows)")
    except Exception as e:
        print(f"   ❌ QUOTE_ALL parsing: FAILED - {e}")
    
    # Test 3: With quoting=3 (QUOTE_NONE)
    try:
        df = pd.read_csv(csv_path, quoting=3, escapechar='\\')
        print(f"   ✅ QUOTE_NONE parsing: SUCCESS ({len(df)} rows)")
    except Exception as e:
        print(f"   ❌ QUOTE_NONE parsing: FAILED - {e}")
    
    # Test 4: Skip bad lines
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip', quoting=1, escapechar='\\')
        print(f"   ✅ Skip bad lines: SUCCESS ({len(df)} rows)")
    except Exception as e:
        print(f"   ❌ Skip bad lines: FAILED - {e}")
    
    # Test 5: Try to identify the issue
    print(f"\n🔧 Suggested fixes:")
    print(f"   1. Check if line {problem_line} has unescaped quotes or commas")
    print(f"   2. Ensure descriptions with commas are properly quoted")
    print(f"   3. Check for special characters that might break CSV format")
    print(f"   4. Consider using the 'skip bad lines' option in the analysis scripts")


def main():
    parser = argparse.ArgumentParser(description='Diagnose CSV parsing issues')
    parser.add_argument('csv_path', help='Path to CSV file to diagnose')
    parser.add_argument('--line', '-l', type=int, default=206, 
                       help='Problem line number (default: 206)')
    
    args = parser.parse_args()
    
    diagnose_csv(args.csv_path, args.line)


if __name__ == "__main__":
    main()
