#!/usr/bin/env python3
"""
Generate image examples with descriptions from a CSV dataset.
Creates a visualization showing 3 random images with their descriptions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np
import argparse
import random
from datetime import datetime


def generate_image_examples(csv_path, image_prefix_path, output_dir=None, num_examples=3, seed=None):
    """
    Generate image examples with descriptions from a CSV dataset.
    
    Args:
        csv_path: Path to CSV file with 'image_path' and 'description' columns
        image_prefix_path: Prefix path to prepend to image_path from CSV
        output_dir: Directory to save the output. If None, saves to './output'
        num_examples: Number of random examples to show (default: 3)
        seed: Random seed for reproducible results (optional)
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
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
    
    # Clean column names
    df.columns = df.columns.str.strip().str.strip("'\"")
    print(f"CSV columns found: {df.columns.tolist()}")
    
    # Find the image and description columns
    image_col = None
    description_col = None
    
    for col in df.columns:
        if 'image' in col.lower() and 'path' in col.lower():
            image_col = col
            print(f"Found image column: '{image_col}'")
        if 'description' in col.lower():
            description_col = col
            print(f"Found description column: '{description_col}'")
    
    if image_col is None:
        raise ValueError(f"Could not find image path column! Available columns: {df.columns.tolist()}")
    if description_col is None:
        raise ValueError(f"Could not find description column! Available columns: {df.columns.tolist()}")
    
    print(f"Total samples: {len(df)}")
    
    # Select random examples
    if len(df) < num_examples:
        print(f"Warning: Only {len(df)} samples available, showing all of them")
        selected_indices = list(range(len(df)))
    else:
        selected_indices = random.sample(range(len(df)), num_examples)
    
    print(f"Selected {len(selected_indices)} random examples")
    
    # Create separate images for each example
    for i, idx in enumerate(selected_indices):
        row = df.iloc[idx]
        image_path = row[image_col]
        description = row[description_col]
        
        # Construct full image path
        full_image_path = Path(image_prefix_path) / image_path
        
        print(f"\nExample {i+1}:")
        print(f"  Image path: {full_image_path}")
        print(f"  Description: {description[:100]}...")
        
        # Create individual figure for this example
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(f'Example {i+1}: {full_image_path.name}', fontsize=16, fontweight='bold')
        
        # Check if image exists
        if not full_image_path.exists():
            print(f"  ⚠️  Warning: Image file not found: {full_image_path}")
            # Create a placeholder
            ax.text(0.5, 0.5, f"Image not found:\n{full_image_path.name}", 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            try:
                # Load and display the image
                img = mpimg.imread(full_image_path)
                ax.imshow(img)
            except Exception as e:
                print(f"  ❌ Error loading image: {e}")
                ax.text(0.5, 0.5, f"Error loading image:\n{str(e)}", 
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add description below the image
        description_text = f"Description: {description}"
        # Wrap long descriptions
        if len(description_text) > 100:
            # Simple word wrapping
            words = description_text.split()
            lines = []
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) <= 100:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
            if current_line:
                lines.append(' '.join(current_line))
            description_text = '\n'.join(lines)
        
        ax.text(0.5, -0.15, description_text, ha='center', va='top', 
               fontsize=10, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Save individual image
        output_path = output_dir / f'example_{i+1:02d}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
        plt.close()  # Close the figure to free memory
    
    # Save details to a text file
    details_file = output_dir / 'examples_details.txt'
    with open(details_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IMAGE EXAMPLES DETAILS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CSV file: {Path(csv_path).absolute()}\n")
        f.write(f"Image prefix: {image_prefix_path}\n")
        f.write(f"Number of examples: {len(selected_indices)}\n")
        f.write(f"Random seed: {seed if seed is not None else 'None'}\n")
        f.write("="*80 + "\n\n")
        
        for i, idx in enumerate(selected_indices):
            row = df.iloc[idx]
            image_path = row[image_col]
            description = row[description_col]
            full_image_path = Path(image_prefix_path) / image_path
            
            f.write(f"EXAMPLE {i+1}:\n")
            f.write(f"  Index: {idx}\n")
            f.write(f"  Image path: {image_path}\n")
            f.write(f"  Full path: {full_image_path}\n")
            f.write(f"  Exists: {'Yes' if full_image_path.exists() else 'No'}\n")
            f.write(f"  Description: {description}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Saved details to: {details_file}")
    
    # Also save the selected examples as a CSV
    selected_df = df.iloc[selected_indices].copy()
    selected_df['full_image_path'] = [str(Path(image_prefix_path) / path) for path in selected_df[image_col]]
    selected_df['image_exists'] = [Path(image_prefix_path, path).exists() for path in selected_df[image_col]]
    
    examples_csv = output_dir / 'selected_examples.csv'
    selected_df.to_csv(examples_csv, index=False)
    print(f"Saved selected examples CSV to: {examples_csv}")
    
    print(f"\n{'='*60}")
    print("Image examples generation complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")
    
    return selected_df


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Generate image examples with descriptions from CSV')
    parser.add_argument('csv_path', help='Path to CSV file with image paths and descriptions')
    parser.add_argument('image_prefix', help='Prefix path to prepend to image paths from CSV')
    parser.add_argument('--output-dir', '-o', default=None, 
                       help='Output directory (default: ./output)')
    parser.add_argument('--num-examples', '-n', type=int, default=3,
                       help='Number of random examples to show (default: 3)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    try:
        selected_df = generate_image_examples(
            csv_path=args.csv_path,
            image_prefix_path=args.image_prefix,
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            seed=args.seed
        )
        print(f"\n✅ Successfully generated {len(selected_df)} image examples!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
