"""
Visualization utilities for VPR Dataset Analysis

Provides advanced plotting capabilities including:
- UTM coordinate scatter plots
- Distance distribution histograms
- Top-k nearest neighbor visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
from typing import List, Tuple
import math
import textwrap


class VPRVisualizer:
    """Visualization utilities for VPR analysis results."""
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize visualizer with results DataFrame.
        
        Args:
            results_df: DataFrame with VPR analysis results
        """
        self.results = results_df
        self._extract_coordinates()
    
    def _extract_coordinates(self):
        """Extract UTM coordinates from results DataFrame."""
        self.query_coords = list(zip(self.results['utm_east'], self.results['utm_north']))
        
        # Extract reference coordinates for each top-k
        self.reference_coords = []
        self.physical_distances = []
        k = 1
        while f'reference_{k}_utm_east' in self.results.columns:
            ref_coords = list(zip(
                self.results[f'reference_{k}_utm_east'],
                self.results[f'reference_{k}_utm_north']
            ))
            self.reference_coords.append(ref_coords)
            
            # Calculate physical distances in meters
            distances = []
            for i, (query_coord, ref_coord) in enumerate(zip(self.query_coords, ref_coords)):
                if not (pd.isna(query_coord[0]) or pd.isna(query_coord[1]) or 
                        pd.isna(ref_coord[0]) or pd.isna(ref_coord[1])):
                    dist = self._calculate_distance_meters(query_coord, ref_coord)
                    distances.append(dist)
                else:
                    distances.append(np.nan)
            
            self.physical_distances.append(distances)
            k += 1
    
    def _calculate_distance_meters(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two UTM coordinates in meters.
        
        Args:
            coord1: First UTM coordinate (east, north)
            coord2: Second UTM coordinate (east, north)
            
        Returns:
            Distance in meters
        """
        east1, north1 = coord1
        east2, north2 = coord2
        return math.sqrt((east2 - east1)**2 + (north2 - north1)**2)
    
    def _shorten_filename(self, filepath: str, max_length: int = 15) -> str:
        """
        Shorten a filename for display purposes.
        
        Args:
            filepath: Full file path
            max_length: Maximum length for display
            
        Returns:
            Shortened filename with ellipsis if needed
        """
        filename = os.path.basename(filepath)
        if len(filename) <= max_length:
            return filename
        
        # Keep the extension and shorten the base name
        name, ext = os.path.splitext(filename)
        if len(ext) >= max_length - 3:
            return filename[:max_length-3] + "..."
        
        available_length = max_length - len(ext) - 3  # 3 for "..."
        if available_length > 0:
            return name[:available_length] + "..." + ext
        else:
            return filename[:max_length-3] + "..."
    
    def _wrap_text(self, text: str, max_width: int = 50) -> str:
        """
        Wrap text to fit within a specified width.
        
        Args:
            text: Text to wrap
            max_width: Maximum width in characters
            
        Returns:
            Wrapped text
        """
        if not text or len(text) <= max_width:
            return text
        
        wrapped_lines = textwrap.wrap(text, width=max_width)
        return '\n'.join(wrapped_lines)
    
    def _has_reference_images(self, query_idx: int) -> bool:
        """
        Check if a query has any reference images.
        
        Args:
            query_idx: Index of the query to check
            
        Returns:
            True if query has at least one reference image, False otherwise
        """
        if query_idx >= len(self.results):
            return False
        
        query_row = self.results.iloc[query_idx]
        
        # Check if any reference path columns have valid data
        k = 1
        while f'reference_{k}_path' in query_row:
            ref_path = query_row[f'reference_{k}_path']
            if pd.notna(ref_path) and str(ref_path).strip() != '':
                return True
            k += 1
        
        return False
    
    def plot_utm_coordinates(self, figsize: Tuple[int, int] = (12, 8), 
                           show_connections: bool = True, max_queries: int = 100):
        """
        Plot UTM coordinates showing query and reference image locations.
        
        Args:
            figsize: Figure size (width, height)
            show_connections: Whether to show lines connecting queries to their nearest neighbors
            max_queries: Maximum number of queries to plot (for performance)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Limit queries for performance
        n_queries = min(max_queries, len(self.query_coords))
        
        # Plot reference images (all)
        all_ref_coords = []
        for ref_list in self.reference_coords:
            all_ref_coords.extend(ref_list)
        
        if all_ref_coords:
            ref_east, ref_north = zip(*all_ref_coords)
            ax.scatter(ref_east, ref_north, c='lightblue', s=20, alpha=0.6, 
                      label='Reference Images', zorder=1)
        
        # Plot query images
        query_east, query_north = zip(*self.query_coords[:n_queries])
        ax.scatter(query_east, query_north, c='red', s=50, marker='*', 
                  label='Query Images', zorder=3)
        
        # Show connections if requested
        if show_connections and n_queries <= 50:  # Limit for clarity
            for i in range(n_queries):
                query_coord = self.query_coords[i]
                
                # Connect to top-1 nearest neighbor
                if self.reference_coords and i < len(self.reference_coords[0]):
                    ref_coord = self.reference_coords[0][i]
                    ax.plot([query_coord[0], ref_coord[0]], 
                           [query_coord[1], ref_coord[1]], 
                           'g-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('UTM East')
        ax.set_ylabel('UTM North')
        ax.set_title('VPR Dataset UTM Coordinates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_distance_distribution(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot distribution of distances to nearest neighbors.
        
        Args:
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Use physical distances instead of original distance columns
        for i, distances in enumerate(self.physical_distances[:4]):  # Plot first 4 distance columns
            if i >= len(axes):
                break
                
            distances_array = np.array(distances)
            valid_distances = distances_array[~np.isnan(distances_array)]
            if len(valid_distances) > 0:
                axes[i].hist(valid_distances, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel('Distance (meters)')
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'Reference {i+1} Distance Distribution')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.physical_distances), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig, axes
    
    def visualize_query_matches(self, query_idx: int, k: int = 3, 
                              figsize: Tuple[int, int] = (18, 10)):
        """
        Visualize a specific query and its top-k nearest neighbors.
        
        Args:
            query_idx: Index of query to visualize
            k: Number of nearest neighbors to show
            figsize: Figure size (width, height)
        """
        if query_idx >= len(self.results):
            raise ValueError(f"Query index {query_idx} out of range")
        
        if not self._has_reference_images(query_idx):
            raise ValueError(f"Query {query_idx} has no reference images to visualize")
        
        query_row = self.results.iloc[query_idx]
        query_path = query_row['query_image_path']
        
        # Create figure with more height to accommodate query description
        fig, axes = plt.subplots(2, k + 1, figsize=figsize, 
                               gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
        if k == 1:
            axes = axes.reshape(2, 1)
        
        # Show query image
        try:
            query_img = Image.open(query_path)
            axes[0, 0].imshow(query_img)
            
            # Very short query filename for display
            short_query_name = self._shorten_filename(query_path, max_length=12)
            axes[0, 0].set_title(f'Query: {short_query_name}', fontsize=9)
            
            # Add hover tooltip for query
            axes[0, 0].text(0.5, 0.5, '', ha='center', va='center', 
                           transform=axes[0, 0].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.1", alpha=0))
            
            axes[0, 0].axis('off')
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                        ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Query Image (Error)')
            axes[0, 0].axis('off')
        
        # Show top-k reference images
        for i in range(k):
            ref_path_col = f'reference_{i+1}_path'
            ref_distance_col = f'reference_{i+1}_distance'
            
            if ref_path_col in query_row and ref_distance_col in query_row:
                ref_path = query_row[ref_path_col]
                ref_distance = query_row[ref_distance_col]
                
                try:
                    ref_img = Image.open(ref_path)
                    # Get physical distance in meters
                    physical_dist = np.nan
                    if i < len(self.physical_distances) and query_idx < len(self.physical_distances[i]):
                        physical_dist = self.physical_distances[i][query_idx]
                    
                    if not np.isnan(physical_dist):
                        distance_text = f'{physical_dist:.1f}m'
                    else:
                        distance_text = 'N/A'
                    
                    axes[0, i + 1].imshow(ref_img)
                    
                    # Very short reference filename for display
                    short_ref_name = self._shorten_filename(ref_path, max_length=12)
                    axes[0, i + 1].set_title(f'Ref{i+1}: {short_ref_name}\n{distance_text}', fontsize=9)
                    
                    # Add hover tooltip for reference
                    axes[0, i + 1].text(0.5, 0.5, '', ha='center', va='center', 
                                       transform=axes[0, i + 1].transAxes, 
                                       bbox=dict(boxstyle="round,pad=0.1", alpha=0))
                    
                    axes[0, i + 1].axis('off')
                except Exception as e:
                    axes[0, i + 1].text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                                    ha='center', va='center', transform=axes[0, i + 1].transAxes)
                    axes[0, i + 1].set_title(f'Ref{i+1} (Error)')
                    axes[0, i + 1].axis('off')
            else:
                axes[0, i + 1].text(0.5, 0.5, 'No reference', 
                                ha='center', va='center', transform=axes[0, i + 1].transAxes)
                axes[0, i + 1].set_title(f'Ref{i+1}')
                axes[0, i + 1].axis('off')
        
        # Add query description spanning full width below images
        if 'query_description' in query_row and pd.notna(query_row['query_description']):
            description = str(query_row['query_description'])
            # Use the full width of the figure for description, positioned at bottom center
            fig.text(0.5, 0.05, f'Query Description: {description}', 
                    ha='center', va='bottom', fontsize=12, wrap=True, 
                    bbox=dict(boxstyle="round,pad=0.8", 
                    facecolor="lightblue", alpha=0.9))
        
        # Hide the bottom row axes
        for i in range(k + 1):
            axes[1, i].axis('off')
        
        # Add simple figure title
        short_query_name = self._shorten_filename(query_path, max_length=20)
        fig.suptitle(f'Query {query_idx}: {short_query_name}', fontsize=10, y=0.95)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_performance_metrics(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot performance metrics and statistics.
        
        Args:
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Distance statistics using physical distances
        if self.physical_distances:
            # Box plot of distances
            valid_distances = []
            labels = []
            for i, distances in enumerate(self.physical_distances):
                distances_array = np.array(distances)
                valid_dist = distances_array[~np.isnan(distances_array)]
                if len(valid_dist) > 0:
                    valid_distances.append(valid_dist)
                    labels.append(f'Ref {i+1}')
            
            if valid_distances:
                axes[0].boxplot(valid_distances)
                axes[0].set_xticklabels(labels, rotation=45)
                axes[0].set_ylabel('Distance (meters)')
                axes[0].set_title('Distance Distribution by Rank')
                axes[0].grid(True, alpha=0.3)
                
                # Mean distance by rank
                mean_distances = [np.mean(dist) for dist in valid_distances]
                ranks = list(range(1, len(valid_distances) + 1))
                axes[1].plot(ranks, mean_distances, 'bo-')
                axes[1].set_xlabel('Rank')
                axes[1].set_ylabel('Mean Distance (meters)')
                axes[1].set_title('Mean Distance vs Rank')
                axes[1].grid(True, alpha=0.3)
        
        # UTM coordinate ranges
        query_east, query_north = zip(*self.query_coords)
        axes[2].scatter(query_east, query_north, c='red', s=20, alpha=0.7)
        axes[2].set_xlabel('UTM East')
        axes[2].set_ylabel('UTM North')
        axes[2].set_title('Query Image Distribution')
        axes[2].grid(True, alpha=0.3)
        
        # Dataset statistics
        stats_text = f"""
        Total Queries: {len(self.results)}
        UTM East Range: {min(query_east):.2f} - {max(query_east):.2f}
        UTM North Range: {min(query_north):.2f} - {max(query_north):.2f}
        """
        axes[3].text(0.1, 0.5, stats_text, transform=axes[3].transAxes, 
                     fontsize=12, verticalalignment='center')
        axes[3].set_title('Dataset Statistics')
        axes[3].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    def save_all_plots(self, output_dir: str = "vpr_plots"):
        """
        Save all visualization plots to a directory.
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # UTM coordinates plot
        fig, _ = self.plot_utm_coordinates()
        fig.savefig(os.path.join(output_dir, 'utm_coordinates.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Distance distribution
        fig, _ = self.plot_distance_distribution()
        fig.savefig(os.path.join(output_dir, 'distance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Performance metrics
        fig, _ = self.plot_performance_metrics()
        fig.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Query visualizations for all queries with reference images
        queries_created = 0
        for i in range(len(self.results)):
            if self._has_reference_images(i):
                try:
                    fig, _ = self.visualize_query_matches(i, k=3)
                    fig.savefig(os.path.join(output_dir, f'query_{i}_matches.png'), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    queries_created += 1
                    print(f"Created visualization for query {i}")
                except Exception as e:
                    print(f"Warning: Could not create visualization for query {i}: {e}")
            else:
                print(f"Skipping query {i} - no reference images found")
        
        print(f"All plots saved to {output_dir}/")

