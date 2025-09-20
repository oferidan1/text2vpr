import os
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from typing import List, Tuple, Dict
import time
import asyncio
import concurrent.futures
from .config import CLIPConfig, CONFIG_PRESETS

class CLIPRetriever:
    def __init__(self, config: CLIPConfig = None, model_name: str = None, device: str = None):
        """
        Initialize CLIP retriever
        
        Args:
            config: Configuration object (if None, uses default)
            model_name: CLIP model variant to use (overrides config if provided)
            device: Device to run inference on (overrides config if provided)
        """
        # Use provided config or default
        if config is None:
            config = CLIPConfig()
        
        # Override config with direct parameters if provided
        if model_name is not None:
            config.model_name = model_name
        if device is not None:
            config.device = device
        
        # Set device
        if config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        
        # Store configuration
        self.config = config
        
        if config.verbose:
            print(f"Loading CLIP model: {config.model_name}")
        
        self.model, self.preprocess = clip.load(config.model_name, device=self.device)
        
        if config.verbose:
            print(f"CLIP model loaded on {self.device}")
            print(f"Similarity metric: {config.similarity_metric}")
            print(f"Normalize features: {config.normalize_features}")
        
        # Store image features and metadata
        self.image_features = None
        self.image_paths = []
        self.image_metadata = []
        
        # Store search history if enabled
        self.search_history = []
        
        # Get similarity function
        self.similarity_function = config.get_similarity_function()
        
        # GPU performance monitoring
        self.gpu_stats = {
            'total_batches_processed': 0,
            'total_images_processed': 0,
            'total_processing_time': 0.0,
            'gpu_memory_peak': 0.0
        }
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text prompt using CLIP text encoder"""
        # Truncate text if it's too long for CLIP's context length (77 tokens)
        # CLIP uses a simple word-based tokenization, so we approximate by character count
        # A rough estimate: 77 tokens ≈ 77 * 4 characters = ~308 characters
        max_chars = 300  # Conservative estimate to stay under 77 tokens
        
        if len(text) > max_chars:
            if self.config.verbose:
                print(f"⚠️  Text truncated from {len(text)} to {max_chars} characters")
            text = text[:max_chars].rsplit(' ', 1)[0]  # Truncate at word boundary
        
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features
        except Exception as e:
            if "too long for context length" in str(e):
                # If still too long, truncate more aggressively
                text = text[:200].rsplit(' ', 1)[0]
                if self.config.verbose:
                    print(f"⚠️  Text further truncated to {len(text)} characters due to token limit")
                text_tokens = clip.tokenize([text]).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features
            else:
                raise e
    
    def is_text_too_long(self, text: str) -> bool:
        """Check if text is too long for CLIP processing"""
        max_chars = 300  # Conservative estimate to stay under 77 tokens
        return len(text) > max_chars
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode single image using CLIP image encoder"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def encode_images_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode a batch of images using CLIP image encoder (GPU optimized)"""
        if not images:
            return None
        
        start_time = time.time()
        
        # Preprocess all images at once
        image_inputs = []
        for image in images:
            try:
                processed = self.preprocess(image)
                image_inputs.append(processed)
            except Exception as e:
                print(f"Error preprocessing image: {e}")
                # Add a dummy tensor to maintain batch structure
                dummy_image = Image.new('RGB', (224, 224), color='black')
                image_inputs.append(self.preprocess(dummy_image))
        
        if not image_inputs:
            return None
            
        # Stack and move to device
        batch_tensor = torch.stack(image_inputs).to(self.device)
        
        # Monitor GPU memory before processing
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        with torch.no_grad():
            # Process entire batch on GPU
            image_features = self.model.encode_image(batch_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Monitor GPU memory after processing
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_used = memory_after - memory_before
            self.gpu_stats['gpu_memory_peak'] = max(self.gpu_stats['gpu_memory_peak'], memory_after)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.gpu_stats['total_batches_processed'] += 1
        self.gpu_stats['total_images_processed'] += len(images)
        self.gpu_stats['total_processing_time'] += processing_time
        
        # Clear GPU cache periodically to prevent memory buildup
        if self.gpu_stats['total_batches_processed'] % 10 == 0 and self.device != "cpu":
            torch.cuda.empty_cache()
        
        return image_features
    
    def _get_optimal_batch_size(self, dataset_size: int = None) -> int:
        """Determine optimal batch size based on GPU memory and dataset size"""
        if self.device == "cpu":
            return 8  # Conservative for CPU
        
        # Get GPU memory info
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            # Base batch size on GPU memory
            if gpu_memory_gb >= 24:  # RTX 4090, A100, etc.
                base_batch_size = 64
            elif gpu_memory_gb >= 16:  # RTX 4080, RTX 3080 Ti, etc.
                base_batch_size = 32
            elif gpu_memory_gb >= 12:  # RTX 4070 Ti, RTX 3080, etc.
                base_batch_size = 16
            elif gpu_memory_gb >= 8:   # RTX 4060 Ti, RTX 3070, etc.
                base_batch_size = 8
            else:  # Lower-end GPUs
                base_batch_size = 4
            
            # For very large datasets (>1M images), increase batch size if memory allows
            if dataset_size and dataset_size > 1000000:
                if gpu_memory_gb >= 24:  # RTX 3090, RTX 4090, A100
                    return min(128, base_batch_size * 2)  # Up to 128 for large datasets
                elif gpu_memory_gb >= 16:
                    return min(64, base_batch_size * 2)   # Up to 64 for large datasets
                elif gpu_memory_gb >= 12:
                    return min(32, base_batch_size * 2)   # Up to 32 for large datasets
            
            return base_batch_size
        else:
            return 8  # Fallback for CPU
    
    def _process_image_batch(self, image_files: List[Path], use_async: bool = True) -> Tuple[torch.Tensor, List[Dict]]:
        """Process a batch of image files and return features and metadata"""
        if use_async:
            # Use asynchronous loading for better I/O performance
            images, metadata = self._load_images_batch_async(image_files)
        else:
            # Fallback to synchronous loading
            images = []
            metadata = []
            for img_path in image_files:
                try:
                    image = Image.open(img_path).convert('RGB')
                    images.append(image)
                    metadata.append({
                        'path': str(img_path),
                        'filename': img_path.name,
                        'size': image.size,
                        'mode': image.mode
                    })
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        if not images:
            return None, []
        
        # Encode batch on GPU
        try:
            batch_features = self.encode_images_batch(images)
            if batch_features is None:
                return None, []
            
            return batch_features, metadata
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            return None, []
    
    def _print_performance_stats(self):
        """Print GPU performance statistics"""
        if self.gpu_stats['total_batches_processed'] == 0:
            return
            
        print(f"\n🚀 GPU Performance Statistics:")
        print(f"   Device: {self.device}")
        print(f"   Total batches processed: {self.gpu_stats['total_batches_processed']}")
        print(f"   Total images processed: {self.gpu_stats['total_images_processed']}")
        print(f"   Total processing time: {self.gpu_stats['total_processing_time']:.2f}s")
        
        if self.gpu_stats['total_images_processed'] > 0:
            avg_time_per_image = self.gpu_stats['total_processing_time'] / self.gpu_stats['total_images_processed']
            images_per_second = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0
            print(f"   Average time per image: {avg_time_per_image:.3f}s")
            print(f"   Images per second: {images_per_second:.1f}")
        
        if self.device != "cpu" and torch.cuda.is_available():
            print(f"   Peak GPU memory usage: {self.gpu_stats['gpu_memory_peak']:.2f} GB")
            current_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"   Current GPU memory: {current_memory:.2f} GB")
    
    def get_gpu_stats(self) -> Dict:
        """Get current GPU performance statistics"""
        stats = self.gpu_stats.copy()
        if self.device != "cpu" and torch.cuda.is_available():
            stats['current_gpu_memory'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)
        return stats
    
    def _load_image_async(self, img_path: Path) -> Tuple[Image.Image, Dict]:
        """Load and preprocess a single image asynchronously"""
        try:
            image = Image.open(img_path).convert('RGB')
            metadata = {
                'path': str(img_path),
                'filename': img_path.name,
                'size': image.size,
                'mode': image.mode
            }
            return image, metadata
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None, None
    
    def _load_images_batch_async(self, image_files: List[Path], max_workers: int = 4) -> Tuple[List[Image.Image], List[Dict]]:
        """Load a batch of images asynchronously using thread pool"""
        images = []
        metadata = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all image loading tasks
            future_to_path = {
                executor.submit(self._load_image_async, img_path): img_path 
                for img_path in image_files
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    image, meta = future.result()
                    if image is not None and meta is not None:
                        images.append(image)
                        metadata.append(meta)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        return images, metadata
    
    def build_image_database(self, image_dir: str, supported_formats: List[str] = None, batch_size: int = None, save_progress: bool = False, progress_interval: int = 1000) -> None:
        """
        Build database of image features from directory
        
        Args:
            image_dir: Directory containing images
            supported_formats: List of supported image formats (default: common formats)
            batch_size: Batch size for GPU processing (auto-determined if None)
            save_progress: Save progress periodically for large datasets
            progress_interval: Save progress every N batches (default: 1000)
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            
        image_dir = Path(image_dir)
        image_files = []
        
        # Find all image files (case-insensitive)
        for ext in supported_formats:
            image_files.extend(image_dir.rglob(f"*{ext}"))
            image_files.extend(image_dir.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates (files found with both lowercase and uppercase extensions)
        image_files = list(set(image_files))
        
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")
            
        print(f"Found {len(image_files)} images. Building feature database...")
        
        # Determine optimal batch size for GPU
        if batch_size is None:
            batch_size = self._get_optimal_batch_size(len(image_files))
        
        print(f"Using batch size: {batch_size} for GPU processing")
        print(f"Processing {len(image_files)} images in {(len(image_files) + batch_size - 1) // batch_size} batches")
        
        # Process images in batches for GPU acceleration
        features_list = []
        for i in tqdm(range(0, len(image_files), batch_size), desc="Encoding images"):
            batch_files = image_files[i:i + batch_size]
            batch_features, batch_metadata = self._process_image_batch(batch_files)
            
            if batch_features is not None:
                features_list.append(batch_features)
                self.image_paths.extend([meta['path'] for meta in batch_metadata])
                self.image_metadata.extend(batch_metadata)
        
        if not features_list:
            raise ValueError("No images could be processed successfully")
            
        # Stack all features (they should all be on the same device)
        self.image_features = torch.cat(features_list, dim=0)
        print(f"Feature database built with {len(self.image_features)} images")
        
        # Display performance statistics
        self._print_performance_stats()
        
        # Automatically check for and remove duplicates
        self.remove_duplicate_features()
        
    def search(self, text_prompt: str, top_k: int = None) -> List[Dict]:
        """
        Search for images most similar to text prompt
        
        Args:
            text_prompt: Text description to search for
            top_k: Number of top results to return (uses config default if None)
            
        Returns:
            List of dictionaries with image info and similarity scores
        """
        if self.image_features is None:
            raise ValueError("Image database not built. Call build_image_database() first.")
        
        # Use config default if top_k not specified
        if top_k is None:
            top_k = self.config.default_top_k
        
        # Encode text prompt
        text_features = self.encode_text(text_prompt)
        
        # Use configurable similarity function
        similarities = self.similarity_function(text_features, self.image_features)
        
        # Apply similarity threshold if configured
        if self.config.min_similarity_threshold > 0:
            valid_indices = torch.where(similarities >= self.config.min_similarity_threshold)[0]
            if len(valid_indices) == 0:
                return []
            similarities = similarities[valid_indices]
            valid_features = self.image_features[valid_indices]
            valid_paths = [self.image_paths[i] for i in valid_indices]
            valid_metadata = [self.image_metadata[i] for i in valid_indices]
        else:
            valid_features = self.image_features
            valid_paths = self.image_paths
            valid_metadata = self.image_metadata
        
        # Get top-k indices
        top_indices = torch.argsort(similarities, descending=True)[:top_k]
        
        # Prepare results
        results = []
        for rank, idx in enumerate(top_indices):
            idx = idx.item()
            result = {
                'rank': rank + 1,
                'image_path': valid_paths[idx],
                'filename': valid_metadata[idx]['filename'],
                'similarity_score': similarities[idx].item(),
                'metadata': valid_metadata[idx]
            }
            results.append(result)
        
        # Store search history if enabled
        if self.config.save_search_history:
            search_record = {
                'timestamp': time.time(),
                'query': text_prompt,
                'top_k': top_k,
                'results_count': len(results),
                'similarity_metric': self.config.similarity_metric,
                'min_threshold': self.config.min_similarity_threshold
            }
            self.search_history.append(search_record)
        
        return results
    
    def save_database(self, save_path: str) -> None:
        """Save the image database to disk"""
        if self.image_features is None:
            raise ValueError("No database to save")
            
        save_data = {
            'image_features': self.image_features.cpu().numpy(),
            'image_paths': self.image_paths,
            'image_metadata': self.image_metadata
        }
        
        np.savez_compressed(save_path, **save_data)
        print(f"Database saved to {save_path}")
        
    def load_database(self, load_path: str) -> None:
        """Load image database from disk"""
        data = np.load(load_path, allow_pickle=True)
        
        self.image_features = torch.from_numpy(data['image_features']).to(self.device)
        self.image_paths = data['image_paths'].tolist()
        self.image_metadata = data['image_metadata'].tolist()
        
        print(f"Database loaded from {load_path} with {len(self.image_features)} images")
        
        # Check for duplicates in loaded database
        self.remove_duplicate_features()

    def remove_duplicate_features(self):
        """Remove duplicate features from the database"""
        if self.image_features is None:
            return
        
        print("🔍 Checking for duplicate features...")
        
        # Find unique features
        unique_features, inverse_indices, counts = torch.unique(
            self.image_features, dim=0, return_inverse=True, return_counts=True
        )
        
        if len(unique_features) == len(self.image_features):
            print("✅ No duplicate features found")
            return
        
        print(f"⚠️  Found {len(self.image_features) - len(unique_features)} duplicate features")
        print(f"   Removing duplicates...")
        
        # Keep only unique features
        self.image_features = unique_features
        
        # Update paths and metadata to match
        unique_paths = []
        unique_metadata = []
        
        for i in range(len(unique_features)):
            # Find the first occurrence of this feature
            first_occurrence = torch.where(inverse_indices == i)[0][0]
            unique_paths.append(self.image_paths[first_occurrence])
            unique_metadata.append(self.image_metadata[first_occurrence])
        
        self.image_paths = unique_paths
        self.image_metadata = unique_metadata
        
        print(f"✅ Database cleaned: {len(self.image_features)} unique images")
    
    def check_database_integrity(self):
        """Check database for duplicates and inconsistencies"""
        if self.image_features is None:
            print("❌ No database loaded")
            return False
        
        print("🔍 Checking database integrity...")
        
        # Check for duplicate features
        unique_features, inverse_indices, counts = torch.unique(
            self.image_features, dim=0, return_inverse=True, return_counts=True
        )
        
        duplicate_count = len(self.image_features) - len(unique_features)
        if duplicate_count > 0:
            print(f"⚠️  Found {duplicate_count} duplicate features")
            
            # Show which features are duplicated
            duplicate_indices = torch.where(counts > 1)[0]
            for idx in duplicate_indices:
                duplicate_positions = torch.where(inverse_indices == idx)[0]
                print(f"   Feature {idx} appears {counts[idx]} times at positions: {duplicate_positions}")
                for pos in duplicate_positions:
                    print(f"     Position {pos}: {self.image_paths[pos]}")
            
            return False
        else:
            print("✅ No duplicate features found")
        
        # Check for path duplicates
        unique_paths = set(self.image_paths)
        if len(unique_paths) != len(self.image_paths):
            print(f"⚠️  Found {len(self.image_paths) - len(unique_paths)} duplicate paths")
            return False
        else:
            print("✅ No duplicate paths found")
        
        # Check for metadata consistency
        if len(self.image_features) != len(self.image_paths) or len(self.image_features) != len(self.image_metadata):
            print("⚠️  Inconsistent database sizes")
            print(f"   Features: {len(self.image_features)}")
            print(f"   Paths: {len(self.image_paths)}")
            print(f"   Metadata: {len(self.image_metadata)}")
            return False
        else:
            print("✅ Database sizes consistent")
        
        print("✅ Database integrity check passed")
        return True

def main():
    parser = argparse.ArgumentParser(description="CLIP-based Image Retrieval")
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--text_prompt", type=str, help="Text prompt to search for")
    parser.add_argument("--top_k", type=int, help="Number of top results to return")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file path")
    parser.add_argument("--preset", type=str, choices=list(CONFIG_PRESETS.keys()), help="Use predefined configuration preset")
    parser.add_argument("--save_db", type=str, default=None, help="Path to save database")
    parser.add_argument("--load_db", type=str, default=None, help="Path to load existing database")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--check_integrity", action="store_true", help="Check database integrity for duplicates")
    parser.add_argument("--remove_duplicates", action="store_true", help="Remove duplicate features from database")
    parser.add_argument("--create_config", type=str, help="Create a new configuration file")
    parser.add_argument("--list_presets", action="store_true", help="List available configuration presets")
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_presets:
        print("Available configuration presets:")
        for name, preset_func in CONFIG_PRESETS.items():
            config = preset_func()
            print(f"  {name}: {config.similarity_metric} similarity, {config.model_name} model")
        return
    
    if args.create_config:
        config = CLIPConfig()
        config.save_to_file(args.create_config)
        print(f"Configuration file created: {args.create_config}")
        return
    
    # Load configuration
    try:
        if args.preset:
            config = CONFIG_PRESETS[args.preset]()
        elif os.path.exists(args.config):
            config = CLIPConfig.from_file(args.config)
        else:
            config = CLIPConfig()
            print(f"Config file not found: {args.config}, using defaults")
    except Exception as e:
        print(f"Error loading config: {e}, using defaults")
        config = CLIPConfig()
    
    # Override config with command line arguments
    if args.top_k is not None:
        config.default_top_k = args.top_k
    
    # Initialize retriever
    retriever = CLIPRetriever(config=config)
    
    # Load or build database
    if args.load_db:
        retriever.load_database(args.load_db)
    elif args.image_dir:
        retriever.build_image_database(args.image_dir)
        if args.save_db:
            retriever.save_database(args.save_db)
    else:
        print("❌ Error: Must specify either --image_dir or --load_db")
        return
    
    # Check integrity and remove duplicates if requested
    if args.check_integrity:
        retriever.check_database_integrity()
        return
    
    if args.remove_duplicates:
        retriever.remove_duplicate_features()
        if args.save_db:
            retriever.save_database(args.save_db)
        return
    
    # Check if search is requested
    if not args.text_prompt:
        print("ℹ️  No text prompt provided. Use --text_prompt to search or --check_integrity/--remove_duplicates for database operations.")
        return
    
    # Perform search
    print(f"\nSearching for: '{args.text_prompt}'")
    print(f"Using {config.similarity_metric} similarity")
    print(f"Returning top {config.default_top_k} results...")
    
    start_time = time.time()
    results = retriever.search(args.text_prompt, top_k=args.top_k)
    search_time = time.time() - start_time
    
    # Display results
    print(f"\nSearch completed in {search_time:.2f} seconds")
    print(f"\nTop {args.top_k} results:")
    print("-" * 80)
    
    for result in results:
        print(f"{result['rank']}. {result['filename']}")
        print(f"   Path: {result['image_path']}")
        print(f"   Similarity: {result['similarity_score']:.4f}")
        print(f"   Size: {result['metadata']['size']}")
        print()
    
    # Save results to file if requested
    if args.output_file:
        output_data = {
            'query': args.text_prompt,
            'top_k': args.top_k,
            'search_time': search_time,
            'results': results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
