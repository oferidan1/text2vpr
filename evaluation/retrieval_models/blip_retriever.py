#!/usr/bin/env python3
"""
BLIP-based Image Retrieval System

This module provides BLIP-based image retrieval capabilities as an alternative to CLIP,
particularly useful for handling longer text descriptions that exceed CLIP's token limits.
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import concurrent.futures
from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipForConditionalGeneration
from .config import CLIPConfig

class BLIPRetriever:
    """BLIP-based image retrieval system for handling longer text descriptions"""
    
    def __init__(self, config: CLIPConfig = None, model_name: str = None, device: str = None):
        """
        Initialize BLIP retriever
        
        Args:
            config: Configuration object (if None, uses default)
            model_name: BLIP model variant to use (overrides config if provided)
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
        
        # Default BLIP model if not specified
        if not hasattr(config, 'blip_model_name') or config.blip_model_name is None:
            self.blip_model_name = "Salesforce/blip-image-captioning-base"
        else:
            self.blip_model_name = config.blip_model_name
        
        if config.verbose:
            print(f"Loading BLIP model: {self.blip_model_name}")
        
        # Load BLIP model and processor
        try:
            self.processor = BlipProcessor.from_pretrained(self.blip_model_name)
            self.model = BlipForImageTextRetrieval.from_pretrained(self.blip_model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
            print("Falling back to BLIP image captioning model...")
            try:
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model.to(self.device)
                self.model.eval()
                self.blip_model_name = "Salesforce/blip-image-captioning-base"
            except Exception as e2:
                raise RuntimeError(f"Failed to load any BLIP model: {e2}")
        
        if config.verbose:
            print(f"BLIP model loaded on {self.device}")
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
        """Encode text prompt using BLIP text encoder"""
        # BLIP can handle much longer text than CLIP (up to 512 tokens typically)
        # We'll use a more generous limit for BLIP
        max_chars = 2000  # Much higher than CLIP's ~300 chars
        
        if len(text) > max_chars:
            if self.config.verbose:
                print(f"⚠️  Text truncated from {len(text)} to {max_chars} characters")
            text = text[:max_chars].rsplit(' ', 1)[0]  # Truncate at word boundary
        
        try:
            # Use BLIP processor to encode text
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # For BLIP image-text retrieval, we need to use the text encoder
                if hasattr(self.model, 'text_encoder'):
                    text_features = self.model.text_encoder(**inputs)
                    text_features = text_features.last_hidden_state.mean(dim=1)  # Pool over sequence length
                else:
                    # Fallback: use the model's forward pass with dummy image
                    # This is a workaround for models that don't expose text encoder directly
                    dummy_image = Image.new('RGB', (224, 224), color='black')
                    image_inputs = self.processor(images=dummy_image, return_tensors="pt")
                    image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                    
                    # Combine text and image inputs
                    combined_inputs = {**inputs, **image_inputs}
                    outputs = self.model(**combined_inputs)
                    
                    # Extract text features from the output
                    text_features = outputs.logits_per_text
                
                # Normalize features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features
            
        except Exception as e:
            if "too long" in str(e).lower():
                # If still too long, truncate more aggressively
                text = text[:1000].rsplit(' ', 1)[0]
                if self.config.verbose:
                    print(f"⚠️  Text further truncated to {len(text)} characters due to token limit")
                return self.encode_text(text)  # Recursive call with shorter text
            else:
                raise e
    
    def is_text_too_long(self, text: str) -> bool:
        """Check if text is too long for BLIP processing"""
        max_chars = 2000  # Much higher limit than CLIP
        return len(text) > max_chars
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode single image using BLIP image encoder"""
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if hasattr(self.model, 'vision_model'):
                    # Use vision model directly if available
                    image_features = self.model.vision_model(**inputs).last_hidden_state
                    image_features = image_features.mean(dim=1)  # Pool over spatial dimensions
                else:
                    # Fallback: use full model with dummy text
                    dummy_text = "a photo"
                    text_inputs = self.processor(text=dummy_text, return_tensors="pt")
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    
                    combined_inputs = {**inputs, **text_inputs}
                    outputs = self.model(**combined_inputs)
                    
                    # Extract image features from the output
                    image_features = outputs.logits_per_image
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features
            
        except Exception as e:
            print(f"Error encoding image: {e}")
            # Return zero features as fallback
            return torch.zeros(1, 768).to(self.device)  # BLIP typically uses 768-dim features
    
    def encode_images_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode a batch of images using BLIP image encoder (GPU optimized)"""
        if not images:
            return None
        
        start_time = time.time()
        
        # Process images in smaller batches to avoid memory issues
        batch_size = min(4, len(images))  # BLIP is more memory intensive than CLIP
        features_list = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            try:
                # Process batch
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    if hasattr(self.model, 'vision_model'):
                        # Use vision model directly if available
                        image_features = self.model.vision_model(**inputs).last_hidden_state
                        image_features = image_features.mean(dim=1)  # Pool over spatial dimensions
                    else:
                        # Fallback: use full model with dummy text
                        dummy_text = "a photo"
                        text_inputs = self.processor(text=[dummy_text] * len(batch_images), return_tensors="pt")
                        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                        
                        combined_inputs = {**inputs, **text_inputs}
                        outputs = self.model(**combined_inputs)
                        
                        # Extract image features from the output
                        image_features = outputs.logits_per_image
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                features_list.append(image_features)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add zero features for failed batch
                dummy_features = torch.zeros(len(batch_images), 768).to(self.device)
                features_list.append(dummy_features)
        
        if not features_list:
            return None
        
        # Concatenate all features
        batch_features = torch.cat(features_list, dim=0)
        
        # Monitor GPU memory
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated() / (1024**3)  # GB
            self.gpu_stats['gpu_memory_peak'] = max(self.gpu_stats['gpu_memory_peak'], memory_after)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.gpu_stats['total_batches_processed'] += 1
        self.gpu_stats['total_images_processed'] += len(images)
        self.gpu_stats['total_processing_time'] += processing_time
        
        # Clear GPU cache periodically
        if self.gpu_stats['total_batches_processed'] % 5 == 0 and self.device != "cpu":
            torch.cuda.empty_cache()
        
        return batch_features
    
    def _get_optimal_batch_size(self, dataset_size: int = None) -> int:
        """Determine optimal batch size based on GPU memory and dataset size"""
        if self.device == "cpu":
            return 2  # Conservative for CPU
        
        # BLIP is more memory intensive than CLIP, so use smaller batch sizes
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            # Smaller batch sizes for BLIP
            if gpu_memory_gb >= 24:  # RTX 4090, A100, etc.
                base_batch_size = 16
            elif gpu_memory_gb >= 16:  # RTX 4080, RTX 3080 Ti, etc.
                base_batch_size = 8
            elif gpu_memory_gb >= 12:  # RTX 4070 Ti, RTX 3080, etc.
                base_batch_size = 4
            elif gpu_memory_gb >= 8:   # RTX 4060 Ti, RTX 3070, etc.
                base_batch_size = 2
            else:  # Lower-end GPUs
                base_batch_size = 1
            
            return base_batch_size
        else:
            return 2  # Fallback for CPU
    
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
            
        print(f"Found {len(image_files)} images. Building feature database with BLIP...")
        
        # Determine optimal batch size for GPU
        if batch_size is None:
            batch_size = self._get_optimal_batch_size(len(image_files))
        
        print(f"Using batch size: {batch_size} for BLIP processing")
        print(f"Processing {len(image_files)} images in {(len(image_files) + batch_size - 1) // batch_size} batches")
        
        # Process images in batches for GPU acceleration
        features_list = []
        for i in tqdm(range(0, len(image_files), batch_size), desc="Encoding images with BLIP"):
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
        print(f"BLIP feature database built with {len(self.image_features)} images")
        
        # Display performance statistics
        self._print_performance_stats()
        
        # Automatically check for and remove duplicates if enabled in config
        if self.config.auto_remove_duplicates:
            self.remove_duplicate_features()
        else:
            print("⚠️  Duplicate removal disabled in configuration")
    
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
            'image_metadata': self.image_metadata,
            'model_type': 'blip',
            'blip_model_name': self.blip_model_name
        }
        
        np.savez_compressed(save_path, **save_data)
        print(f"BLIP database saved to {save_path}")
        
    def load_database(self, load_path: str) -> None:
        """Load image database from disk"""
        data = np.load(load_path, allow_pickle=True)
        
        self.image_features = torch.from_numpy(data['image_features']).to(self.device)
        self.image_paths = data['image_paths'].tolist()
        self.image_metadata = data['image_metadata'].tolist()
        
        # Check if this is a BLIP database
        if 'model_type' in data and data['model_type'] == 'blip':
            print(f"BLIP database loaded from {load_path} with {len(self.image_features)} images")
        else:
            print(f"Database loaded from {load_path} with {len(self.image_features)} images (compatibility mode)")
        
        # Check for duplicates in loaded database if enabled in config
        if self.config.auto_remove_duplicates:
            self.remove_duplicate_features()
        else:
            print("⚠️  Duplicate removal disabled in configuration")
    
    def remove_duplicate_features(self):
        """Remove duplicate features from the database using memory-efficient batch processing"""
        if self.image_features is None:
            return
        
        print("🔍 Checking for duplicate features...")
        
        # Get the device and determine batch size for memory efficiency
        device = self.image_features.device
        feature_dim = self.image_features.shape[1]
        total_images = len(self.image_features)
        
        # Estimate memory usage and determine appropriate batch size
        # Each feature is feature_dim * 4 bytes (float32)
        feature_size_mb = feature_dim * 4 / (1024 * 1024)
        
        # Conservative batch size to avoid OOM - process in chunks of 1000-5000 features
        max_batch_size = min(5000, total_images)
        if feature_size_mb > 1.0:  # If features are large, use smaller batches
            max_batch_size = min(1000, total_images)
        
        print(f"   Processing {total_images} features in batches of {max_batch_size}")
        
        # Use a more memory-efficient approach: process in batches and use hash-based deduplication
        unique_features_list = []
        unique_indices = []
        seen_hashes = set()
        
        # Process features in batches
        for i in tqdm(range(0, total_images, max_batch_size), desc="Removing duplicates"):
            batch_end = min(i + max_batch_size, total_images)
            batch_features = self.image_features[i:batch_end]
            
            # Convert batch to CPU for hashing to avoid GPU memory issues
            batch_features_cpu = batch_features.cpu()
            
            for j, feature in enumerate(batch_features_cpu):
                global_idx = i + j
                
                # Create a more robust hash of the feature for quick comparison
                # Use a combination of feature statistics for hashing
                feature_stats = torch.cat([
                    feature.mean().unsqueeze(0),
                    feature.std().unsqueeze(0),
                    feature.min().unsqueeze(0),
                    feature.max().unsqueeze(0)
                ])
                feature_hash = hash(tuple(feature_stats.round(decimals=4).tolist()))
                
                if feature_hash not in seen_hashes:
                    seen_hashes.add(feature_hash)
                    unique_features_list.append(batch_features[j].unsqueeze(0))
                    unique_indices.append(global_idx)
                else:
                    # For potential duplicates, do a more thorough comparison
                    is_duplicate = False
                    # Only check against the most recent features to save memory
                    check_limit = min(50, len(unique_features_list))
                    for existing_feature in unique_features_list[-check_limit:]:
                        if torch.allclose(batch_features[j], existing_feature.squeeze(), atol=1e-5):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        unique_features_list.append(batch_features[j].unsqueeze(0))
                        unique_indices.append(global_idx)
                
                # Clear CPU tensor to free memory
                del feature
                del feature_stats
            
            # Clear batch from GPU memory
            del batch_features
            del batch_features_cpu
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Reconstruct the unique features tensor
        if len(unique_features_list) == total_images:
            print("✅ No duplicate features found")
            return
        
        print(f"⚠️  Found {total_images - len(unique_features_list)} duplicate features")
        print(f"   Removing duplicates...")
        
        # Concatenate unique features
        unique_features = torch.cat(unique_features_list, dim=0)
        self.image_features = unique_features
        
        # Update paths and metadata to match unique features
        unique_paths = [self.image_paths[idx] for idx in unique_indices]
        unique_metadata = [self.image_metadata[idx] for idx in unique_indices]
        
        self.image_paths = unique_paths
        self.image_metadata = unique_metadata
        
        print(f"✅ Database cleaned: {len(self.image_features)} unique images")
    
    def _print_performance_stats(self):
        """Print GPU performance statistics"""
        if self.gpu_stats['total_batches_processed'] == 0:
            return
            
        print(f"\n🚀 BLIP GPU Performance Statistics:")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.blip_model_name}")
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
