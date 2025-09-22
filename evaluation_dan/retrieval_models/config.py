#!/usr/bin/env python3
"""
Configuration system for CLIP-based image retrieval
"""

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
import json
import os

@dataclass
class CLIPConfig:
    """Configuration for CLIP retriever"""
    
    # Model settings
    model_name: str = "ViT-B/32"
    device: Optional[str] = None  # None = auto-detect
    model_type: Literal["clip", "blip"] = "clip"  # Model type selection
    blip_model_name: Optional[str] = None  # BLIP model name (auto-selected if None)
    
    # Database settings
    supported_formats: list = None  # None = default formats
    auto_remove_duplicates: bool = True
    save_metadata: bool = True
    
    # Similarity settings
    similarity_metric: Literal["cosine", "dot_product", "euclidean", "manhattan"] = "cosine"
    normalize_features: bool = True
    similarity_params: Dict[str, Any] = None  # Additional parameters for similarity
    
    # Search settings
    default_top_k: int = 5
    min_similarity_threshold: float = 0.0  # Filter results below this threshold
    
    # Processing settings
    batch_size: int = 1
    use_tqdm: bool = True
    verbose: bool = True
    
    # Output settings
    save_search_history: bool = False
    output_format: Literal["detailed", "simple", "json"] = "detailed"
    
    def __post_init__(self):
        """Set default values after initialization"""
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        if self.similarity_params is None:
            self.similarity_params = {}
    
    @classmethod
    def from_file(cls, config_path: str) -> 'CLIPConfig':
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = {
            'model_name': self.model_name,
            'device': self.device,
            'model_type': self.model_type,
            'blip_model_name': self.blip_model_name,
            'supported_formats': self.supported_formats,
            'auto_remove_duplicates': self.auto_remove_duplicates,
            'save_metadata': self.save_metadata,
            'similarity_metric': self.similarity_metric,
            'normalize_features': self.normalize_features,
            'similarity_params': self.similarity_params,
            'default_top_k': self.default_top_k,
            'min_similarity_threshold': self.min_similarity_threshold,
            'batch_size': self.batch_size,
            'use_tqdm': self.use_tqdm,
            'verbose': self.verbose,
            'save_search_history': self.save_search_history,
            'output_format': self.output_format
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_similarity_function(self):
        """Get the appropriate similarity function based on configuration"""
        if self.similarity_metric == "cosine":
            return self._cosine_similarity
        elif self.similarity_metric == "dot_product":
            return self._dot_product_similarity
        elif self.similarity_metric == "euclidean":
            return self._euclidean_similarity
        elif self.similarity_metric == "manhattan":
            return self._manhattan_similarity
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def _cosine_similarity(self, query_features, database_features):
        """Calculate cosine similarity between query and database features"""
        import torch
        
        # Ensure both tensors are on the same device
        device = query_features.device
        database_features = database_features.to(device)
        
        if self.normalize_features:
            # Features should already be normalized, but ensure it
            query_norm = query_features / (query_features.norm(dim=-1, keepdim=True) + 1e-8)
            db_norm = database_features / (database_features.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            query_norm = query_features
            db_norm = database_features
        
        # Cosine similarity is dot product of normalized vectors
        similarities = torch.mm(query_norm, db_norm.T)
        
        # Handle different tensor shapes safely
        if similarities.dim() > 1:
            similarities = similarities.squeeze()
        
        # Ensure we return a 1D tensor
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)
            
        return similarities
    
    def _dot_product_similarity(self, query_features, database_features):
        """Calculate dot product similarity"""
        import torch
        
        # Ensure both tensors are on the same device
        device = query_features.device
        database_features = database_features.to(device)
        
        if self.normalize_features:
            # Normalize features for dot product
            query_norm = query_features / (query_features.norm(dim=-1, keepdim=True) + 1e-8)
            db_norm = database_features / (database_features.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            query_norm = query_features
            db_norm = database_features
        
        similarities = torch.mm(query_norm, db_norm.T)
        
        # Handle different tensor shapes safely
        if similarities.dim() > 1:
            similarities = similarities.squeeze()
        
        # Ensure we return a 1D tensor
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)
            
        return similarities
    
    def _euclidean_similarity(self, query_features, database_features):
        """Calculate negative Euclidean distance (higher = more similar)"""
        import torch
        
        # Ensure both tensors are on the same device
        device = query_features.device
        database_features = database_features.to(device)
        
        # Calculate squared Euclidean distance
        diff = query_features.unsqueeze(1) - database_features.unsqueeze(0)
        distances = torch.sum(diff ** 2, dim=-1)
        
        # Handle different tensor shapes safely
        if distances.dim() > 1:
            distances = distances.squeeze()
        
        # Ensure we return a 1D tensor
        if distances.dim() == 0:
            distances = distances.unsqueeze(0)
        
        # Convert to similarity (negative distance, higher = more similar)
        similarities = -distances
        return similarities
    
    def _manhattan_similarity(self, query_features, database_features):
        """Calculate negative Manhattan distance (higher = more similar)"""
        import torch
        
        # Ensure both tensors are on the same device
        device = query_features.device
        database_features = database_features.to(device)
        
        # Calculate Manhattan distance
        diff = query_features.unsqueeze(1) - database_features.unsqueeze(0)
        distances = torch.sum(torch.abs(diff), dim=-1)
        
        # Handle different tensor shapes safely
        if distances.dim() > 1:
            distances = distances.squeeze()
        
        # Ensure we return a 1D tensor
        if distances.dim() == 0:
            distances = distances.unsqueeze(0)
        
        # Convert to similarity (negative distance, higher = more similar)
        similarities = -distances
        return similarities

# Predefined configurations
def get_default_config() -> CLIPConfig:
    """Get default configuration for general use"""
    return CLIPConfig()

def get_vpr_config() -> CLIPConfig:
    """Get configuration optimized for Visual Place Recognition"""
    return CLIPConfig(
        model_name="ViT-B/32",
        similarity_metric="cosine",
        normalize_features=True,
        default_top_k=10,
        min_similarity_threshold=0.1,
        verbose=True
    )

def get_fast_config() -> CLIPConfig:
    """Get configuration optimized for speed"""
    return CLIPConfig(
        model_name="ViT-B/32",
        similarity_metric="dot_product",
        normalize_features=False,
        batch_size=4,
        verbose=False,
        use_tqdm=False
    )

def get_high_accuracy_config() -> CLIPConfig:
    """Get configuration optimized for high accuracy"""
    return CLIPConfig(
        model_name="ViT-L/14",
        similarity_metric="cosine",
        normalize_features=True,
        default_top_k=20,
        min_similarity_threshold=0.05,
        verbose=True
    )

def get_blip_config() -> CLIPConfig:
    """Get configuration optimized for BLIP with long text support"""
    return CLIPConfig(
        model_type="blip",
        blip_model_name="Salesforce/blip-image-captioning-base",
        similarity_metric="cosine",
        normalize_features=True,
        default_top_k=10,
        min_similarity_threshold=0.1,
        verbose=True
    )

def get_blip_large_config() -> CLIPConfig:
    """Get configuration for larger BLIP model with better performance"""
    return CLIPConfig(
        model_type="blip",
        blip_model_name="Salesforce/blip-image-captioning-large",
        similarity_metric="cosine",
        normalize_features=True,
        default_top_k=15,
        min_similarity_threshold=0.05,
        verbose=True
    )

# Configuration presets
CONFIG_PRESETS = {
    "default": get_default_config,
    "vpr": get_vpr_config,
    "fast": get_fast_config,
    "high_accuracy": get_high_accuracy_config,
    "blip": get_blip_config,
    "blip_large": get_blip_large_config
}

