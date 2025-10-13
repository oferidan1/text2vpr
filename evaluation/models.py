import torch
import numpy as np
#add parent directory to path
import os
import sys
from pathlib import Path
import peft
from sentence_transformers import SentenceTransformer
import vpr_models

class VLM_Model:
    def __init__(self, args):
        self.model_name = args.model_name
        self.vision_model_name = args.vision_model_name
        self.text_model_name = args.text_model_name
        self.device = args.device
        if 'bge' in self.text_model_name:
            self.text_encoder_dim = 1024
            self.text_encoder = SentenceTransformer(self.text_model_name)
        if 'mixvpr' in self.vision_model_name:           
            self.vision_encoder_dim = args.descriptors_dimension
            self.vision_encoder = vpr_models.get_model('mixvpr', 'ResNet50', self.vision_encoder_dim)
            self.vision_encoder = self.vision_encoder.eval().to(args.device)
        self.encoder_dim = self.text_encoder_dim + self.vision_encoder_dim
        if args.encode_mode == 'text':
            self.encoder_dim = self.text_encoder_dim 
        elif args.encode_mode == 'image':
            self.encoder_dim = self.vision_encoder_dim
        
            
    def encode_dual(self, images, texts):
        with torch.no_grad():
            image_features = self.vision_encoder(images)
            text_features = self.text_encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
        return image_features, text_features
    
    def encode_single(self, images, texts):
        with torch.no_grad():
            image_features = self.vision_encoder(images)
            text_features = self.text_encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
        return image_features
    
    def encode_image(self, images):
        with torch.no_grad():
            image_features = self.vision_encoder(images)            
        return image_features
    
    def encode_text(self, texts):
        with torch.no_grad():            
            text_features = self.text_encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
        return text_features


        
