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
        if 'mixvpr' in self.vision_model_name:           
            self.vision_encoder = vpr_models.get_model('mixvpr', 'ResNet50', 512)
            self.vision_encoder = self.vision_encoder.eval().to(args.device)
        if 'bge' in self.text_model_name:
            self.text_encoder = SentenceTransformer(self.text_model_name)
            
    def encode(self, images, texts):
        with torch.no_grad():
            image_features = self.vision_encoder(images)
            text_features = self.text_encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
        return image_features, text_features

        
