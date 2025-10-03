import torch
import numpy as np
from transformers import BlipProcessor, BlipModel
#add parent directory to path
import os
import sys
from pathlib import Path
from models.blip_retrieval import blip_retrieval
import peft
import ruamel.yaml as yaml

class VLM_Model:
    def __init__(self, args):
        config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        self.model_name = args.model_name
        self.model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
        if args.lora_path is not None:
            print(f"apply LoRA weights from {args.lora_path}")
            self.model = peft.PeftModel.from_pretrained(self.model, args.lora_path)
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.prompt = "a photo of a "
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        
    def encode_images(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        return image_features.cpu().float().numpy()
    
    def encode_texts(self, texts):
        with torch.no_grad():
            text_features = self.model.encode_text_tokens(texts)
        return text_features.cpu().float().numpy()
    
    def get_processor(self):
        return self.processor   
        
