import torch
import numpy as np
from transformers import BlipProcessor, BlipModel
from blip_model import BlipForImageTextRetrievalVision, BlipForImageTextRetrievalText, BlipForImageTextRetrievalWrapper
#add parent directory to path
import os
import sys
from pathlib import Path
from BLIP.models.blip_retrieval import BLIP_Retrieval
import peft

class VLM_Model:
    def __init__(self, model_name, lora_path=None):
        self.model_name = model_name
        # self.visual_model = BlipForImageTextRetrievalVision.from_pretrained(self.model_name)
        # self.text_model = BlipForImageTextRetrievalText.from_pretrained(self.model_name)
        #self.model = BlipForImageTextRetrievalWrapper.from_pretrained(self.model_name)
        self.model = BLIP_Retrieval(pretrained=self.model_name)
        if lora_path is not None:
            print(f"apply LoRA weights from {lora_path}")
            self.model = peft.PeftModel.from_pretrained(self.model, lora_path)
            # self.visual_model = peft.PeftModel.from_pretrained(self.visual_model, lora_path)
            # self.text_model = peft.PeftModel.from_pretrained(self.text_model, lora_path)
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.prompt = "a photo of a "
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        # self.visual_model.to(self.device).eval()
        # self.text_model.to(self.device).eval()

        
    def encode_images(self, images):
        with torch.no_grad():
            #image_features = self.model.get_image_features(images)
            #image_features = self.visual_model(images)
            image_features = self.model.encode_image(images)
        return image_features.cpu().float().numpy()
    
    def encode_texts(self, texts):
        #texts = [self.prompt[0] + text for text in texts]
        #inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            #text_features = self.model.get_text_features(texts)
            #text_features = self.text_model(texts)
            #text_features = self.model.encode_text(texts)
            text_features = self.model.encode_text_tokens(texts)
        return text_features.cpu().float().numpy()
    
    def get_processor(self):
        return self.processor   
        
