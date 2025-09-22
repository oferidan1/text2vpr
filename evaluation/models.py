import torch
import numpy as np
from transformers import BlipProcessor, BlipModel
from blip_model import BlipForImageTextRetrievalVision, BlipForImageTextRetrievalText

class VLM_Model:
    def __init__(self):
        self.model_name = 'Salesforce/blip-itm-base-coco'
        #self.model_name = 'Salesforce/blip-image-captioning-base'
        #self.model = BlipModel.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
        self.visual_model = BlipForImageTextRetrievalVision.from_pretrained(self.model_name)
        self.text_model = BlipForImageTextRetrievalText.from_pretrained(self.model_name)
        self.processor = BlipProcessor.from_pretrained(self.model_name)       
        self.prompt = "a photo of a "
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        # self.model.eval()
        self.visual_model.to(self.device)
        self.visual_model.eval()
        self.text_model.to(self.device)
        self.text_model.eval()
        
    def encode_images(self, images):
        with torch.no_grad():
            #image_features = self.model.get_image_features(images)
            image_features = self.visual_model(images)
        return image_features.cpu().float().numpy()
    
    def encode_texts(self, texts):
        #texts = [self.prompt[0] + text for text in texts]
        #inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            #text_features = self.model.get_text_features(texts)
            text_features = self.text_model(texts)
        return text_features.cpu().float().numpy()
    
    def get_processor(self):
        return self.processor   
        
