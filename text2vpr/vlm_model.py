from xml.parsers.expat import model
import torch
import numpy as np
#add parent directory to path
import os
import sys
from pathlib import Path
import peft
from sentence_transformers import SentenceTransformer
import vpr_models
from vpr_text import VPR_Text_Model
from transformers import AutoTokenizer, AutoModel
from blip_model import BlipForImageTextRetrievalWrapper
from transformers import BlipProcessor, BlipModel


class VLM_Model:
    def __init__(self, args):
        self.model_name = args.model_name
        self.vpr_model_name = args.vpr_model_name
        self.text_model_name = args.text_model_name
        self.device = args.device
        self.text_encoder_dim = 1024
        self.vpr_encoder_dim = args.vpr_dim
        if 'blip' in self.vpr_model_name:            
            self.text_encoder_dim = args.vpr_dim
            
        if args.fusion_type == 'text_adapter':
            args.is_text_pooling = 1
        
        if args.cross_modal==1:
            if 'blip' in self.vpr_model_name:
                self.vpr_encoder = BlipForImageTextRetrievalWrapper.from_pretrained(self.vpr_model_name)
                self.processor = BlipProcessor.from_pretrained(self.vpr_model_name)
                self.vpr_encoder = self.vpr_encoder.eval().to(args.device)
            self.encoder_dim = self.vpr_encoder_dim
            
        elif args.is_dual_encoder or args.encode_mode!='both':    
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)  
            self.text_encoder = AutoModel.from_pretrained(self.text_model_name).to(args.device)
            self.text_encoder.eval()
            
            self.vpr_encoder = vpr_models.get_model(args.vpr_model_name.lower(), args.vpr_model_backbone, self.vpr_encoder_dim)
            self.vpr_encoder = self.vpr_encoder.eval().to(args.device)
            
            self.encoder_dim = self.text_encoder_dim + self.vpr_encoder_dim
            if args.encode_mode == 'text':
                self.encoder_dim = self.text_encoder_dim 
            elif args.encode_mode == 'image':
                self.encoder_dim = self.vpr_encoder_dim
        else:            
            self.single_encoder = VPR_Text_Model(   
                #---- Encoder
                vpr_model_name=args.vpr_model_name,
                vpr_model_backbone=args.vpr_model_backbone,
                vpr_encoder_dim=args.vpr_dim,                
               
                fusion_type=args.fusion_type,
                is_encode_image=args.is_encode_image,
                is_encode_text=args.is_encode_text,
                is_trainable_text_encoder=args.is_trainable_text_encoder,
                embeds_dim=args.pca_dim,
                is_pca=args.is_pca,
                is_text_pooling=args.is_text_pooling,
                is_image_pooling=args.is_image_pooling,
            )

            if args.lora_path is not None:
                print("loading lora from:", args.lora_path)
                self.single_encoder.text_encoder = peft.PeftModel.from_pretrained(self.single_encoder.text_encoder, args.lora_path, is_trainable=False)            
            else:            
                model_state_dict = torch.load(args.model_name)['state_dict']
                self.single_encoder.load_state_dict(model_state_dict)
            
            # #not sure why, text encoder weight are bad
            # self.single_encoder.text_encoder = AutoModel.from_pretrained(args.text_model_name)            
            
            
            self.single_encoder = self.single_encoder.to(args.device)
            self.single_encoder.eval()
            
            # self.single_encoder.is_encode_image = 0
            # self.single_encoder.embeds_dim = 256
             
            self.encoder_dim = self.single_encoder.embeds_dim           
            
        
    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        # Sum of the attention mask
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9).unsqueeze(1)
        # Mean Pooling
        return sum_embeddings / sum_mask
            
    def encode_dual(self, images, texts):
        with torch.no_grad():
            image_features = self.vpr_encoder(images)
            text_features = self.encode_text(texts)       
        return image_features, text_features
    
    def encode_single(self, images, texts):
        with torch.no_grad():
            features, text_features, w, _, _ = self.single_encoder(images, texts)
        return features, text_features, w 
    
    def encode_image(self, images):
        if 'blip' in self.vpr_model_name:
            with torch.no_grad():
                image_features = self.vpr_encoder.encode_image(images)[:,0]
        else:
            with torch.no_grad():
                image_features = self.vpr_encoder(images)            
        return image_features
    
    def encode_text(self, texts):
        if 'blip' in self.vpr_model_name:            
            text_inputs = self.processor(text=texts, return_tensors="pt", padding=True).input_ids.to(self.device)
            with torch.no_grad():     
                text_features = self.vpr_encoder.encode_text(text_inputs)[:,0]
        else:
            text_tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():      
                model_output = self.text_encoder(**text_tokens)                        
                text_features = model_output[0][:, 0]            
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)   
        return text_features


        
