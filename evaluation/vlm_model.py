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
from mixvpr_text import VPRModel_text
from transformers import AutoTokenizer, AutoModel

class VLM_Model:
    def __init__(self, args):
        self.model_name = args.model_name
        self.vision_model_name = args.vision_model_name
        self.text_model_name = args.text_model_name
        self.device = args.device
        self.text_encoder_dim = 1024
        self.vision_encoder_dim = args.vision_dimension
        if args.is_dual_encoder or args.encode_mode!='both':                        
            #self.text_encoder = SentenceTransformer(self.text_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)  
            self.text_encoder = AutoModel.from_pretrained(self.text_model_name).to(args.device)
            self.text_encoder.eval()
            
            self.vision_encoder = vpr_models.get_model(args.vision_model_name, args.vision_model_backbone, self.vision_encoder_dim)
            self.vision_encoder = self.vision_encoder.eval().to(args.device)
            
            self.encoder_dim = self.text_encoder_dim + self.vision_encoder_dim
            if args.encode_mode == 'text':
                self.encoder_dim = self.text_encoder_dim 
            elif args.encode_mode == 'image':
                self.encoder_dim = self.vision_encoder_dim
        else:            
            self.single_encoder = VPRModel_text(   
                #---- Encoder
                backbone_arch=args.vision_model_backbone.lower(),
                pretrained=True,
                layers_to_freeze=2,
                layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
                agg_arch='MixVPR',
                agg_config={'in_channels' : 1024,
                        'in_h' : 20,
                        'in_w' : 20,
                        'out_channels' : args.vision_dimension//args.vpr_rows,
                        'mix_depth' : 4,
                        'mlp_ratio' : 1,
                        'out_rows' : args.vpr_rows}, # the output dim will be (out_rows * out_channels))
                fusion_type=args.fusion_type,
                is_encode_image=args.is_encode_image,
                is_encode_text=args.is_encode_text,
                is_trainable_text_encoder=args.is_trainable_text_encoder,
                 embeds_dim=args.vision_dimension,
            )
            
            model_state_dict = torch.load(args.model_name)['state_dict']
            self.single_encoder.load_state_dict(model_state_dict)
            
            #not sure why, text encoder weight are bad
            self.single_encoder.text_encoder = AutoModel.from_pretrained(args.text_model_name)            
            if  args.lora_path is not None:
                print("loading lora from:", args.lora_path)
                self.single_encoder.text_encoder = peft.PeftModel.from_pretrained(self.single_encoder.text_encoder, args.lora_path, is_trainable=False)            
            
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
            image_features = self.vision_encoder(images)
            text_features = self.encode_text(texts)       
        return image_features, text_features
    
    def encode_single(self, images, texts):
        with torch.no_grad():
            features, text_features, w = self.single_encoder(images, texts)
        return features, text_features, w 
    
    def encode_image(self, images):
        with torch.no_grad():
            image_features = self.vision_encoder(images)            
        return image_features
    
    def encode_text(self, texts):
        # with torch.no_grad():            
        #     text_features = self.text_encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
        text_tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():      
            model_output = self.text_encoder(**text_tokens)                        
            text_features = model_output[0][:, 0]
            #text_features = model_output.last_hidden_state[:, 0]
        #text_features = self.mean_pooling(model_output, text_tokens['attention_mask'])
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)   
        return text_features


        
