import parser
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from vlm_model import VLM_Model 
import os
from test_dataset import TestDataset
import visualizations
from math import sqrt

def normlize_features(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)    

def encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha):
    if args.encode_mode == 'text':
        # single vector - text
        descriptors = model.encode_text(texts)
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors        
    elif args.encode_mode == 'image':
        # single vector - image
        descriptors = model.encode_image(images.to(args.device))
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors    
    elif args.is_dual_encoder:
        image_features, text_features = model.encode_dual(images.to(args.device), texts)
        # cat fusion: concat text and vision vectors
        if args.dual_encoder_fusion == 'cat':
            descriptors = torch.cat((image_features, text_features), dim=1)
            descriptors = descriptors.cpu().numpy()
            if args.is_normalize_features:
                descriptors = normlize_features(descriptors)
            all_descriptors[indices.numpy(), :] = descriptors
        else:
            # each fusion: save each modality
            image_features = image_features.cpu().numpy()
            vision_descriptors[indices.numpy(), :] = image_features
            text_features = text_features.cpu().numpy()
            text_descriptors[indices.numpy(), :] = text_features                    
    else:
        # single vector of both image and text
        descriptors, text_features, w = model.encode_single(images.to(args.device), texts)
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors
        if args.fusion_type == 'dynamic_weighting':
            vision_descriptors[indices.numpy(), :] = descriptors
            text_features = text_features.cpu().numpy()
            text_descriptors[indices.numpy(), :] = text_features  
            w = w.cpu().numpy()
            w_alpha[indices.numpy(), :] = w
            

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    start_time = datetime.now()
    
    args.is_dual_encoder = 1
    args.encode_mode = 'both'
    args.dual_encoder_fusion = 'each'
    # args.database_folder = "/mnt/d/data/gsv_cities/Images"
    # args.queries_folder = "/mnt/d/data/gsv_cities/Images"
    # args.queries_csv = "/mnt/d/data/gsv_cities/gsv_cities_predictions.csv"
    # args.image_root = "/mnt/d/data/gsv_cities"

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(f"Testing with {args.vpr_model_name}")
    logger.info(f"The outputs are being saved in {log_dir}")

    model = VLM_Model(args)

    test_ds = TestDataset(
        args.database_folder,   
        None,
        args.queries_csv,
        args.image_root,        
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=False,
        is_sample=True,
    )
    logger.info(f"Testing on {test_ds}")
    all_descriptors = None
    vision_descriptors = None
    text_descriptors = None    

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        #database_subset_ds = Subset(test_ds, list(range(100)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        
        vision_descriptors = np.empty((len(test_ds), model.vpr_encoder_dim), dtype="float32")
        text_descriptors = np.empty((len(test_ds), model.text_encoder_dim), dtype="float32")            
        all_descriptors = np.empty((len(test_ds), model.encoder_dim), dtype="float32")
        w_alpha = np.empty((len(test_ds), 2), dtype="float32")
            
        for images, indices, texts in tqdm(database_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha)            
        
        
        sim_matrix_t = np.matmul(text_descriptors, text_descriptors.T)
        #calc data mu and std over text sim matrix but without diagonal elements and lower triangle
        upper_indices = np.triu_indices_from(sim_matrix_t, k=0) 
        sim_matrix_t_upper = sim_matrix_t[upper_indices]
        mu_t  = np.mean(sim_matrix_t_upper)
        std_t = np.std(sim_matrix_t_upper)
        min_t = np.min(sim_matrix_t_upper)
        max_t = np.max(sim_matrix_t_upper)
        logger.info(f"Text Similarity matrix mean: {mu_t:.8f}")
        logger.info(f"Text Similarity matrix std: {std_t:.8f}")
        logger.info(f"Text Similarity matrix min: {min_t:.8f}")
        logger.info(f"Text Similarity matrix max: {max_t:.8f}")
        
        sim_t = (sim_matrix_t_upper-mu_t)/std_t
        
        logger.info(f"Text Similarity after mu matrix mean: {np.mean(sim_t):.8f}")
        logger.info(f"Text Similarity after mu matrix std: {np.std(sim_t):.8f}")
        logger.info(f"Text Similarity after mu matrix min: {np.min(sim_t):.8f}")
        logger.info(f"Text Similarity after mu matrix max: {np.max(sim_t):.8f}")
        
        min_t = np.min(sim_t)
        max_t = np.max(sim_t)
        #sim_t_new = ((sim_t-min_t)/(max_t-min_t))
        sim_t_new = ((sim_t-min_t)/(max_t-min_t))*2-1        
        
        logger.info(f"Text Similarity after min-max matrix mean: {np.mean(sim_t_new):.8f}")
        logger.info(f"Text Similarity after min-max matrix std: {np.std(sim_t_new):.8f}")
        logger.info(f"Text Similarity after min-max matrix min: {np.min(sim_t_new):.8f}")
        logger.info(f"Text Similarity after min-max matrix max: {np.max(sim_t_new):.8f}")
        

        sim_matrix_v = np.matmul(vision_descriptors, vision_descriptors.T)
        #calc data mu and std over vision sim matrix but without diagonal elements and lower triangle
        upper_indices = np.triu_indices_from(sim_matrix_v, k=1) 
        sim_matrix_v_upper = sim_matrix_v[upper_indices]
        mu_v  = np.mean(sim_matrix_v_upper)
        std_v = np.std(sim_matrix_v_upper)
        min_v = np.min(sim_matrix_v_upper)
        max_v = np.max(sim_matrix_v_upper)
        logger.info(f"Vision Similarity matrix mean: {mu_v:.8f}")
        logger.info(f"Vision Similarity matrix std: {std_v:.8f}")
        logger.info(f"Vision Similarity matrix min: {min_v:.8f}")
        logger.info(f"Vision Similarity matrix max: {max_v:.8f}")
        
        sim_v = (sim_matrix_v_upper-mu_v)/std_v
        
        logger.info(f"Vision Similarity after mu matrix mean: {np.mean(sim_v):.8f}")
        logger.info(f"Vision Similarity after mu matrix std: {np.std(sim_v):.8f}")
        logger.info(f"Vision Similarity after mu matrix min: {np.min(sim_v):.8f}")
        logger.info(f"Vision Similarity after mu matrix max: {np.max(sim_v):.8f}")
        
        min_v = np.min(sim_v)
        max_v = np.max(sim_v)
        #sim_v_new = ((sim_v-min_v)/(max_v-min_v))
        sim_v_new = ((sim_v-min_v)/(max_v-min_v))*2-1
        
        logger.info(f"Vision Similarity after min-max matrix mean: {np.mean(sim_v_new):.8f}")
        logger.info(f"Vision Similarity after min-max matrix std: {np.std(sim_v_new):.8f}")
        logger.info(f"Vision Similarity after min-max matrix min: {np.min(sim_v_new):.8f}")
        logger.info(f"Vision Similarity after min-max matrix max: {np.max(sim_v_new):.8f}")


if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
