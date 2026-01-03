import argparse
import parser
from argparse import Namespace
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
from dataloaders.MapillaryTestDataset import MSLSTest
import visualizations
from math import sqrt
from sklearn.decomposition import PCA
from scipy.stats import norm
from typing import Tuple, List
from scipy.interpolate import interp1d
from scipy.stats import ecdf
import pandas as pd
import cv2
import kornia as K
import kornia.feature as KF

def get_alpha_vision_batches(matcher, query_path, database_paths, preds, device='cuda'):
    # 1. Helper to load and preprocess a single image path to a tensor
    def load_image_tensor(path):
        # Load as RGB32 (0-1 float)
        img = K.io.load_image(path, K.io.ImageLoadType.RGB32, device=device)[None, ...]
        # Resize to your desired dimensions
        img = K.geometry.resize(img, (600, 375), antialias=True)
        # Convert to grayscale
        return K.color.rgb_to_grayscale(img)

    # 2. Prepare Query Tensor [1, 1, 600, 375]
    # We load it once and move to device
    query_tensor = load_image_tensor(query_path)
    
    # 3. Prepare Database Batch [B, 1, 600, 375]
    # We load all paths in 'database_paths' and stack them
    db_tensors = torch.cat([load_image_tensor(p) for p in database_paths], dim=0)
    
    batch_size = db_tensors.shape[0]

    # Efficiently expand query to match the database batch size
    query_expanded = query_tensor.expand(batch_size, -1, -1, -1)

    input_dict = {
        "image0": query_expanded.bfloat16(), 
        "image1": db_tensors.bfloat16()
    }

    # 4. Inference on batch
    with torch.inference_mode():
        correspondences = matcher(input_dict)

    # 5. Process results
    batch_idx = correspondences["batch_indexes"].to(torch.long).cpu().numpy()
    mkpts0 = correspondences["keypoints0"].to(torch.float32).cpu().numpy()
    mkpts1 = correspondences["keypoints1"].to(torch.float32).cpu().numpy()
    
    alphas = {}
    for i in range(batch_size):
        mask = (batch_idx == i)
        pts0 = mkpts0[mask]
        pts1 = mkpts1[mask]

        if len(pts0) < 8:
            alphas[preds[i]] = 0.0
            continue

        # Fast RANSAC
        _, inliers = cv2.findFundamentalMat(pts0, pts1, cv2.USAC_MAGSAC, 0.5, 0.999, 1000)
        alphas[preds[i]] = (np.sum(inliers) / len(inliers) if inliers is not None else 0.0)

    return alphas

def get_alpha_vision_by_image_matching(matcher, query_image_path, database_image_path):
    img1 = K.io.load_image(query_image_path, K.io.ImageLoadType.RGB32)[None, ...]
    img2 = K.io.load_image(database_image_path, K.io.ImageLoadType.RGB32)[None, ...]

    img1 = K.geometry.resize(img1, (600, 375), antialias=True)
    img2 = K.geometry.resize(img2, (600, 375), antialias=True)    

    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
        "image1": K.color.rgb_to_grayscale(img2),
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0

    vision_alpha = np.sum(inliers)/len(inliers)

    return vision_alpha


def standarize_scores(text_scores, vision_scores, vpr_dim, vpr_model):
     # pre-computed mu and std for normalization over GSV
    mu_text  = 0.65
    std_text = 0.07    
    min_text = -6.07
    max_text = 4.92           
    if vpr_model == 'mixvpr':
        # #mixvpr 512
        if vpr_dim == 512:
            mu_img   = 0.0111
            std_img  = 0.05
            min_img  = -5.24
            max_img  = 15.26       
        else: #mixvpr 4096 GSV
            mu_img   = 0.0048
            std_img  = 0.027
            min_img  = -5.55
            max_img  = 30.67
    elif vpr_model == 'eigenplaces':
        mu_img   = 0.043
        std_img  = 0.0596
        min_img  = -5.24
        max_img  = 15.08
    elif vpr_model == 'cricavpr':
        mu_img   = 0.0094
        std_img  = 0.026
        min_img  = -5.67
        max_img  = 28.56
        
        # mu_img   = 0.068
        # std_img  = 0.045
        # min_img  = -3.32
        # max_img  = 20.63
    
    # pre-computed mu and std for normalization over nordland 
    # mu_text  = 0.75
    # std_text = 0.0619
    # min_text = -6.236
    # max_text = 4.02
    # mu_img   = 0.278
    # std_img  = 0.105
    # min_img  = -4.57
    # max_img  = 6.238  
    # mu_text  = 0.75
    # std_text = 0.0619
    # min_text = -6.236
    # max_text = 4.02
    #mixvpr 4096 norland
    # mu_img   = 0.216
    # std_img  = 0.088
    # min_img  = -3.43
    # max_img  = 7.9
    
    # pre-computed mu and std for normalization over pitts
    # mu_text  = 0.641
    # std_text = 0.071
    # min_text = -5.31
    # max_text = 5.03
    # mu_img   = 0.053
    # std_img  = 0.066
    # min_img  = -4.24
    # max_img  = 13.5
    
    # clip scores to avoid negative nan in standarization
    text_scores = np.clip(text_scores, a_min=-1.0, a_max=1.0)
    vision_scores = np.clip(vision_scores, a_min=-1.0, a_max=1.0)
     
    # # normalize scores
    text_scores = (text_scores - mu_text) / std_text
    vision_scores  = (vision_scores - mu_img) / std_img
    text_scores = ((text_scores - min_text) / (max_text - min_text))*2-1
    vision_scores  = ((vision_scores - min_img) / (max_img - min_img))*2-1        
   
    return text_scores, vision_scores


def rerank_predictions_by_text(vision_scores, vision_predictions, text_scores, text_predictions, max_results):
    #get max_results of vision predictions and sort these ids by their text scores
    top_vision_ids = vision_predictions[:, :100]
    # filter  top_vision_ids from text predictions     
    top_text_predictions = np.take_along_axis(text_predictions, top_vision_ids, axis=1)
    top_text_scores = np.take_along_axis(text_scores, top_vision_ids, axis=1)
    # #sort top_text_scores and return their indices from highest to lowest
    text_indices = np.argsort(-top_text_scores, axis=1)
    # #get the final predictions
    final_predictions = np.take_along_axis(top_text_predictions, text_indices, axis=1)
    final_scores = np.take_along_axis(top_text_scores, text_indices, axis=1)
    return final_scores, final_predictions


def rerank_predictions_by_scores(test_ds, vision_scores, vision_predictions, text_scores, text_predictions, w_alpha, max_results, query_index, is_normalize, rerank_by_matching, max_rerank, vision_scores_ref=None):
    # sum scores according the where vision and text predictions are the same
    combined_scores = []
    combined_predictions = []
    
    # standarize scores
    if is_normalize:
        text_scores, vision_scores = standarize_scores(text_scores, vision_scores, args.vpr_dim, args.vpr_model_name)
        if vision_scores_ref is not None:
            vision_scores = wasserstein_transform_batch(vision_scores, vision_scores_ref)    

    logger.info(f"mean w_alpha vision: {w_alpha[:,0].mean()}, {w_alpha[:,0].std()}")
    logger.info(f"mean w_alpha text: {w_alpha[:,1].mean()}, {w_alpha[:,1].std()}")
    
    if rerank_by_matching:
        matcher = KF.LoFTR(pretrained="outdoor").to('cuda').bfloat16()
   
    try:    
        for v_scores, v_preds, t_scores, t_preds in zip(vision_scores, vision_predictions, text_scores, text_predictions):
            score_dict = {}                  
            query_image_path = test_ds.images_paths[query_index]            
            alpha_vision_dict = {}
            w_query_v = w_alpha[query_index][0]         
            if rerank_by_matching:
                selected_paths = [test_ds.images_paths[i] for i in v_preds]
                selected_paths = selected_paths[:max_rerank]
                alpha_vision_dict = get_alpha_vision_batches(matcher, query_image_path, selected_paths, v_preds)
            for score, pred in zip(v_scores, v_preds):                               
                if rerank_by_matching:
                    # database_image_path = test_ds.images_paths[pred]
                    # alpha_vision = get_alpha_vision_by_image_matching(matcher, query_image_path, database_image_path)
                    # alpha_vision_list[pred] = alpha_vision
                    if pred in alpha_vision_dict:
                        alpha_vision = alpha_vision_dict[pred]
                    else:
                        continue
                else:
                    alpha_vision = (w_alpha[pred][0]+w_query_v)/2                
                if pred not in score_dict:
                    score_dict[pred] = 0
                #score_dict[pred] += w_alpha[pred][0] * score 
                score_dict[pred] += alpha_vision * score 
            w_query_t = w_alpha[query_index][1]
            for score, pred in zip(t_scores, t_preds):
                if rerank_by_matching:
                    if pred in alpha_vision_dict:
                        alpha_text = 1-alpha_vision_dict[pred] 
                    else:
                        continue
                else:
                    alpha_text = (w_alpha[pred][1]+w_query_t)/2                
                if pred not in score_dict:
                    score_dict[pred] = 0            
                #score_dict[pred] += w_alpha[pred][1] * score 
                score_dict[pred] += alpha_text * score 
            # sort by score
            sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            preds_sorted = [item[0] for item in sorted_items][:max_results]
            scores_sorted = [item[1] for item in sorted_items][:max_results]
            combined_predictions.append(preds_sorted)
            combined_scores.append(scores_sorted)
            query_index += 1
        
        combined_predictions = np.array(combined_predictions)
        combined_scores = np.array(combined_scores)
    except Exception as e:
        print(f'Error: {e}')
        
    return combined_scores, combined_predictions

def rerank_predictions_by_rank(vision_scores, vision_predictions, text_scores, text_predictions, w_alpha, max_results, query_index):
    # sum scores according the where vision and text predictions are the same
    combined_scores = []
    combined_predictions = []

    logger.info(f"mean w_alpha vision: {w_alpha[:,0].mean()}, {w_alpha[:,0].std()}")
    logger.info(f"mean w_alpha text: {w_alpha[:,1].mean()}, {w_alpha[:,1].std()}")
   
    try:    
        for v_scores, v_preds, t_scores, t_preds in zip(vision_scores, vision_predictions, text_scores, text_predictions):
            score_dict = {}      
            w_query_v = w_alpha[query_index][0]
            i = 1
            for score, pred in zip(v_scores, v_preds):
                if pred not in score_dict:
                    score_dict[pred] = 0
                score_dict[pred] += (w_alpha[pred][0]+w_query_v)/2 * i
                i+=1
            w_query_t = w_alpha[query_index][1]
            i = 1
            for score, pred in zip(t_scores, t_preds):
                if pred not in score_dict:
                    score_dict[pred] = 0            
                score_dict[pred] += (w_alpha[pred][1]+w_query_t)/2 * i
                i+=1
            # sort by score
            sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=False)
            preds_sorted = [item[0] for item in sorted_items][:max_results]
            scores_sorted = [item[1] for item in sorted_items][:max_results]
            combined_predictions.append(preds_sorted)
            combined_scores.append(scores_sorted)
            query_index += 1
        
        combined_predictions = np.array(combined_predictions)
        combined_scores = np.array(combined_scores)
    except Exception as e:
        print(f'Error: {e}')
        
    return combined_scores, combined_predictions

def normlize_features(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)    

def compute_ecdf(x):
    """Return sorted samples and empirical CDF values."""
    xs = np.sort(x)
    n = len(xs)
    ps = (np.arange(n) + 0.5) / n   # midpoint ECDF in (0,1)
    return xs, ps


def wasserstein_transform_batch(X, target):
    """
    Apply row-wise 1D Wasserstein transform:
    For each i:  X[i] is mapped to target[i].

    X:      array (N, K)
    target: array (N, K)

    Returns:
        transformed: array (N, K)
    """
    N, K = X.shape
    Nt, Kt = target.shape
    assert N == Nt and K == Kt, "X and target must have identical shapes"

    X_out = np.zeros_like(X)

    for i in range(N):

        row = X[i]
        tgt = target[i]

        # --- ECDF of source row ---
        xs, ps = compute_ecdf(row)
        order = np.argsort(row)
        inv_order = np.argsort(order)
        p_row = ps[inv_order]       # ECDF values in original order

        # --- ICDF of target row ---
        xt_sorted, pt = compute_ecdf(tgt)
        icdf = interp1d(
            pt, xt_sorted,
            bounds_error=False,
            fill_value="extrapolate"
        )

        # clamp probabilities to target ECDF domain
        pmin, pmax = pt[0], pt[-1]
        p_clamped = np.clip(p_row, pmin, pmax)

        # --- Wasserstein map ---
        X_out[i] = icdf(p_clamped)

    return X_out

def encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha):
    if args.bfloat16:
        images = images.bfloat16()
    if args.encode_mode == 'text':
        # single vector - text
        descriptors = model.encode_text(texts)
        descriptors = descriptors.to(torch.float32).cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors        
        text_descriptors[indices.numpy(), :] = descriptors 
    elif args.encode_mode == 'image':
        # single vector - image
        descriptors = model.encode_image(images.to(args.device))
        descriptors = descriptors.to(torch.float32).cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors    
        vision_descriptors[indices.numpy(), :] = descriptors
    elif args.cross_modal==1:
        image_features = model.encode_text(texts)
        image_features = image_features.cpu().numpy()
        vision_descriptors[indices.numpy(), :] = image_features     
        text_features = model.encode_image(images.to(args.device))
        text_features = text_features.cpu().numpy()
        text_descriptors[indices.numpy(), :] = text_features            
    elif args.is_dual_encoder:
        image_features, text_features = model.encode_dual(images.to(args.device), texts)
        # cat fusion: concat text and vision vectors
        if args.dual_encoder_fusion == 'cat':
            descriptors = torch.cat((image_features, text_features), dim=1)
            descriptors = descriptors.to(torch.float32).cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        # each fusion: save each modality
        image_features = image_features.to(torch.float32).cpu().numpy()
        vision_descriptors[indices.numpy(), :] = image_features        
        text_features = text_features.to(torch.float32).cpu().numpy()
        text_descriptors[indices.numpy(), :] = text_features                    
    else:
        # single vector of both image and text
        descriptors, text_features, w = model.encode_single(images.to(args.device), texts)
        descriptors = descriptors.cpu().numpy()        
        if args.cross_modal:
            vision_descriptors[indices.numpy(), :] = descriptors
            text_features = text_features.cpu().numpy()
            text_descriptors[indices.numpy(), :] = text_features
        elif args.fusion_type == 'dynamic_weighting' or args.fusion_type == 'fixed_weighting' or args.fusion_type == 'text_adapter' or args.fusion_type == 'transformer':
            vision_descriptors[indices.numpy(), :] = descriptors
            text_features = text_features.cpu().numpy()
            text_descriptors[indices.numpy(), :] = text_features  
            w = w.cpu().numpy()
            if args.fusion_type == 'fixed_weighting':
                #make w a 2D vector of [w, 1-w]. w in numpy
                w = np.repeat(w, indices.shape[0], axis=0)
                #make w a 2D vector of [w, 1-w]
                w_alpha[indices.numpy(), :] = np.stack([w, 1-w], axis=1)                
            else:
                w_alpha[indices.numpy(), :] = w
        else:
            all_descriptors[indices.numpy(), :] = descriptors        
            
def get_queries_predictions(encoder_dim, database_descriptors, all_descriptors, queries_descriptors, max_results):
     # Use a kNN to find predictions
    #faiss_index = faiss.IndexFlatL2(encoder_dim)
    faiss_index = faiss.IndexFlatIP(encoder_dim)
    #normilize descriptors for cosine similarity
    database_descriptors = normlize_features(database_descriptors)      
    queries_descriptors = normlize_features(queries_descriptors)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    scores, predictions = faiss_index.search(queries_descriptors, max_results)
    return scores, predictions

def do_pca(descriptors, pca_dim):
    logger.debug("Fitting PCA on all descriptors")
    pca = PCA(n_components=pca_dim)
    pca.fit(descriptors)
    logger.debug("Transforming all descriptors using PCA")                        
    descriptors = pca.transform(descriptors)
    return descriptors

def save_worst_queries(test_ds, predictions, args, K):
    query_results = []
    positives_per_query = test_ds.get_positives()

    # recall_values usually looks like [1, 5, 10, 20]
    # We'll use the largest N to define "worst" (e.g., Recall@20)
    max_n_index = -1 
    max_n = args.recall_values[max_n_index]
    k = 5

    for query_index, preds in enumerate(predictions):
        # Calculate recall for this specific query across all N values        
        bFound = 0
        if np.any(np.in1d(preds[:k], positives_per_query[query_index])):
            bFound = 1
        image_path = test_ds.queries_paths[query_index]
        desc_index = test_ds.images_paths_csv.index(image_path)
        description = test_ds.descriptions[desc_index]
        
        # 2. Store the metadata and the target recall metric
        query_results.append({            
            'image_path': image_path,
            'description': description,
            'query_index': query_index,
            'recall_at_max': bFound,            
        })

    # 3. Create DataFrame and sort to find the "worst"
    df_results = pd.DataFrame(query_results)

    # Sort by recall (ascending) so the 0s and lowest values are at the top
    df_worst = df_results.sort_values(by='recall_at_max', ascending=True).head(K)

    # 4. Save to CSV
    df_worst.to_csv('top_worst_queries.csv', index=False)
    print(f"Saved the 50 worst queries to top_worst_queries.csv")

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    start_time = datetime.now()

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(f"Testing with {args.vpr_model_name}")
    logger.info(f"The outputs are being saved in {log_dir}")

    IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    BLIP_MEAN_STD = {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
    SIGLIP_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    dataset_mean_std = IMAGENET_MEAN_STD
    if 'blip' in args.vpr_model_name.lower() or 'clip' in args.vpr_model_name.lower() or 'eva' in args.vpr_model_name.lower():
        dataset_mean_std = BLIP_MEAN_STD
    elif 'siglip' in args.vpr_model_name.lower():
        dataset_mean_std = SIGLIP_MEAN_STD

    model = VLM_Model(args)
    logger.info(f"VLM encoder dim: {model.encoder_dim}")
    
    ref_vision_scores = None
    if args.is_ref_model:
        # shallow copy args to ref_args
        ref_args = argparse.Namespace(**vars(args)) 
        ref_args.is_dual_encoder = True
        ref_args.encode_mode = 'image'
        ref_args.vision_dimension = 512
        ref_args.vpr_rows = 2
        ref_model = VLM_Model(ref_args)

    is_msls_challenge = False
    if 'msls_challenge' in args.image_root:        
        test_ds = MSLSTest(dataset_root=args.database_folder, image_root=args.image_root, csv_path=args.queries_csv, mean_std=dataset_mean_std, image_size=args.image_size)
        is_msls_challenge = True
    else:
        test_ds = TestDataset(
            args.database_folder,   
            args.queries_folder,
            args.queries_csv,
            args.image_root,        
            mean_std=dataset_mean_std,
            positive_dist_threshold=args.positive_dist_threshold,
            image_size=args.image_size,
            use_labels=args.use_labels,
        )
    logger.info(f"Testing on {test_ds}")
    all_descriptors = None
    vision_descriptors = None
    text_descriptors = None
    
    max_results = max(args.recall_values)
    query_index = 0

    logger.info(f"VPR dimension: {model.vpr_encoder_dim}, text dimension: {model.text_encoder_dim}, fusion type: {args.fusion_type}, is text pooling: {args.is_text_pooling}, is dual encoder: {args.is_dual_encoder}")

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )

        # if args.is_pca and args.fusion_type == 'dynamic_weighting':
        #     vision_descriptors = np.empty((len(test_ds), args.pca_dim), dtype="float32")
        # else:
        #     vision_descriptors = np.empty((len(test_ds), model.vpr_encoder_dim), dtype="float32")
        if args.is_ref_model:
            ref_vision_descriptors = np.empty((len(test_ds), ref_model.vpr_encoder_dim), dtype="float32")
        vision_descriptors = np.empty((len(test_ds), model.vpr_encoder_dim), dtype="float32")
        text_descriptors = np.empty((len(test_ds), model.text_encoder_dim), dtype="float32")            
        all_descriptors = np.empty((len(test_ds), model.encoder_dim), dtype="float32")
        w_alpha = np.empty((len(test_ds), 2), dtype="float32")
        w_alpha[:,0] = args.alpha_vision
        w_alpha[:,1] = 1.0-args.alpha_vision
            
        for images, indices, texts in tqdm(database_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha)
            if args.is_ref_model:
                descriptors = ref_model.encode_image(images.to(args.device))
                descriptors = descriptors.cpu().numpy()
                ref_vision_descriptors[indices.numpy(), :] = descriptors                    

        query_index = test_ds.num_database
        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size)#1)
        for images, indices, texts in tqdm(queries_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha)
            if args.is_ref_model:
                descriptors = ref_model.encode_image(images.to(args.device))
                descriptors = descriptors.cpu().numpy()
                ref_vision_descriptors[indices.numpy(), :] = descriptors                    
        
        if args.is_pca:            
            vision_descriptors = do_pca(vision_descriptors, args.pca_dim)
            model.vpr_encoder_dim = args.pca_dim
            logger.info(f"PCA reduced vision descriptors to dimension: {model.vpr_encoder_dim}")
            if args.encode_mode == 'image':
                all_descriptors = vision_descriptors
                model.encoder_dim = all_descriptors.shape[1]
            if args.fusion_type == 'cat' or (args.is_dual_encoder and args.dual_encoder_fusion == 'cat'):
                all_descriptors = np.concatenate((vision_descriptors, text_descriptors), axis=1)
                model.encoder_dim = all_descriptors.shape[1]
                logger.info(f"Concatenated descriptors dimension: {model.encoder_dim}")
   
    alpha = args.alpha_vision
    max_results_reranking = test_ds.num_database
    #alpha = w_alpha
    # Get queries predictions with alpha between 0.6 to 0.9 with jumps of 0.1
    #for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    if 1:
        # w_alpha[:,0] = alpha
        # w_alpha[:,1] = 1.0-alpha
        
        if args.cross_modal:
            vision_database_descriptors = vision_descriptors[: test_ds.num_database]    
            text_queries_descriptors = text_descriptors[test_ds.num_database :]
            scores, predictions = get_queries_predictions(model.encoder_dim, vision_database_descriptors, all_descriptors, text_queries_descriptors, max_results)
            
        elif (args.is_dual_encoder and args.dual_encoder_fusion=='each') or args.fusion_type=='dynamic_weighting' or args.fusion_type=='fixed_weighting' or args.fusion_type=='text_adapter' or args.fusion_type == 'transformer': 
            # vision
            vision_queries_descriptors = vision_descriptors[test_ds.num_database :]
            vision_database_descriptors = vision_descriptors[: test_ds.num_database]    
            
            vision_scores, vision_predictions = get_queries_predictions(model.vpr_encoder_dim, vision_database_descriptors, vision_descriptors, vision_queries_descriptors, max_results_reranking)
            # text
            text_queries_descriptors = text_descriptors[test_ds.num_database :]
            text_database_descriptors = text_descriptors[: test_ds.num_database]                
            text_scores, text_predictions = get_queries_predictions(model.text_encoder_dim, text_database_descriptors, text_descriptors, text_queries_descriptors, max_results_reranking)
            if args.rerank_by_text:
                scores, predictions = rerank_predictions_by_text(vision_scores, vision_predictions, text_scores, text_predictions, max_results)
            # join vision and text predictions        
            elif args.rerank_by_scores:
                if args.is_ref_model:
                    ref_vision_queries_descriptors = ref_vision_descriptors[test_ds.num_database :]
                    ref_vision_database_descriptors = ref_vision_descriptors[: test_ds.num_database]    
                    ref_vision_scores, ref_vision_predictions = get_queries_predictions(ref_model.vpr_encoder_dim, ref_vision_database_descriptors, ref_vision_descriptors, ref_vision_queries_descriptors, max_results_reranking)
                    mu_img   = 0.0111
                    std_img  = 0.05
                    min_img  = -5.24
                    max_img  = 15.26
                    ref_vision_scores = np.clip(ref_vision_scores, a_min=-1.0, a_max=1.0)
                    ref_vision_scores  = (ref_vision_scores - mu_img) / std_img
                    ref_vision_scores  = ((ref_vision_scores - min_img) / (max_img - min_img))*2-1     
                scores, predictions = rerank_predictions_by_scores(test_ds, vision_scores, vision_predictions, text_scores, text_predictions, w_alpha, max_results, query_index, args.is_normalize, args.rerank_by_matching, args.max_rerank, ref_vision_scores)
            else:
                scores, predictions = rerank_predictions_by_rank(vision_scores, vision_predictions, text_scores, text_predictions, w_alpha, max_results, query_index)
                
        else:
            queries_descriptors = all_descriptors[test_ds.num_database :]
            database_descriptors = all_descriptors[: test_ds.num_database]    
            logger.info(f"dim database descriptors: {all_descriptors.shape[1]}")
            # get queries predictions
            scores, predictions = get_queries_predictions(model.encoder_dim, database_descriptors, all_descriptors, queries_descriptors, max_results)
        
        if is_msls_challenge:
            # save predictions to msls_challenge format
            test_ds.save_predictions(predictions, log_dir / "msls_challenge_predictions.txt", k=25)
        else:
            # For each query, check if the predictions are correct
            if args.use_labels:
                positives_per_query = test_ds.get_positives()
                recalls = np.zeros(len(args.recall_values))
                for query_index, preds in enumerate(predictions):
                    for i, n in enumerate(args.recall_values):
                        if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                            recalls[i:] += 1
                            break

                # Divide by num_queries and multiply by 100, so the recalls are in percentages
                recalls = recalls / test_ds.num_queries * 100
                recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
                logger.info(recalls_str)
                
                #open eval_vpr_results.csv in append mode and write the recalls
                with open("eval_vpr_results.csv", "a") as f:
                    f.write(f"{args.vpr_model_name},{w_alpha[0,0]},{args.fusion_type},{args.is_text_pooling},{args.vpr_dim},{args.is_pca},{args.encode_mode},{recalls_str}\n")

                
                #save_worst_queries(test_ds, predictions, args, 800)
                
    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels, test_ds.images_paths_csv, texts=test_ds.descriptions
        )


if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
