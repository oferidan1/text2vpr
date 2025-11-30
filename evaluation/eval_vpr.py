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
from sklearn.decomposition import PCA

def rerank_predictions_by_scores(vision_scores, vision_predictions, text_scores, text_predictions, w_alpha, max_results, query_index):
    # sum scores according the where vision and text predictions are the same
    combined_scores = []
    combined_predictions = []
    # pre-computed mu and std for normalization over GSV
    # mu_text  = 0.65
    # std_text = 0.07    
    # min_text = -6.07
    # max_text = 4.92           
    # # #mixvpr 512
    # mu_img   = 0.0111
    # std_img  = 0.05
    # min_img  = -5.24
    # max_img  = 15.26
    #mixvpr 4096 GSV
    # mu_img   = 0.0048
    # std_img  = 0.027
    # min_img  = -5.55
    # max_img  = 30.67
    
    # pre-computed mu and std for normalization over amstertime
    # mu_text  = 0.66
    # std_text = 0.066
    # min_text = -4.81
    # max_text = 4.43
    # mu_img   = 0.06
    # std_img  = 0.06
    # min_img  = -4.57
    # max_img  = 15.32
    
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
    #clip to avoid negative scores
    text_scores = np.clip(text_scores, a_min=-1.0, a_max=1.0)
    vision_scores = np.clip(vision_scores, a_min=-1.0, a_max=1.0)
    # # normalize scores
    # text_scores = (text_scores - mu_text) / std_text
    # vision_scores  = (vision_scores - mu_img) / std_img
    # text_scores = ((text_scores - min_text) / (max_text - min_text))*2-1
    # vision_scores  = ((vision_scores - min_img) / (max_img - min_img))*2-1        

    print("mean w_alpha vision:", w_alpha[:,0].mean(), w_alpha[:,0].std())
    print("mean w_alpha text:", w_alpha[:,1].mean(), w_alpha[:,1].std())
   
    try:    
        for v_scores, v_preds, t_scores, t_preds in zip(vision_scores, vision_predictions, text_scores, text_predictions):
            score_dict = {}      
            w_query_v = w_alpha[query_index][0]
            for score, pred in zip(v_scores, v_preds):
                if pred not in score_dict:
                    score_dict[pred] = 0
                #score_dict[pred] += w_alpha[pred][0] * score 
                score_dict[pred] += (w_alpha[pred][0]+w_query_v)/2 * score 
            w_query_t = w_alpha[query_index][1]
            for score, pred in zip(t_scores, t_preds):
                if pred not in score_dict:
                    score_dict[pred] = 0            
                #score_dict[pred] += w_alpha[pred][1] * score 
                score_dict[pred] += (w_alpha[pred][1]+w_query_t)/2 * score 
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

    print("mean w_alpha vision:", w_alpha[:,0].mean(), w_alpha[:,0].std())
    print("mean w_alpha text:", w_alpha[:,1].mean(), w_alpha[:,1].std())
   
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
        if args.fusion_type == 'dynamic_weighting' or args.fusion_type == 'fixed_weighting' or args.fusion_type == 'text_adapter':
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


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    start_time = datetime.now()

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(f"Testing with {args.vision_model_name}")
    logger.info(f"The outputs are being saved in {log_dir}")

    model = VLM_Model(args)

    test_ds = TestDataset(
        args.database_folder,   
        args.queries_folder,
        args.queries_csv,
        args.image_root,        
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

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        
        vision_descriptors = np.empty((len(test_ds), model.vision_encoder_dim), dtype="float32")
        text_descriptors = np.empty((len(test_ds), model.text_encoder_dim), dtype="float32")            
        all_descriptors = np.empty((len(test_ds), model.encoder_dim), dtype="float32")
        w_alpha = np.empty((len(test_ds), 2), dtype="float32")
        w_alpha[:,0] = args.alpha_vision
        w_alpha[:,1] = 1.0-args.alpha_vision
            
        for images, indices, texts in tqdm(database_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha)

        query_index = test_ds.num_database
        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size)#1)
        for images, indices, texts in tqdm(queries_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha)
            
        if args.is_pca:
            logger.debug("Fitting PCA on all descriptors")
            pca = PCA(n_components=args.pca_dim)
            pca.fit(all_descriptors)
            logger.debug("Transforming all descriptors using PCA")
            all_descriptors = pca.transform(all_descriptors)
            if (args.is_dual_encoder and args.dual_encoder_fusion=='each') or args.fusion_type=='dynamic_weighting' or args.fusion_type=='fixed_weighting' or args.fusion_type=='text_adapter': 
                logger.debug("Transforming vision descriptors using PCA")
                vision_descriptors = pca.transform(vision_descriptors)
              

    alpha = args.alpha_vision
    #alpha = w_alpha
    # Get queries predictions with alpha between 0.6 to 0.9 with jumps of 0.1
    #for alpha in [0.6, 0.7, 0.8, 0.9, 0.95]:
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # if 1:
        w_alpha[:,0] = alpha
        w_alpha[:,1] = 1.0-alpha
        if (args.is_dual_encoder and args.dual_encoder_fusion=='each') or args.fusion_type=='dynamic_weighting' or args.fusion_type=='fixed_weighting' or args.fusion_type=='text_adapter': 
            # vision
            vision_queries_descriptors = vision_descriptors[test_ds.num_database :]
            vision_database_descriptors = vision_descriptors[: test_ds.num_database]    
            vision_scores, vision_predictions = get_queries_predictions(model.vision_encoder_dim, vision_database_descriptors, vision_descriptors, vision_queries_descriptors, args.max_results_reranking)
            # text
            text_queries_descriptors = text_descriptors[test_ds.num_database :]
            text_database_descriptors = text_descriptors[: test_ds.num_database]    
            text_scores, text_predictions = get_queries_predictions(model.text_encoder_dim, text_database_descriptors, text_descriptors, text_queries_descriptors, args.max_results_reranking)
            # join vision and text predictions        
            if args.rerank_by_scores:
                scores, predictions = rerank_predictions_by_scores(vision_scores, vision_predictions, text_scores, text_predictions, w_alpha, max_results, query_index)
            else:
                scores, predictions = rerank_predictions_by_rank(vision_scores, vision_predictions, text_scores, text_predictions, w_alpha, max_results, query_index)
                
        else:
            queries_descriptors = all_descriptors[test_ds.num_database :]
            database_descriptors = all_descriptors[: test_ds.num_database]    
            # get queries predictions
            scores, predictions = get_queries_predictions(model.encoder_dim, database_descriptors, all_descriptors, queries_descriptors, max_results)
            
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
