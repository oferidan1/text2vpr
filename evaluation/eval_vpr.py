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
            
def get_queries_predictions(encoder_dim, database_descriptors, all_descriptors, queries_descriptors, max_results):
     # Use a kNN to find predictions
    #faiss_index = faiss.IndexFlatL2(encoder_dim)
    faiss_index = faiss.IndexFlatIP(encoder_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    scores, predictions = faiss_index.search(queries_descriptors, max_results)
    return scores, predictions

def rerank_predictions(vision_scores, vision_predictions, text_scores, text_predictions, w_alpha, max_results):
    # sum scores according the where vision and text predictions are the same
    combined_scores = []
    combined_predictions = []
    for v_scores, v_preds, t_scores, t_preds in zip(vision_scores, vision_predictions, text_scores, text_predictions):
        score_dict = {}
        for score, pred in zip(v_scores, v_preds):
            if pred not in score_dict:
                score_dict[pred] = 0
            score_dict[pred] += w_alpha[pred][0] * score
        for score, pred in zip(t_scores, t_preds):
            if pred not in score_dict:
                score_dict[pred] = 0
            #score = (score-0.4)/sqrt(0.4)  # convert cosine sim to z-score
            score_dict[pred] += w_alpha[pred][1] * score
        # sort by score
        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        preds_sorted = [item[0] for item in sorted_items][:max_results]
        scores_sorted = [item[1] for item in sorted_items][:max_results]
        combined_predictions.append(preds_sorted)
        combined_scores.append(scores_sorted)
        
    combined_predictions = np.array(combined_predictions)
    combined_scores = np.array(combined_scores)
    return combined_scores, combined_predictions

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
    logger.info(f"Testing with {args.method}")
    logger.info(f"The outputs are being saved in {log_dir}")

    model = VLM_Model(args)
    
    #if databaase descriptors already exist, skip their computation
    database_descriptors_path = os.path.join(args.descriptor_dir, "database_descriptors.npy")
    is_database_descriptors_exist = False
    positives_per_query = None
    if os.path.exists(database_descriptors_path):        
        database_descriptors = np.load(database_descriptors_path)
        queries_descriptors = np.load(os.path.join(args.descriptor_dir, "queries_descriptors.npy"))            
        positives_per_query = np.load(os.path.join(args.descriptor_dir, "positives_per_query.npy"), allow_pickle=True)
        is_database_descriptors_exist = True

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

        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
        for images, indices, texts in tqdm(queries_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, w_alpha)

    alpha = args.alpha_vision
    alpha = w_alpha
    # Get queries predictions with alpha between 0.6 to 0.9 with jumps of 0.1
    #for alpha in [0.6, 0.7, 0.8, 0.9]:
    #for alpha in [0.9, 0.92, 0.94, 0.96, 0.98]:
    if 1:
        if (args.is_dual_encoder and args.dual_encoder_fusion=='each') or args.fusion_type=='dynamic_weighting':        
            # vision
            vision_queries_descriptors = vision_descriptors[test_ds.num_database :]
            vision_database_descriptors = vision_descriptors[: test_ds.num_database]    
            vision_scores, vision_predictions = get_queries_predictions(model.vision_encoder_dim, vision_database_descriptors, vision_descriptors, vision_queries_descriptors, args.max_results_reranking)
            # text
            text_queries_descriptors = text_descriptors[test_ds.num_database :]
            text_database_descriptors = text_descriptors[: test_ds.num_database]    
            text_scores, text_predictions = get_queries_predictions(model.text_encoder_dim, text_database_descriptors, text_descriptors, text_queries_descriptors, args.max_results_reranking)
            # join vision and text predictions        
            scores, predictions = rerank_predictions(vision_scores, vision_predictions, text_scores, text_predictions, alpha, max_results)
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


    if args.save_descriptors and not is_database_descriptors_exist:
        logger.info(f"Saving the descriptors in {args.descriptor_dir}")
        if not Path(args.descriptor_dir).exists():
            Path(args.descriptor_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(args.descriptor_dir, "queries_descriptors.npy"), queries_descriptors)
        np.save(os.path.join(args.descriptor_dir, "database_descriptors.npy"), database_descriptors)
        positives_per_query = test_ds.get_positives()
        np.save(os.path.join(args.descriptor_dir, "positives_per_query.npy"), positives_per_query)

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
