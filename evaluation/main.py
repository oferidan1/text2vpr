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
from models import VLM_Model
import os
from test_dataset import TestDataset, QueryTextDataset
import visualizations


def main(args):
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

    model = VLM_Model()
    
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
        model.get_processor(),
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
        positives_per_query=positives_per_query,
    )
    logger.info(f"Testing on {test_ds}")

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        if not is_database_descriptors_exist:
            for images, indices in tqdm(database_dataloader):
                descriptors = model.encode_images(images.to(args.device))            
                all_descriptors[indices.numpy(), :] = descriptors

        #logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        # queries_subset_ds = Subset(
        #     test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        # )
        
            queries_ds = QueryTextDataset(args.queries_csv, model.get_processor())        
            queries_dataloader = DataLoader(dataset=queries_ds, num_workers=args.num_workers, batch_size=1)
            for input_ids, indices in tqdm(queries_dataloader):
                descriptors = model.encode_texts(input_ids.to(args.device))
                all_descriptors[indices.numpy(), :] = descriptors

            queries_descriptors = all_descriptors[test_ds.num_database :]
            database_descriptors = all_descriptors[: test_ds.num_database]

    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {args.descriptor_dir}")
        if not Path(args.descriptor_dir).exists():
            Path(args.descriptor_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(args.descriptor_dir, "queries_descriptors.npy"), queries_descriptors)
        np.save(os.path.join(args.descriptor_dir, "database_descriptors.npy"), database_descriptors)
        positives_per_query = test_ds.get_positives()
        np.save(os.path.join(args.descriptor_dir, "positives_per_query.npy"), positives_per_query)

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))

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
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels
        )


if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
