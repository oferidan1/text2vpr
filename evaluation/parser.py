import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--positive_dist_threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mixvpr",
        help="_",
    )
    parser.add_argument("--vision_dimension", type=int, default=512, help="_")
    parser.add_argument("--vpr_rows", type=int, default=2, help="number of rows for vpr embeddings")
    
    # parser.add_argument("--database_folder", type=str, default="/mnt/d/data/amstertime/test/database")    
    # parser.add_argument("--queries_folder", type=str, default="/mnt/d/data/amstertime/test/queries")
    # parser.add_argument("--queries_csv", type=str, default="/mnt/d/data/amstertime/test/amstertime_test_predictions.csv")
    # parser.add_argument("--image_root", type=str, default="/mnt/d/data/amstertime/test")
    # parser.add_argument("--database_folder", type=str, default="/mnt/d/data/nordland/images/test/database")    
    # parser.add_argument("--queries_folder", type=str, default="/mnt/d/data/nordland/images/test/queries")
    # parser.add_argument("--queries_csv", type=str, default="/mnt/d/data/nordland/images/test/nordland_predictions.csv")
    # parser.add_argument("--image_root", type=str, default="/mnt/d/data/nordland/images/test")
    # parser.add_argument("--database_folder", type=str, default="/mnt/d/data/Pittsburgh30K/images/test/database")    
    # parser.add_argument("--queries_folder", type=str, default="/mnt/d/data/Pittsburgh30K/images/test/queries")
    # parser.add_argument("--queries_csv", type=str, default="/mnt/d/data/Pittsburgh30K/images/test/Pittsburgh30K_test_predictions.csv")
    # parser.add_argument("--image_root", type=str, default="/mnt/d/data/Pittsburgh30K/images/test")    
    parser.add_argument("--database_folder", type=str, default="/mnt/d/data/gsv_cities/Images")    
    parser.add_argument("--queries_folder", type=str, default="/mnt/d/data/gsv_cities/Images")
    parser.add_argument("--queries_csv", type=str, default="/mnt/d/data/gsv_cities/gsv_cities_predictions.csv")
    parser.add_argument("--image_root", type=str, default="/mnt/d/data/gsv_cities")
    
    parser.add_argument("--num_workers", type=int, default=4, help="_")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="set to 1 if database images may have different resolution"
    )
    parser.add_argument(
        "--log_dir", type=str, default="default", help="experiment name, output logs will be saved under logs/log_dir"
    )
    parser.add_argument("--descriptor_dir", type=str, default="descriptors", help="_")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
    parser.add_argument(
        "--recall_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="values for recall (e.g. recall@1, recall@5)",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="set to true if you have no labels and just want to "
        "do standard image retrieval given two folders of queries and DB",
    )
    parser.add_argument(
        "--num_preds_to_save", type=int, default=0, help="set != 0 if you want to save predictions for each query"
    )
    parser.add_argument(
        "--save_only_wrong_preds",
        action="store_true",
        help="set to true if you want to save predictions only for " "wrongly predicted queries",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=320,
        nargs="+",
        help="Resizing shape for images (HxW). If a single int is passed, set the"
        "smallest edge of all images to this value, while keeping aspect ratio",
    )
    parser.add_argument(
        "--save_descriptors",
        action="store_true",
        help="set to True if you want to save the descriptors extracted by the model",
    )
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")
    parser.add_argument("--model_name", type=str, default='/mnt/d/ofer/localization/text2vpr/MixVPR_text/LOGS/resnet50/lightning_logs/version_3/checkpoints/resnet50_epoch(04)_step(2605)_R1[0.9071]_R5[0.9551].ckpt')
    parser.add_argument("--vision_model_name", type=str, default="mixvpr")
    parser.add_argument("--text_model_name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--lora_path", type=str, default=None)    
    parser.add_argument("--is_dual_encoder", type=int, default="1", help="is dual encoder")    
    parser.add_argument("--dual_encoder_fusion", type=str, default="each", help="cat/each")    
    parser.add_argument("--encode_mode", type=str, default="both", help="both/image/text")   
    parser.add_argument("--fusion_type", type=str, default='dynamic_weighting', help="type of fusion to use: mlp, transformer, dynamic_weighting")
    parser.add_argument("--is_normalize_features", type=int, default="0", help="is normalize features")    
    parser.add_argument("--max_results_reranking", type=int, default="10000", help="max results for reranking")    
    parser.add_argument("--alpha_vision", type=float, default=0.9, help="weight for vision scores in reranking")    
    parser.add_argument("--alpha_loop", type=int, default=0, help="try multiple alpha values in loop of reranking")    

    args = parser.parse_args()
    
    args.use_labels = not args.no_labels
        
    return args
