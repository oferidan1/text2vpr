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
        default="blip",
        choices=[
            "clip",
            "blip",
            "siglip",            
        ],
        help="_",
    )
    parser.add_argument("--descriptors_dimension", type=int, default=256, help="_")
    parser.add_argument("--database_folder", type=str, default="/mnt/d/dan/datasets/sf_xl/processed/test/database")
    #parser.add_argument("--database_folder", type=str, default="/mnt/d/dan/datasets/sf_xl/processed/test/dummy")
    parser.add_argument("--queries_folder", type=str, default="/mnt/d/dan/datasets/sf_xl/processed/test/queries_night")
    parser.add_argument("--queries_csv", type=str, default="/mnt/d/dan/datasets/descriptions.csv")
    parser.add_argument("--num_workers", type=int, default=4, help="_")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="set to 1 if database images may have different resolution"
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
        "--num_preds_to_save", type=int, default=3, help="set != 0 if you want to save predictions for each query"
    )
    parser.add_argument(
        "--save_only_wrong_preds",
        action="store_true",
        help="set to true if you want to save predictions only for " "wrongly predicted queries",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        nargs="+",
        help="Resizing shape for images (HxW). If a single int is passed, set the"
        "smallest edge of all images to this value, while keeping aspect ratio",
    )
    parser.add_argument(
        "--save_descriptors",
        action="store_true",
        help="set to True if you want to save the descriptors extracted by the model",
        default=True,
    )
    
    args = parser.parse_args()
    
    args.use_labels = not args.no_labels
        
    return args
