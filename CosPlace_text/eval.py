
import sys
import torch
import logging
import multiprocessing
from datetime import datetime

import test as test
import parser
import commons
from cosplace_model import cosplace_network
from datasets.test_dataset import TestDataset, QueryTextDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

args = parser.parse_arguments(is_training=False)
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

#### Model
#model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
model = cosplace_network.cosplace_text(args.backbone, args.fc_output_dim, args.train_all_layers, args.text_encoder, args)    

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.info(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)
else:
    logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

model = model.to(args.device)

test_ds = TestDataset(args.test_set_folder, queries_folder="queries_night",
                      positive_dist_threshold=args.positive_dist_threshold, labels_csv=args.test_csv, image_root=args.image_root)

recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
logging.info(f"{test_ds}: {recalls_str}")
