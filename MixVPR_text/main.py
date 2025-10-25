import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
import utils
from torch import nn

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models import helper
from sentence_transformers import SentenceTransformer
import os
import argparse
from mixvpr_text import VPRModel_text


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Resume parameters
    parser.add_argument("--resume_model", type=str, default='checkpoints/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt', help="path to model to resume, e.g. logs/.../best_model.pth")
    # Other parameters    
    parser.add_argument("--gpu", type=str, default='0', help="gpu id(s) to use")
    parser.add_argument("--vpr_dim", type=int, default=512, help="dimension of the vpr embeddings")
    parser.add_argument("--train_csv", type=str, default="/mnt/d/data/gsv_cities/gsv_cities_predictions.csv")    
    parser.add_argument("--image_root", type=str, default="/mnt/d/data/gsv_cities/", help="root directory for images")
    parser.add_argument("--text_encoder", type=str, default="BAAI/bge-large-en-v1.5", help="text encoder model name")
    parser.add_argument("--is_freeze_text", type=int, default="1", help="freeze text encoder or not")
    parser.add_argument("--is_freeze_vpr", type=int, default="1", help="freeze vpr encoder or not")    
    parser.add_argument("--embeds_dim", type=int, default=1024, help="dimension of the embeddings")
    parser.add_argument("--fusion_type", type=str, default='dynamic_weighting', help="type of fusion to use: mlp, transformer, dynamic_weighting")
    parser.add_argument("--is_encode_image", type=int, default="1", help="encode image or not")
    parser.add_argument("--is_encode_text", type=int, default="1", help="encode text or not")
    parser.add_argument("--is_trainable_text_encoder", type=int, default="0", help="train text encoder or not")
    parser.add_argument("--batch_size", type=int, default="120", help="batch size for training")
    parser.add_argument("--loss_name", type=str, default="MultiSimilarityLoss_Sij", help="name of the loss function to use")

    args = parser.parse_args()
    
    return args            
            
if __name__ == '__main__':    
    pl.utilities.seed.seed_everything(seed=190223, workers=True)
    
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(320, 320),
        num_workers=4,#28,
        show_data_stats=True,
        #val_set_names=['pitts30k_val', 'pitts30k_test', 'msls_val'], # pitts30k_val, pitts30k_test, msls_val
        val_set_names=['pitts30k_test'],
    )
    
    # examples of backbones
    # resnet18, resnet50, resnet101, resnet152,
    # resnext50_32x4d, resnext50_32x4d_swsl , resnext101_32x4d_swsl, resnext101_32x8d_swsl
    # efficientnet_b0, efficientnet_b1, efficientnet_b2
    # swinv2_base_window12to16_192to256_22kft1k
    model = VPRModel_text(
        #---- Encoder
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        agg_arch='MixVPR',
        
        agg_config={'in_channels' : 1024,
                'in_h' : 20,
                'in_w' : 20,
                'out_channels' : args.vpr_dim // 4, # final out dim will be out_channels * out_rows
                'mix_depth' : 4,
                'mlp_ratio' : 1,
                'out_rows' : 4}, # the output dim will be (out_rows * out_channels)
        
        #---- Train hyperparameters
        lr=0.05, # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
        optimizer='sgd', # sgd, adamw
        weight_decay=0.001, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        warmpup_steps=650,
        milestones=[2,4,6,8], # epochs where lr is decayed
        lr_mult=0.3,

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name=args.loss_name,
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False,
        text_encoder_name=args.text_encoder,
        embeds_dim=args.embeds_dim,
        is_freeze_vpr=args.is_freeze_vpr,
        is_freeze_text=args.is_freeze_text,
        fusion_type=args.fusion_type,
        is_encode_image=args.is_encode_image,
        is_encode_text=args.is_encode_text,
        is_trainable_text_encoder=args.is_trainable_text_encoder,
    )
    
    if args.is_encode_image and  args.resume_model is not None:
        model_state_dict = torch.load(args.resume_model)
        model.vpr_encoder.load_state_dict(model_state_dict)
        
    model = model.to('cuda')
    
    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = ModelCheckpoint(
        monitor='pitts30k_test/R1',
        filename=f'{"resnet50"}' +
        '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_test/R1:.4f}]_R5[{pitts30k_test/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max',)

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu', devices=[0],
        default_root_dir=f'./LOGS/{"resnet50"}', # Tensorflow can be used to viz

        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=16, # we use half precision to reduce  memory usage
        max_epochs=10,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    )
    
    # # Manually call validation
    # trainer.validate(model=model, datamodule=datamodule)

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
