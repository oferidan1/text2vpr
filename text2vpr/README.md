# text2vpr 
# this repo targets the task of text to visual place recognition (VPR)

# evaluation
# image only
python eval_vpr.py --vpr_dim=512 --vpr_model_name=mixvpr --encode_mode=image --is_encode_text=0 --fusion_type=none --is_dual_encoder=0

# text only
python eval_vpr.py --vpr_dim=512 --vpr_model_name=mixvpr --encode_mode=text --is_encode_image=0 --fusion_type=none --is_dual_encoder=0

# concat
python eval_vpr.py --vpr_dim=512 --vpr_model_name=mixvpr --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none

# each modality seperately
python eval_vpr.py --vpr_dim=512 --vpr_model_name=mixvpr --fusion_type=none --is_dual_encoder=1 --dual_encoder_fusion=each

# cross modal
python eval_vpr.py --vpr_dim=256 --image_size=384 --vpr_model_name=Salesforce/blip-itm-base-coco --is_dual_encoder=0 --fusion_type=none --cross_modal=1 --text_dim=256

python eval_vpr.py --vpr_dim=512 --vpr_model_name=openai/clip-vit-base-patch32  --is_dual_encoder=0 --fusion_type=none 
--cross_modal=1 --image_size=224 --text_dim=512

python eval_vpr.py --vpr_dim=768 --vpr_model_name=google/siglip2-base-patch16-224  --is_dual_encoder=0 --fusion_type=none --cross_modal=1 --image_size=224 --text_dim=768

python eval_vpr.py --vpr_dim=512 --vpr_model_name=EVA02-B-16 --is_dual_encoder=0 --fusion_type=none --cross_modal=1 --image_size=224 --text_dim=512

python train.py --cross_modal=3 --fusion_type=none --vpr_model_name=dinov2 --vpr_dim=768 --is_text_pooling=1 --is_image_pooling=1 --image_size=224

python train.py --cross_modal=2 --fusion_type=none --vpr_model_name=Salesforce/blip-itm-base-coco --vpr_dim=256 --is_text_pooling=0 --is_image_pooling=0 --image_size=384 --loss_name=MultiSimilarityLossCM --is_trainable_text_encoder=1 --batch_size=20

python train.py --cross_modal=2 --fusion_type=none --vpr_model_name=openai/clip-vit-base-patch32 --vpr_dim=512 --is_text_pooling=0 --is_image_pooling=0 --image_size=224 --is_trainable_text_encoder=1 --loss_name=MultiSimilarityLossCM --batch_size=20

python eval_vpr.py --vpr_dim=256 --image_size=384 --vpr_model_name=Salesforce/blip-itm-base-coco --fusion_type=none --cross_modal=2 --is_text_pooling=0 --is_dual_encoder=0 --pca_dim=256 --text_dim=256 --is_trainable_text_encoder=0 --lora_path=LOGS/resnet50/blip_lora_r16_qkv/

python eval_vpr.py --vpr_dim=512 --image_size=224 --vpr_model_name=openai/clip-vit-base-patch32 --fusion_type=none --cross_modal=4 --is_text_pooling=0 --is_dual_encoder=0 --pca_dim=512 --is_trainable_text_encoder=1 --lora_path=LOGS/resnet50/clip_lora_03/

python eval_vpr.py --vpr_dim=768 --image_size=224 --vpr_model_name=google/siglip2-base-patch16-224 --fusion_type=none --cross_modal=2 --is_text_pooling=0 --is_dual_encoder=0 --pca_dim=768 --text_dim=768 --is_trainable_text_encoder=1 --lora_path=LOGS/resnet50/siglip_lora_02/

# text adapter
python eval_vpr.py --vpr_dim=512 --vpr_model_name=mixvpr --fusion_type=text_adapter --is_dual_encoder=0

# supported vpr_model_name: NetVLAD, AP-GeM, SFRS, CosPlace, Conv-AP, MixVPR, EigenPlaces, AnyLoc, SALAD, EigenPlaces-indoor, SALAD-indoor, CricaVPR, CliqueMining, MegaLoc . CricaVPR vpr_dim=10752


