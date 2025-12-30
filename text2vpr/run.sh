#!/bin/bash
TEXT_ENCODER=BAAI/bge-large-en-v1.5
TEXT_ENCODER=lightonai/modernbert-embed-large
# TEXT_ENCODER=Qwen/Qwen3-Embedding-8B
BFLOAT16=0
python eval_vpr.py --encode_mode=text --fusion_type=none --is_dual_encoder=0 --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=32768 --vpr_model_name=netvlad --encode_mode=image --fusion_type=none --is_dual_encoder=0 --is_pca=0
python eval_vpr.py --vpr_dim=32768 --vpr_model_name=netvlad --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=32768 --vpr_model_name=netvlad --is_dual_encoder=1 --dual_encoder_fusion=each --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=512 --vpr_model_name=mixvpr --encode_mode=image --fusion_type=none --is_dual_encoder=0 --is_pca=0
python eval_vpr.py --vpr_dim=512 --vpr_model_name=mixvpr --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=512 --vpr_model_name=mixvpr --is_dual_encoder=1 --dual_encoder_fusion=each --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=4096 --vpr_model_name=mixvpr --encode_mode=image --fusion_type=none --is_dual_encoder=0 --is_pca=0
python eval_vpr.py --vpr_dim=4096 --vpr_model_name=mixvpr --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=4096 --vpr_model_name=mixvpr --is_dual_encoder=1 --dual_encoder_fusion=each --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=512 --vpr_model_name=cosplace --encode_mode=image --fusion_type=none --is_dual_encoder=0 --is_pca=0
python eval_vpr.py --vpr_dim=512 --vpr_model_name=cosplace --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=512 --vpr_model_name=cosplace --is_dual_encoder=1 --dual_encoder_fusion=each --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=2048 --vpr_model_name=eigenplaces --encode_mode=image --fusion_type=none --is_dual_encoder=0 --is_pca=0
python eval_vpr.py --vpr_dim=2048 --vpr_model_name=eigenplaces --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=2048 --vpr_model_name=eigenplaces --is_dual_encoder=1 --dual_encoder_fusion=each --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=8448 --vpr_model_name=salad --encode_mode=image --fusion_type=none --is_dual_encoder=0 --is_pca=0
python eval_vpr.py --vpr_dim=8448 --vpr_model_name=salad --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=8448 --vpr_model_name=salad --is_dual_encoder=1 --dual_encoder_fusion=each --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=10752 --vpr_model_name=cricavpr --encode_mode=image --fusion_type=none --is_dual_encoder=0 --is_pca=0 
python eval_vpr.py --vpr_dim=10752 --vpr_model_name=cricavpr --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
python eval_vpr.py --vpr_dim=10752 --vpr_model_name=cricavpr --is_dual_encoder=1 --dual_encoder_fusion=each --fusion_type=none --is_pca=0 --text_model_name="$TEXT_ENCODER" --bfloat="$BFLOAT16"
# python eval_vpr.py --vpr_dim=10752 --vpr_model_name=cricavpr --encode_mode=image --fusion_type=none --is_dual_encoder=0 --is_pca=1 --pca_dim=4096
# python eval_vpr.py --vpr_dim=10752 --vpr_model_name=cricavpr --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none --is_pca=1 --pca_dim=4096

# #python eval_vpr.py --vpr_dim=8448 --vpr_model_name=salad --encode_mode=image --fusion_type=none --is_dual_encoder=0 --is_pca=0 --text_model_name=lightonai/modernbert-embed-large
# python eval_vpr.py --vpr_dim=8448 --vpr_model_name=salad --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none --is_pca=0 --text_model_name=Qwen/Qwen3-Embedding-8B --text_dim=4096
# #python eval_vpr.py --vpr_dim=10752 --vpr_model_name=cricavpr --encode_mode=image --fusion_type=none --is_dual_encoder=0 --is_pca=0 
# python eval_vpr.py --vpr_dim=10752 --vpr_model_name=cricavpr --is_dual_encoder=1 --dual_encoder_fusion=cat --fusion_type=none --is_pca=0 --text_model_name=Qwen/Qwen3-Embedding-8B --text_dim=4096