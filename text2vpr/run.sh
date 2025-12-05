#!/bin/bash
python eval_vpr.py --is_dual_encoder=1 --dual_encoder_fusion=each --vision_dimension=512 --alpha_vision=0.5
python eval_vpr.py --is_dual_encoder=1 --dual_encoder_fusion=each --vision_dimension=512 --alpha_vision=0.6
python eval_vpr.py --is_dual_encoder=1 --dual_encoder_fusion=each --vision_dimension=512 --alpha_vision=0.65
python eval_vpr.py --is_dual_encoder=1 --dual_encoder_fusion=each --vision_dimension=512 --alpha_vision=0.7
python eval_vpr.py --is_dual_encoder=1 --dual_encoder_fusion=each --vision_dimension=512 --alpha_vision=0.75