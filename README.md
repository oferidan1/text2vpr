# text2vpr 
# this repo targets the task of text to visual place recognition (VPR)

python text_to_image_retriever.py --input /mnt/d/dan/datasets/descriptions.csv --database /mnt/d/dan/datasets/sf_xl/processed/test/database/ --output blip_on_night_test_sf_xl.csv --model_type blip --top_k 3 --verbose --device cuda --batch_size 12 --no_remove_duplicates

python dataset_creator_gemini_batch.py --job_name=train-job-6 --result=train-6.csv --start_image=50000


blip eval on queries night:  R@1: 1.7, R@5: 2.6, R@10: 4.1, R@20: 6.4

amstertime bge text only: 					R@1: 9.4,  R@5: 18.0, R@10: 23.6, R@20: 32.4
amstertime mixvpr orig [512]:      			R@1: 35.7, R@5: 53.0, R@10: 60.4, R@20: 65.8										
amstertime mixvpr orig [512] IP:			R@1: 35.7, R@5: 53.1, R@10: 60.4, R@20: 65.9
amstertime mixvpr_bge_concat [1536]:		R@1: 37.0, R@5: 58.1, R@10: 65.7, R@20: 71.6	
amstertime mixvpr_bge_reranking	a=0.5 [512] R@1: 37.0, R@5: 58.1, R@10: 65.7, R@20: 71.6
amstertime mixvpr_bge_reranking	a=0.6 [512] R@1: 38.3, R@5: 58.5, R@10: 66.0, R@20: 72.8		
amstertime mixvpr_bge_reranking	a=0.65[512] R@1: 38.6, R@5: 58.7, R@10: 65.3, R@20: 73.2
amstertime mixvpr_bge_reranking	a=0.7 [512] R@1: 38.3, R@5: 58.4, R@10: 65.1, R@20: 72.4
amstertime mixvpr_bge_reranking	a=0.75 [512]R@1: 38.2, R@5: 57.6, R@10: 64.6, R@20: 71.2
amstertime mixvpr_bge_reranking	a=0.8 [512]	R@1: 37.9, R@5: 57.0, R@10: 63.5, R@20: 70.5	
amstertime mixvpr_bge_reranking	a=0.9 [512] R@1: 37.0, R@5: 55.2, R@10: 61.9, R@20: 68.5		

amstertime mixvpr orig [4096]:     			R@1: 40.8, R@5: 58.9, R@10: 65.6, R@20: 72.1											
amstertime mixvpr_bge_concat [5120]: 		R@1: 39.3, R@5: 60.0, R@10: 66.6, R@20: 72.4
amstertime mixvpr_bge_reranking	a=0.65[4096]R@1: 38.2, R@5: 57.6, R@10: 64.6, R@20: 71.2
amstertime mixvpr_bge_reranking	a=0.9 [4096]R@1: 43.0, R@5: 62.0, R@10: 68.6, R@20: 75.5
amstertime mixvpr_bge_fusion_mlp [1024]:	R@1: 30.1, R@5: 48.9, R@10: 55.5, R@20: 63.6
amstertime mixvpr_bge_fusion_mlp_res [1024]:R@1: 31.1, R@5: 46.5, R@10: 52.5, R@20: 59.1
amstertime mixvpr_bge_fusion_mlp vpr[1024]: R@1: 31.1, R@5: 48.3, R@10: 55.0, R@20: 61.7

nordland bge text only:                 	R@1: 4.8,  R@5: 11.3, R@10: 15.7, R@20: 21.6
nordland mixvpr orig [512]:        			R@1: 66.6, R@5: 80.8, R@10: 85.7, R@20: 89.9
nordland mixvpr_bge_concat [1536]: 			R@1: 59.7, R@5: 76.3, R@10: 82.3, R@20: 87.4
nordland mixvpr_bge_reranking a=0.6[512]    R@1: 63.3, R@5: 78.2, R@10: 83.5, R@20: 88.0
nordland mixvpr_bge_reranking a=0.65[512]   R@1: 64.6, R@5: 78.9, R@10: 84.1, R@20: 88.6
nordland mixvpr_bge_reranking a=0.7[512]    R@1: 65.2, R@5: 79.4, R@10: 84.4, R@20: 88.9
nordland mixvpr_bge_reranking a=0.8[512]    R@1: 65.9, R@5: 79.6, R@10: 84.7, R@20: 89.0
nordland mixvpr_bge_reranking a=0.9[512]    R@1: 65.9, R@5: 80.0, R@10: 85.1, R@20: 89.4
nordland mixvpr_bge_reranking a=0.92 [512]  R@1: 66.1, R@5: 80.4, R@10: 85.5, R@20: 89.7
nordland mixvpr_bge_reranking a=0.94[512]   R@1: 66.5, R@5: 80.9, R@10: 85.9, R@20: 90.1
nordland mixvpr_bge_reranking a=0.96[512]   R@1: 67.1, R@5: 81.4, R@10: 86.3, R@20: 90.3
nordland mixvpr_bge_reranking a=0.98[512]   R@1: 67.3, R@5: 81.5, R@10: 86.2, R@20: 90.4

nordland mixvpr orig [4096]:       			R@1: 76.4, R@5: 87.1, R@10: 90.6, R@20: 93.6
nordland mixvpr_bge_concat [5120]:			R@1: 66.2, R@5: 80.6, R@10: 85.3, R@20: 89.6
nordland mixvpr_bge_fusion_mlp [1024]:  	R@1: 62.1, R@5: 76.8, R@10: 82.3, R@20: 87.1
nordland mixvpr_bge_fusion_mlp_res [1024]:  R@1: 63.5, R@5: 78.2, R@10: 83.4, R@20: 87.9




