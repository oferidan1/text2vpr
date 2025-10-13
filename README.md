# text2vpr 
# this repo targets the task of text to visual place recognition (VPR)

python text_to_image_retriever.py --input /mnt/d/dan/datasets/descriptions.csv --database /mnt/d/dan/datasets/sf_xl/processed/test/database/ --output blip_on_night_test_sf_xl.csv --model_type blip --top_k 3 --verbose --device cuda --batch_size 12 --no_remove_duplicates

python dataset_creator_gemini_batch.py --job_name=train-job-6 --result=train-6.csv --start_image=50000


blip eval on queries night:  R@1: 1.7, R@5: 2.6, R@10: 4.1, R@20: 6.4

amstertime mixvpr orig 512:      		R@1: 35.7, R@5: 53.0, R@10: 60.4, R@20: 65.8
amstertime mixvpr orig 512 IP:			R@1: 35.7, R@5: 53.1, R@10: 60.4, R@20: 65.9
amstertime mixvpr_bge_cat 1536:  		R@1: 37.0, R@5: 58.1, R@10: 65.7, R@20: 71.6

amstertime mixvpr_bge_each 1:1 			R@1: 1.9, R@5: 8.2,  R@10: 18.4, R@20: 36.6
amstertime mixvpr_bge_each 0.1/0.8 		R@1: 3.0, R@5: 14.2, R@10: 27.9, R@20: 56.6 

amstertime mixvpr orig 4096:     		R@1: 40.8, R@5: 58.9, R@10: 65.6, R@20: 72.1
amstertime mixvpr_bge_cat 5120:  		R@1: 39.3, R@5: 60.0, R@10: 66.6, R@20: 72.4

nordland mixvpr orig 512:        		R@1: 66.6, R@5: 80.8, R@10: 85.7, R@20: 89.9
nordland mixvpr_bge_cat 1536:    		R@1: 59.7, R@5: 76.3, R@10: 82.3, R@20: 87.4
nordland mixvpr orig 4096:       		R@1: 76.4, R@5: 87.1, R@10: 90.6, R@20: 93.6
nordland mixvpr_bge_cat 5120:    		R@1: 66.2, R@5: 80.6, R@10: 85.3, R@20: 89.6