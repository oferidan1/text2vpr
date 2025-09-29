# text2vpr 
# this repo targets the task of text to visual place recognition (VPR)

python text_to_image_retriever.py --input /mnt/d/dan/datasets/descriptions.csv --database /mnt/d/dan/datasets/sf_xl/processed/test/database/ --output blip_on_night_test_sf_xl.csv --model_type blip --top_k 3 --verbose --device cuda --batch_size 12 --no_remove_duplicates

python dataset_creator_gemini_batch.py --job_name=train-job-6 --result=train-6.csv --start_image=50000


blip eval on queries night
R@1: 1.7, R@5: 2.6, R@10: 4.1, R@20: 6.4