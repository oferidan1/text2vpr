# text2vpr 
# this repo targets the task of text to visual place recognition (VPR)

python text_to_image_retriever.py --input /mnt/d/dan/datasets/descriptions.csv --database /mnt/d/dan/datasets/sf_xl/processed/test/database/ --output blip_on_night_test_sf_xl.csv --model_type blip --top_k 3 --verbose --device cuda --batch_size 12 --no_remove_duplicates