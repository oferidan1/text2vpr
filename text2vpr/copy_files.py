import os
import pandas as pd

root_src = '/mnt/d/data/amstertime/test'
files_csv = 'amstertime_w_text.csv'
dst_folder = '/mnt/d/data/amstertime/test_with_text'

df = pd.read_csv(os.path.join(root_src, files_csv))
#loop over rows and copy files to new folder
if not df.empty:
    db_folder = os.path.join(dst_folder, 'database')
    queries_folder =  os.path.join(dst_folder, 'queries')
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        os.makedirs(db_folder)
        os.makedirs(queries_folder)
        
    for idx, row in df.iterrows():
        src_image_path = os.path.join(root_src, row['image_path'])
        if 'queries' in row['image_path']:
            dst_image_path = os.path.join(queries_folder, os.path.basename(row['image_path']))
            os.system(f'cp "{src_image_path}" "{dst_image_path}"')
        # if 'database' in row['image_path']:
        #     query_name = row['image_path'].replace('database', 'queries')
        #     #check if query_name exists in df
        #     if query_name not in df['image_path'].values:
        #         continue            
        #     dst_image_path = os.path.join(db_folder, os.path.basename(row['image_path']))
        # else:
        #     #queries
        #     db_name = row['image_path'].replace('queries', 'database')
        #     #check if query_name exists in df
        #     if db_name not in df['image_path'].values:
        #         continue            
        #     dst_image_path = os.path.join(queries_folder, os.path.basename(row['image_path']))
        # os.system(f'cp "{src_image_path}" "{dst_image_path}"')
