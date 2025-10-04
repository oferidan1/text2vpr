
import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import datasets.dataset_utils as dataset_utils


class TestDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="database",
                 queries_folder="queries", positive_dist_threshold=25,
                 image_size=512, resize_test_imgs=False, labels_csv=None, image_root=None):
        self.database_folder = dataset_folder + "/" + database_folder
        self.queries_folder = dataset_folder + "/" + queries_folder
        self.database_paths = dataset_utils.read_images_paths(self.database_folder, get_abs_path=True)
        self.queries_paths = dataset_utils.read_images_paths(self.queries_folder, get_abs_path=True)
        
        self.image_path, self.description = TestDataset.read_csv_file(labels_csv, image_root)
        
        self.dataset_name = os.path.basename(dataset_folder)
        
        #### Read paths and UTM coordinates for all images.
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.positives_per_query = knn.radius_neighbors(
            self.queries_utms, radius=positive_dist_threshold, return_distance=False
        )
        
        self.images_paths = self.database_paths + self.queries_paths
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

        transforms_list = []
        if resize_test_imgs:
            # Resize to image_size along the shorter side while maintaining aspect ratio
            transforms_list += [transforms.Resize(image_size, antialias=True)]
        transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        self.base_transform = transforms.Compose(transforms_list)
    
    @staticmethod
    def open_image(path):
        return Image.open(path).convert("RGB")
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = TestDataset.open_image(image_path)
        normalized_img = self.base_transform(pil_img)
        # get the description for this image: find image_path index in self.image_path        
        if image_path not in self.image_path:
            return normalized_img, index, ""  # empty description if not found
        
        desc_index = self.image_path.index(image_path)
        description = self.description[desc_index]          
        return normalized_img, index, description
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"
    
    def get_positives(self):
        return self.positives_per_query
    
    @staticmethod
    def read_csv_file(labels_file, image_root):    
        df = pd.read_csv(labels_file, 
            engine='python',  # Use python engine for better path handling
            encoding='utf-8',
            on_bad_lines='skip',
            quotechar='"',
            skipinitialspace=True)
        image_path = df['image_path'].values
        description = df['description'].values    
        image_path = [os.path.join(image_root, p) for p in image_path]
        return image_path, description
        
