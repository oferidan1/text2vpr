from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os


class csv_dataset(Dataset):
    def __init__(self, csv_file, image_root, transform):
        self.transform = transform
        self.image_path, self.description = read_csv_file(csv_file, image_root)
        print(f'Loaded {len(self.image_path)} samples from {csv_file}')       
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):    
        image = Image.open(self.image_path[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        text = self.description[index]
        return image, text, index
        
def read_csv_file(labels_file, image_root):
    #df = pd.read_csv(labels_file)
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