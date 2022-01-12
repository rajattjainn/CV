import pandas as pd
from torch.utils.data import Dataset
import torch

class ImageDataDataset (Dataset):
    def __init__(self, input_file_path, transform = None, 
    target_transform = None):
        self.transform = transform
        self.target_transform = target_transform
        self.input_file_df = pd.read_csv(input_file_path)

    def __getitem__(self, idx):
        input_file_df = self.input_file_df
        pixel_data = input_file_df.iloc[idx, 1:len(input_file_df.columns)]
        label = input_file_df.iloc[idx, 0]
        
        if self.transform:
            pixel_data = self.transform(pixel_data)

        if self.target_transform:
            label = self.target_transform(label)

        return torch.tensor(pixel_data), torch.tensor(label)

    def __len__(self):
        return len(self.input_file_df)
