import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

current_path = os.path.dirname(os.path.realpath(__file__))
train_data = os.path.join(current_path, "Data", "test.csv")
test_data = os.path.join(current_path, "Data", "test.csv")


class TitanicTestData (Dataset):
    def __init__(self, file_path, transform = None, target_transoform = None):
        print ("abc")
        self.test_df = pd.read_csv(file_path)
        self.transform = transform
        self.target_transform = target_transoform

    def __getitem__(self, idx):
        test_df = self.test_df
        y = test_df.loc[idx, 'Survived']
        x = test_df.loc[0, test_df.columns != 'Survived']

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)
    
        x_tensor = torch.tensor(x)
        y_tensor = torch.tensor(y)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.test_file_df)

def get_test_dataloader(batch_size = 64, shuffle = True):
    titanic_test_data = TitanicTestData(train_data)
    titanic_data_loader = DataLoader(titanic_test_data, batch_size, shuffle)
    return titanic_data_loader