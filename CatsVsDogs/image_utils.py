import os

from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms 

#ToDO: Handle cases when file does not start with .jpg 
class DogsCats(Dataset):
    def __init__(self, folder_path, input_transform) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.transform = input_transform


    def read_input_data(self, folder_path, idx):

        filename = os.listdir(folder_path)[idx]

        if filename.lower().startswith("dog") and filename.endswith("jpg"):
            label = 1
        elif filename.lower().startswith("cat") and filename.endswith("jpg"):
            label = 0
        else:
            label = -1
            
        if (label == 1 or label == 0):
            image = Image.open(os.path.join(folder_path, filename))
            x_tensor = self.transform(image)

        if label == -1:
            print (filename)
            print (folder_path)
            return

        return x_tensor, torch.tensor(label)
        
    def __getitem__(self, idx):
        return self.read_input_data(self.folder_path, idx)

    def __len__(self):
        return len(os.listdir(self.folder_path))


class DogsCatsPredict(Dataset):
    def __init__(self, folder_path, input_transform) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.transform = input_transform

    def read_input_data(self, folder_path, idx):

        filename = os.listdir(folder_path)[idx]
        if filename.endswith("jpg"):
            image = Image.open(os.path.join(folder_path, filename))
            x_tensor = self.transform(image)
        else:
            print (folder_path)
        return x_tensor, filename
    
       
    def __getitem__(self, idx):
        return self.read_input_data(self.folder_path, idx)

    def __len__(self):
        return len(os.listdir(self.folder_path))
    