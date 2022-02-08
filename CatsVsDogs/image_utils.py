import os

from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms 

#ToDO: Handle cases when file does not start with .jpg 
class DogsCats(Dataset):
    def __init__(self, input_transform, folder_path, file_list, label = None) -> None:
        super().__init__()
        self.file_list = file_list
        self.label = label
        self.transform = input_transform
        self.folder_path = folder_path

    def __read_input_data__(self, idx):

        image = Image.open(os.path.join(self.folder_path, self.file_list[idx]))
        x_tensor = self.transform(image)

        if self.label != None:
            return x_tensor, torch.tensor(self.label)
        
        if self.label == None:
            return x_tensor, self.file_list[idx].split(".")[0]

    def __getitem__(self, idx):
        return self.__read_input_data__(idx)

    def __len__(self):
        return len(self.file_list)
