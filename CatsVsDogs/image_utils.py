import os

from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms 

class DogsCats(Dataset):
    def __init__(self, folder_path, input_transform) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.transform = input_transform
        self.pixel_data, self.label_data = self.read_input_data(folder_path)

    def read_input_data(self, folder_path):
        pixel_data = []
        label_data = []

        for filename in os.listdir(folder_path):

            if filename.lower().startswith("dog") and filename.endswith("jpg"):
                label = 1
            elif filename.lower().startswith("cat") and filename.endswith("jpg"):
                label = 0
            else:
                label = -1
            
            if (label == 1 or label == 0):
                image = Image.open(os.path.join(folder_path, filename))
                x_tensor = self.transform(image)

                pixel_data.append(x_tensor)
                label_data.append(torch.tensor(label))


        return pixel_data, label_data     
        
    def __getitem__(self, idx):
        return self.pixel_data[idx], self.label_data[idx]

    def __len__(self):
        assert len(self.pixel_data) == len(self.label_data)
        return len(self.pixel_data)