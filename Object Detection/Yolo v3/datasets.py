import os

from torch.utils.data import Dataset
from PIL import Image

class ObjectDataSet(Dataset):
    """
    A custom Dataset class to read input images and (optional) labels.
    """
    def __init__(self, image_folder_path, label_folder_path = None, transform = None, shuffle = False) -> None:
        super().__init__()
        self.image_folder = image_folder_path
        self.label_folder = label_folder_path
        self.transform = transform
        
        self.image_objects =  [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', 'png'))]
        if self.label_folder is not None:
            self.label_objects = [f for f in os.listdir(self.label_folder) if f.endswith(('.txt'))]
            
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_objects[idx])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        if self.label_folder:
            label_path = os.path.join(self.label_folder, self.image_objects[idx].split(".")[0] + ".txt")
            with open(label_path) as f:
                # Don't split the target information into various lines. 
                # Different targets files contain different number of targets. 
                # The loader expects all targets of same size. 
                # Targets could be split in the code downstream.
                lines = f.read()
            return image, lines

        else:
            return image

    def __len__(self):
        if self.label_folder is not None:
            assert len(self.label_objects) == len(self.image_objects)
        
        return len(self.image_objects)
        