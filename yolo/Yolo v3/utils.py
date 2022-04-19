import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

import datasets
def get_image_transform():
    image_transform = transforms.Compose([
            transforms.Resize([416, 416]),
            transforms.ToTensor()])

    return image_transform

def image_to_tensor(image_path):
    """
    Converts the input image to a tensor. Before the output is returned, the tensor is 
    divided by 255 as a file contains values from 0 to 255, but the operations are 
    performed on values from 0 to 1.
    
    """
    image = Image.open(image_path)
    tform = transforms.Compose([transforms.PILToTensor(), transforms.Resize((416, 416))])
    img_tensor = tform(image)
    img_tensor = img_tensor/255
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def get_dataloader(image_folder, label_folder):
    image_transform = get_image_transform()
    train_data = datasets.ObjectDataSet(image_folder, label_folder_path = label_folder, transform=image_transform)
    train_dataloader = DataLoader(train_data, batch_size = 8, shuffle = True)
    return train_dataloader

