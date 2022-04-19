import numpy as np
import torch
from torch import nn as nn
import torchvision.ops as tvo
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import ObjectDataSet

import matplotlib.pyplot as plt
    

def get_train_data(image_folder_path, label_folder_path):
    train_data = ObjectDataSet(image_folder_path, label_folder_path)
    train_dataloader = DataLoader(train_data, batch_size = 8, shuffle = True)
    return train_dataloader
        
label_path = "/Users/Jain/code/cloned/ultralytics/coco128_backup/labels/train2017"
image_path = "/Users/Jain/code/cloned/ultralytics/coco128_backup/images/train2017"

train_loader = get_train_data(image_path, label_path)
