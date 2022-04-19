import numpy as np
import torch
from torch import nn as nn
import torchvision.ops as tvo
from torch.utils.data import DataLoader

from datasets import ObjectDataSet
from utils import get_image_transform

import matplotlib.pyplot as plt
    


def get_train_data(image_folder, label_folder):
    image_transform = get_image_transform()
    train_data = ObjectDataSet(image_folder, label_folder_path = label_folder, transform=image_transform)
    train_dataloader = DataLoader(train_data, batch_size = 2, shuffle = True)
    return train_dataloader

def get_eval_data(image_folder, label_folder):
    image_transform = get_image_transform()
    eval_data = ObjectDataSet(image_folder, label_folder_path = label_folder, transform=image_transform)
    eval_dataloader = DataLoader(eval_data, batch_size = 8, shuffle = True)
    return eval_dataloader


train_label_path = "/Users/Jain/code/cloned/ultralytics/coco128_backup/train/labels/train2017"
train_image_path = "/Users/Jain/code/cloned/ultralytics/coco128_backup/train/images/train2017"
train_loader = get_train_data(train_image_path, train_label_path)

eval_label_path = "/Users/Jain/code/cloned/ultralytics/coco128_backup/eval/labels/train2017"
eval_image_path = "/Users/Jain/code/cloned/ultralytics/coco128_backup/eval/images/train2017"
eval_loader = get_eval_data(eval_image_path, eval_label_path)

