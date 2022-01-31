import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchinfo import summary

from image_utils import DogsCats
from simple_cnn import SimpleCNN

curr_dir = os.path.dirname(os.path.realpath(__file__))
train_data_path = os.path.join(curr_dir, "Data")

"""
The resize dimensions 768x1050 has been arrived at by iterating
over all the input images (corresponding to both test and train
data) and finding the maximum dimensions of an image."""

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([768, 1050])
])

train_data = DogsCats(train_data_path, input_transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

simple_cnn = SimpleCNN()
batch_size = 64

learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
sgd_optimiser = torch.optim.SGD(simple_cnn.parameters(), 
                lr=learning_rate, momentum = 0.9)

for batch, (pixels, labels) in enumerate(train_loader):
    summary(simple_cnn, pixels.size())
    
    y_preds = simple_cnn(pixels)
    loss = loss_fn(y_preds, labels)
    sgd_optimiser.zero_grad()
    loss.backward()
    sgd_optimiser.step()
    print (f"batch: {batch}, loss = {loss.item()}")

# def train_model():
