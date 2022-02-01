import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchinfo import summary

from image_utils import DogsCats
from simple_cnn import SimpleCNN

import logging 


curr_dir = os.path.dirname(os.path.realpath(__file__))
train_data_path = os.path.join(curr_dir, "Data")

logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

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


#ToDo: put the text outside the text, at bottom or somewhere.
def plot_loss(x_data, y_data, xlabel, ylabel, title, plot_text):
    plt.plot(x_data,y_data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    text = plot_text.split(";")
    print ("\n\n")
    print (plot_text)
    print (text)
    print ("\n\n")
    for count in range(len(text)):
        plt.gcf().text(0.02, 0.2 + count * 0.1, text[count], fontsize=8)
    plt.subplots_adjust(left=0.2)
    plt.savefig(title + '_{date:%Y-%m-%d_%H:%M:%S}.png'.format( date=datetime.datetime.now()))
    

def train_model(neural_net, train_loader, lr, momentum, 
            optimiser_type, loss_fn, epochs = 10):
    optimiser = optimiser_type(neural_net.parameters(), 
                lr=lr, momentum = momentum)
    y_data = []
    x_data = []
    for epoch in range(epochs):
        for batch, (pixels, labels) in enumerate(train_loader):
            # summary(neural_net, pixels.size())
            y_preds = neural_net(pixels)
            loss = loss_fn(y_preds, labels)
            y_data.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print (f"batch: {batch}, loss = {loss.item()}")
            logger.debug("batch: " + str(batch) + ", loss = " + str(loss.item()))
            x_data.append((epoch + 1) + (batch + 1))

    plot_text = "LR: " + str(lr) + ";momentum: " + str(momentum) + ";epochs: " + str(epochs) + ";optimiser: " + type(optimiser).__name__ + ";Loss fn: " + type(loss_fn).__name__
    plot_loss(x_data, y_data, xlabel="Total Runs", ylabel="Cost", title = type(neural_net).__name__, plot_text = plot_text)



def train_simple_cnn():
    simple_cnn = SimpleCNN()
    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()
    sgd_optimiser = torch.optim.SGD
    momentum = 0.9
    total_epochs = 10
    train_model(simple_cnn, train_loader, learning_rate, momentum, 
            sgd_optimiser, loss_fn, total_epochs)


train_simple_cnn()