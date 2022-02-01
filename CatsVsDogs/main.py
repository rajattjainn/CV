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
def plot_loss(y1_data, y2_data, x_data, xlabel, ylabel, title, plot_text):
    plt.plot(y1_data, color='blue', label='Training loss')
    plt.plot(y2_data, color='red', label='Validation loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    text = plot_text.split(";")
    for count in range(len(text)):
        plt.gcf().text(0.02, 0.2 + count * 0.1, text[count], fontsize=8)
    plt.subplots_adjust(left=0.2)
    plt.savefig(title + '_{date:%Y-%m-%d_%H:%M:%S}.png'.format(date=datetime.datetime.now()))
    

def train_model(neural_net, train_loader, lr, momentum, 
            optimiser_type, loss_fn):
    
    optimiser = optimiser_type(neural_net.parameters(), 
                lr=lr, momentum = momentum)
    neural_net.train()
    running_loss = 0

    for batch, (pixels, labels) in enumerate(train_loader):
            # summary(neural_net, pixels.size())

        y_preds, _ = neural_net(pixels)
        loss = loss_fn(y_preds, labels)
        running_loss = running_loss + (loss.item() * labels.size(0))
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print (f"batch: {batch}, loss = {loss.item()}")
        logger.debug("batch: " + str(batch) + ", loss = " + str(loss.item()))
        
    epoch_loss = running_loss/len(train_loader)

    # plot_text = "LR: " + str(lr) + ";momentum: " + str(momentum) + ";epochs: " + str(epochs) + ";optimiser: " + type(optimiser).__name__ + ";Loss fn: " + type(loss_fn).__name__
    # plot_loss(x_data, y_data, xlabel="Total Runs", ylabel="Cost", title = type(neural_net).__name__ + " Train", plot_text = plot_text)

    return neural_net, epoch_loss


def validate_model(neural_net, validate_loader, loss_fn):
    neural_net.eval()
    running_loss = 0
    for _, (pixels, labels) in enumerate(validate_loader):
        y_preds, _ = neural_net(pixels)
        loss = loss_fn(y_preds, labels)
        running_loss = running_loss + (loss.item() * labels.size(0))
    
    epoch_loss = running_loss/len(validate_loader)
    # plot_text = "epochs: " + str(epochs) + ";Loss fn: " + type(loss_fn).__name__
    # plot_loss(x_data, y_data, xlabel="Total Runs", ylabel="Cost", title = type(neural_net).__name__+ " Validate", plot_text = plot_text)

    return neural_net, epoch_loss


def get_accuracy(neural_net, data_loader):
    correct_preds = 0

    for _, (pixels, labels) in enumerate(data_loader):
        _, y_probs = neural_net(pixels)
        predicted_label = torch.max(y_probs, 1)
        correct_preds = correct_preds + (predicted_label == labels).sum(0)
    total_accuracy = correct_preds/(len(data_loader)) * 100
    return total_accuracy


def train_simple_cnn():
    simple_cnn = SimpleCNN()
    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()
    sgd_optimiser = torch.optim.SGD
    momentum = 0.9
    total_epochs = 10

    validation_loss = []
    training_loss = []

    for epoch in range(total_epochs):
        neural_net, train_epoch_loss = train_model(simple_cnn, train_loader, learning_rate, momentum, 
            sgd_optimiser, loss_fn, total_epochs)
        
        neural_net, validate_epoch_loss = validate_model(neural_net, validate_loader, loss_fn)

        accuracy  = get_accuracy(neural_net, validate_loader)

        print (f"Epoch: {epoch}, Train Loss: {train_epoch_loss}, Validate Loss: {validate_epoch_loss}, Training Accuracy: {accuracy}")

        validation_loss.append(validate_epoch_loss)
        training_loss.append(train_epoch_loss)

    plot_text = "LR: " + str(learning_rate) + ";momentum: " + str(momentum) + ";epochs: " + str(total_epochs) + ";optimiser: " + type(sgd_optimiser).__name__ + ";Loss fn: " + type(loss_fn).__name__
    plot_loss(training_loss, validation_loss, total_epochs, xlabel="Total Runs", ylabel="Cost", title = type(neural_net).__name__ + " Train", plot_text = plot_text)

    train_model(simple_cnn, train_loader, learning_rate, momentum, 
            sgd_optimiser, loss_fn, total_epochs)


train_simple_cnn()