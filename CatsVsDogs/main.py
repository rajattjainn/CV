import os
import datetime

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchinfo import summary

from image_utils import DogsCats, DogsCatsPredict
from simple_cnn import SimpleCNN

import logging 

curr_dir = os.path.dirname(os.path.realpath(__file__))
# train_data_path = os.path.join(curr_dir, "Data", "train")
# validate_data_path = os.path.join(curr_dir, "Data", "validate")
# predict_data_path = os.path.join(curr_dir, "Data", "predict")


train_data_path = "/home/azureuser/cloudfiles/code/Users/addresseerajat/learnML/CatsVsDogs/Data/Data/train"
validate_data_path = "/home/azureuser/cloudfiles/code/Users/addresseerajat/learnML/CatsVsDogs/Data/validate"
predict_data_path = "/home/azureuser/cloudfiles/code/Users/addresseerajat/learnML/CatsVsDogs/Data/test/test1"

print ("initialized path")
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

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([50, 50])
])

train_data = DogsCats(train_data_path, data_transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=32)
print ("train dataloader")

validate_data = DogsCats(validate_data_path, data_transform)
validate_loader = DataLoader(validate_data, shuffle=True, batch_size=32)
print ("validate dataloader")

predict_data = DogsCatsPredict(predict_data_path, data_transform)
predict_loader = DataLoader(predict_data, shuffle=True, batch_size=32)

print ("initialised loaders")

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
        summary(neural_net, pixels.size())
        print (pixels.size())
        print (type(pixels))
        print ("batch: " + str(batch))

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
    for batch, (pixels, labels) in enumerate(validate_loader):
        y_preds, _ = neural_net(pixels)
        loss = loss_fn(y_preds, labels)
        running_loss = running_loss + (loss.item() * labels.size(0))
        print ("validate batch: " + str(batch))

    epoch_loss = running_loss/len(validate_loader)

    return neural_net, epoch_loss


def get_accuracy(neural_net, data_loader):
    correct_preds = 0

    for batch, (pixels, labels) in enumerate(data_loader):
        _, y_probs = neural_net(pixels)
        _, predicted_label = torch.max(y_probs, 1)
        correct_preds += (predicted_label == labels).sum()
        print ("accuracy batch: " + str(batch))
    
    total_accuracy = correct_preds/(len(data_loader))
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
        print ("epoch: " + str(epoch))
        neural_net, train_epoch_loss = train_model(simple_cnn, train_loader, learning_rate, momentum, 
            sgd_optimiser, loss_fn)
        
        with torch.no_grad():
            neural_net, validate_epoch_loss = validate_model(neural_net, validate_loader, loss_fn)

        accuracy  = get_accuracy(neural_net, validate_loader)

        print (f"Epoch: {epoch}, Train Loss: {train_epoch_loss}, Validate Loss: {validate_epoch_loss}, Training Accuracy: {accuracy}")

        validation_loss.append(validate_epoch_loss)
        training_loss.append(train_epoch_loss)

    plot_text = "LR: " + str(learning_rate) + ";momentum: " + str(momentum) + ";epochs: " + str(total_epochs) + ";optimiser: " + type(sgd_optimiser).__name__ + ";Loss fn: " + type(loss_fn).__name__
    plot_loss(training_loss, validation_loss, total_epochs, xlabel="Total Runs", ylabel="Cost", title = type(neural_net).__name__ + " Train", plot_text = plot_text)

    return neural_net


def predict_output(neural_net, predict_loader):
    data = {'id':[],
        'label':[]}
    for _, (x_pixels, filename) in enumerate(predict_loader):
        _, pred_probs = neural_net(x_pixels)
        _, predicted_label = torch.max(pred_probs, 1)
        data["id"].extend(filename)
        data["label"].extend(predicted_label.numpy())
    
    df = pd.DataFrame(data)
    df.to_csv("output.csv",index=False)

trained_net = train_simple_cnn()
print ("defined the net, starting with training")
torch.save(trained_net.state_dict(), 'simple_cnn.pth')
predict_output(trained_net, predict_loader)



#ToDo: check accuracy and other output variables for correctness. 
