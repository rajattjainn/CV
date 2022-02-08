import os
import datetime

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchinfo import summary

from image_utils import DogsCats
from simple_cnn import SimpleCNN

import logging 

curr_dir = os.path.dirname(os.path.realpath(__file__))
train_data_path = os.path.join(curr_dir, "Data", "demo_train")
validate_data_path = os.path.join(curr_dir, "Data", "demo_validate")
predict_data_path = os.path.join(curr_dir, "Data", "demo_predict")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# train_data_path = "/home/azureuser/cloudfiles/code/Users/addresseerajat/code/learnML/CatsVsDogs/Data/train"
# validate_data_path = "/home/azureuser/cloudfiles/code/Users/addresseerajat/code/learnML/CatsVsDogs/Data/validate"
# predict_data_path = "/home/azureuser/cloudfiles/code/Users/addresseerajat/code/learnML/CatsVsDogs/Data/test1"

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
    transforms.Resize([50,50]),
    transforms.ToTensor(),
])

train_files = os.listdir(train_data_path)
dog_train_files = [tf for tf in train_files if tf.startswith('dog')]
cat_train_files = [tf for tf in train_files if tf.startswith('cat')]
dog_train_data =  DogsCats(data_transform, train_data_path, dog_train_files, 1)
cat_train_data =  DogsCats(data_transform, train_data_path, cat_train_files, 0)
train_data = ConcatDataset([dog_train_data, cat_train_data])
train_loader = DataLoader(train_data, shuffle=True, batch_size=32)
print ("train dataloader")


validate_files = os.listdir(validate_data_path)
dog_valid_files = [tf for tf in validate_files if tf.startswith('dog')]
cat_valid_files = [tf for tf in validate_files if tf.startswith('cat')]
dog_validate_data =  DogsCats(data_transform, validate_data_path, dog_valid_files, 1)
cat_validate_data =  DogsCats(data_transform, validate_data_path, cat_valid_files, 0)
validate_data = ConcatDataset([dog_validate_data, cat_validate_data])
validate_loader = DataLoader(validate_data, shuffle=True, batch_size=32)
print ("validate dataloader")

predict_files = os.listdir(predict_data_path)
predict_files = [tf for tf in predict_files if tf.endswith("jpg")]
predict_data =  DogsCats(data_transform, predict_data_path, predict_files, None)
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
            optimiser_type, loss_fn, DEVICE):
    
    optimiser = optimiser_type(neural_net.parameters(), 
                lr=lr, momentum = momentum)
    neural_net.train()
    running_loss = 0

    for batch, (pixels, labels) in enumerate(train_loader):
        # summary(neural_net, pixels.size())
        pixels.to(DEVICE)
        labels.to(DEVICE)

        y_preds, _ = neural_net(pixels)
        loss = loss_fn(y_preds, labels)
        running_loss = running_loss + (loss.item() * labels.size(0))
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print (f"train batch: {batch}, loss = {loss.item()}")
        logger.debug("batch: " + str(batch) + ", loss = " + str(loss.item()))
        
    epoch_loss = running_loss/len(train_loader)

    # plot_text = "LR: " + str(lr) + ";momentum: " + str(momentum) + ";epochs: " + str(epochs) + ";optimiser: " + type(optimiser).__name__ + ";Loss fn: " + type(loss_fn).__name__
    # plot_loss(x_data, y_data, xlabel="Total Runs", ylabel="Cost", title = type(neural_net).__name__ + " Train", plot_text = plot_text)

    return neural_net, epoch_loss


def validate_model(neural_net, validate_loader, loss_fn, DEVICE):
    neural_net.eval()
    running_loss = 0
    for batch, (pixels, labels) in enumerate(validate_loader):
        pixels.to(DEVICE)
        labels.to(DEVICE)
        y_preds, _ = neural_net(pixels)
        loss = loss_fn(y_preds, labels)
        running_loss = running_loss + (loss.item() * labels.size(0))
        print (f"validate batch: {batch}, loss = {loss.item()}")

    epoch_loss = running_loss/len(validate_loader)

    return neural_net, epoch_loss


def get_accuracy(neural_net, data_loader, DEVICE):
    correct_preds = 0

    for batch, (pixels, labels) in enumerate(data_loader):
        pixels.to(DEVICE)
        labels.to(DEVICE)
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
            sgd_optimiser, loss_fn, DEVICE)
        
        with torch.no_grad():
            neural_net, validate_epoch_loss = validate_model(neural_net, validate_loader, loss_fn, DEVICE)

        accuracy  = get_accuracy(neural_net, validate_loader, DEVICE)

        print (f"Epoch: {epoch}, Train Loss: {train_epoch_loss}, Validate Loss: {validate_epoch_loss}, Training Accuracy: {accuracy}")

        validation_loss.append(validate_epoch_loss)
        training_loss.append(train_epoch_loss)

    plot_text = "LR: " + str(learning_rate) + ";momentum: " + str(momentum) + ";epochs: " + str(total_epochs) + ";optimiser: " + type(sgd_optimiser).__name__ + ";Loss fn: " + type(loss_fn).__name__
    plot_loss(training_loss, validation_loss, total_epochs, xlabel="Total Runs", ylabel="Cost", title = type(neural_net).__name__ + " Train", plot_text = plot_text)

    return neural_net


def predict_output(neural_net, predict_loader, DEVICE):
    data = {'id':[],
        'label':[]}
    for _, (x_pixels, filename) in enumerate(predict_loader):
        x_pixels.to(DEVICE)
        _, pred_probs = neural_net(x_pixels)
        _, predicted_label = torch.max(pred_probs, 1)
        data["id"].extend(filename)
        data["label"].extend(predicted_label.numpy())
    
    df = pd.DataFrame(data)
    df.to_csv("output.csv",index=False)

trained_net = train_simple_cnn()
print ("defined the net, starting with training")
torch.save(trained_net.state_dict(), 'simple_cnn.pth')
predict_output(trained_net, predict_loader, DEVICE)



#ToDo: check accuracy and other output variables for correctness. 
