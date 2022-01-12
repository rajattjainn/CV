import os
import torch
from torch import nn

import load_data as ld
from simple_neural_net import SimpleNeuralNet
from torch.utils.data import DataLoader

from image_dataset import ImageDataDataset

def train_loop(optimizer, loss_fn, model, train_dataloader):
    iter = 1
    for batch, (pixels, labels) in enumerate(train_dataloader):
        pixels = pixels.float()
        y_pred = model(pixels)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print (f"count: {iter}, train loss: {loss.item()}")
        iter += 1


# def test_loop(model, loss_fn, test_data):
#     with torch.no_grad():
#         X_test, y_test = test_data
#         X_test = X_test.float()
#         y_pred = model(X_test)
#         loss = loss_fn(y_pred, y_test)
#         print (f"test loss: {loss}")


simpleModel = SimpleNeuralNet()

learning_rate = 1e-2
epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(simpleModel.parameters(), lr = learning_rate)

curr_path = os.path.dirname(os.path.realpath(__file__))
train_data_path = os.path.join(curr_path, "Data", "train.csv")
train_dataset = ImageDataDataset(train_data_path)
train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle=True)


# test_data_path = os.path.join(curr_path, "Data", "test.csv")
# test_dataset = ImageDataDataset(test_data_path)
# test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=True)

train_data = ld.read_train_data()
# test_data = ld.read_test_data()

for t in range (0, epochs):
    print (f" \nEpoch {t}")
    train_loop(optimizer, loss_fn, simpleModel, train_dataloader)

