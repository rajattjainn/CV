import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

import load_data as ld
from simple_neural_net import SimpleNeuralNet
from image_dataset import ImageDataDataset

def train_loop(optimizer, loss_fn, model, train_dataloader):
    value = True
    for batch, (pixels, labels) in enumerate(train_dataloader):
        pixels = pixels.float()
        y_pred = model(pixels)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print (f"count: {batch * len(pixels)}, train loss: {loss.item()}")



simpleModel = SimpleNeuralNet()

learning_rate = 1e-2
epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(simpleModel.parameters(), lr = learning_rate)

curr_path = os.path.dirname(os.path.realpath(__file__))
train_data_path = os.path.join(curr_path, "Data", "train.csv")
train_dataset = ImageDataDataset(train_data_path)
train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle=True)


def predict_on_test(model, test_data):
    y_pred = model(test_data).argmax(1)
    pred_df = pd.DataFrame(y_pred)
    image_id_series = pd.Series(range(1, len(test_data) + 1))
    output_df = image_id_series.to_frame().merge(pred_df, left_index=True, right_index=True)
    output_df.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Data", "prediction_data.csv"))

for t in range (0, epochs):
    print (f" \nEpoch {t}")
    train_loop(optimizer, loss_fn, simpleModel, train_dataloader)

test_data_path = os.path.join(curr_path, "Data", "test.csv")
test_df = pd.read_csv(test_data_path)

simpleModel.load_state_dict(torch.load('simplemodel_weights.pth'))
simpleModel.eval()

predict_on_test(simpleModel, torch.tensor(test_df.values).float())