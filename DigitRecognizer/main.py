import torch
from torch import nn

import load_data as ld
from simple_neural_net import SimpleNeuralNet


def train_loop(optimizer, loss_fn, model):
    X_train, y_train = ld.read_train_data()
    X_train = X_train.float()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print (f"train loss: {loss.item()}")


def test_loop(model, loss_fn):
    with torch.no_grad():
        X_test, y_test = ld.read_test_data()
        X_test = X_test.float()
        y_pred = model(X_test)
        loss = loss_fn(y_pred, y_test)
        print (f"test loss: {loss}")


simpleModel = SimpleNeuralNet()

learning_rate = 1e-2
epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(simpleModel.parameters(), lr = learning_rate)

for t in range (0, epochs):
    print (f" \nEpoch {t}")
    train_loop(optimizer, loss_fn, simpleModel)