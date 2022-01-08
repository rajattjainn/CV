import os

from torch.utils.data import DataLoader
import torch
import pandas as pd


curr_path = os.path.dirname(os.path.realpath(__file__))
train_data = os.path.join(curr_path, "Data", "train.csv")
test_data = os.path.join(curr_path, "Data" , "test.csv")

def read_train_data():
    train_df = pd.read_csv(train_data)
    y_df = train_df.iloc[:,0]
    y_tensor = torch.tensor(y_df)
    x_df = train_df.iloc[:, 1:len(train_df.columns)]
    x_tensor = torch.tensor(x_df.values)
    return x_tensor, y_tensor

# def read_test_data():
#     test_df = pd.read_csv(test_data)
#     y_df = test_df.iloc[:,0]
#     y_tensor = torch.tensor(y_df)
#     x_df = test_df.iloc[:, 1:len(test_df.columns)]
#     x_tensor = torch.tensor(x_df.values)
#     return x_tensor, y_tensor


#ToDo: convert these functions to return a Dataset instead of tensors