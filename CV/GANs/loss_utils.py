import torch.nn as nn


def get_bce_loss():
    """Return a torch.nn.BCELoss object"""
    return nn.BCELoss()