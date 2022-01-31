from torch import nn
import torch


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding = 'same'),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 'same'),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1, padding = 'same'))

        self.classifier = nn.Sequential(
            nn.Linear(in_features = 7680, out_features = 120),
            nn.ReLU(),
            nn.Linear(in_features = 120, out_features = 84),
            nn.ReLU(),
            nn.Linear(in_features = 84, out_features = 3)
        )

    def forward(self,x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits