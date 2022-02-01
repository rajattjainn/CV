from torch import nn
import torch


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 2),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 2),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 2),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 120, out_channels = 180, kernel_size = 5, stride = 2))

        self.classifier = nn.Sequential(
            nn.Linear(in_features = 4320, out_features = 1200),
            nn.ReLU(),
            nn.Linear(in_features = 1200, out_features = 600),
            nn.ReLU(),
            nn.Linear(in_features = 600, out_features = 84),
            nn.ReLU(),
            nn.Linear(in_features = 84, out_features = 3)

        )

    def forward(self,x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = nn.Functional.Softmax(logits, dim=1)
        return logits, probs