from torch import nn

class SimpleNeuralNet (nn.Module):
    def __init__(self):
        super (SimpleNeuralNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

