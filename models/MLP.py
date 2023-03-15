import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        h = x
        return self.model(h)
