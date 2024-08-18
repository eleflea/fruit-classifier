import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        input_features = 3 * cfg.DATASET.SIZE * cfg.DATASET.SIZE
        self.fc_layers = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x
