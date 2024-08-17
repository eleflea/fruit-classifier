import torch.nn as nn

from config import cfg

class SVM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        input_features = 3 * cfg.DATASET.SIZE * cfg.DATASET.SIZE
        self.fc = nn.Linear(input_features, num_classes)
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.fc(x)
