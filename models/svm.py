import torch.nn as nn

from config import cfg

def gaussian_kernel(x, y, sigma=450):
    xs = x.pow(2).sum(-1)
    ys = y.pow(2).sum(-1)
    distances = (xs.unsqueeze(-1) + ys.unsqueeze(0) - 2 * x @ y.t())
    distances = distances.div(-sigma * sigma).exp()
    return distances


class SVM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        input_features = 3 * cfg.DATASET.SIZE * cfg.DATASET.SIZE
        self.fc = nn.Linear(input_features, num_classes)
    
        # self.fc = nn.Linear(cfg.MODEL.X_SIZE, num_classes)
    
    def set_x(self, data):
        self.x_data = self.register_buffer("xdata", data)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        # if getattr(self, 'xdata') is not None:
        #     y = self.get_buffer('xdata').flatten(start_dim=1)
        #     x = gaussian_kernel(x, y)
        return self.fc(x)
