import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBNReLU(3, 12, 5, stride=2, padding=2),
            ConvBNReLU(12, 32, 3, stride=2, padding=1),
            ConvBNReLU(32, 32, 3, stride=1, padding=1),
            ConvBNReLU(32, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 128, 3, stride=2, padding=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.avg_pool(x).flatten(start_dim=1)
        x = self.fc(x)
        return x
