import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, padding='same')
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.elu(x)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class EEGNetAdvanced(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=0),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        self.depthwise = DepthwiseSeparableConv(16, 32, (16, 1))
        self.attention = SpatialAttention(32)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.depthwise(x)
        x = self.attention(x)
        x = self.pool(x).squeeze()
        x = self.dropout(x)
        return self.classifier(x)
