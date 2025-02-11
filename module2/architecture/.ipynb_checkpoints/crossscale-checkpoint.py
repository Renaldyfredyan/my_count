import torch
from torch import nn
from torch.nn import functional as F

class CrossScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossScaleFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_low, x_high):
        x_low = self.conv1(x_low)
        x_low = self.bn1(x_low)
        x_low = self.relu(x_low)

        x_high = F.interpolate(x_high, scale_factor=2, mode='bilinear', align_corners=False)
        x_high = self.conv2(x_high)
        x_high = self.bn2(x_high)
        x_high = self.relu(x_high)

        x_fused = x_low + x_high
        x_fused = self.conv3(x_fused)
        x_fused = self.bn3(x_fused)
        x_fused = self.relu(x_fused)

        return x_fused