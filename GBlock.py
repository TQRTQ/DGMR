import torch
import torch.nn as nn

class GBlockUp(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(GBlockUp, self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)  # 图像大小保持不变 通道数不变
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)  # 图像大小保持不变 通道数翻倍

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = self.conv1(x1)

        x2 = self.BN(x)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.conv3_1(x2)
        x2 = self.BN(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)

        out = x1 + x2

        return out



class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GBlock, self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)  # 图像大小保持不变 通道数不变

    def forward(self, x):
        # x1 = self.up  sample(x)
        x1 = self.conv1(x)

        x2 = self.BN(x)
        x2 = self.relu(x2)
        # x2 = self.upsample(x2)
        x2 = self.conv3_1(x2)
        x2 = self.BN(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_1(x2)

        out = x1 + x2
        print()
        return out