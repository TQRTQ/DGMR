import torch
import torch.nn as nn
from spectralNormalization import  SpectralNorm


class DBlockDown(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(DBlockDown, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 =SpectralNorm( nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = SpectralNorm(nn.Conv2d(in_channels, in_channels, 3,stride=1,padding=1))#图像大小保持不变
        self.conv3_2 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)) # 图像大小保持不变
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)

        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out


class DBlockDownFirst(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(DBlockDownFirst, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = SpectralNorm(nn.Conv2d(in_channels, in_channels, 3,stride=1,padding=1))#图像大小保持不变
        self.conv3_2 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1) ) # 图像大小保持不变
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)


        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out



class DBlock(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(DBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3,stride=1,padding=1))#图像大小保持不变)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu(x)
        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.conv3(x2)
        out = x1 + x2
        return out


class DBlock3D_1(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_1, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1,stride=1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)


        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2

        return out

class DBlock3D_2(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_2, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1,stride=1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)

        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2

        return out