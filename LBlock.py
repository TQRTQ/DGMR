import torch
import torch.nn as nn

class LBlock(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(LBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels,out_channels-in_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)  # 图像大小保持不变
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)  # 图像大小保持不变

    def forward(self, x):
        x1 = self.relu(x)
        x1 = self.conv3_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv3_2(x1)

        x2 = self.conv1(x)
        x3=x
        x23=torch.cat([x2,x3],axis=1)

        out = x1 + x23
        return out