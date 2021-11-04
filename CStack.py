import torch
import torch.nn as nn
import numpy as np
from spectralNormalization import SpectralNorm
from utils import space_to_depth
from DBlock import DBlockDown


class conditioningStack(nn.Module):
    def __init__(self, in_channels):
        super(conditioningStack, self).__init__()
        # self.s2d = space_to_depth()
        self.DBlockDown_1 = DBlockDown(4,24)
        self.DBlockDown_2 = DBlockDown(24,48)
        self.DBlockDown_3 = DBlockDown(48,96)
        self.DBlockDown_4 = DBlockDown(96,192)
        self.relu = nn.ReLU()
        self.conv3_1 = SpectralNorm(nn.Conv2d(96, 48, 3,stride=1,padding=1))#图像大小保持不变
        self.conv3_2 = SpectralNorm(nn.Conv2d(192, 96, 3, stride=1, padding=1))  # 图像大小保持不变
        self.conv3_3 = SpectralNorm(nn.Conv2d(384, 192, 3, stride=1, padding=1))  # 图像大小保持不变
        self.conv3_4 = SpectralNorm(nn.Conv2d(768, 384, 3, stride=1, padding=1))  # 图像大小保持不变

    def forward(self, x):

        dataList=[]

        for i in range(x.shape[1]):
            x_new=x[:,i,:,:,:]
            x_new=space_to_depth(x_new,2)
            x_new=np.squeeze(x_new)
            x_new=self.DBlockDown_1(x_new)

            if i==0:
                data_0=x_new
            else:
                data_0=torch.cat((data_0,x_new),1)
                if i ==3:
                  data_0=self.conv3_1(data_0)
                  dataList.append(data_0)

            x_new=self.DBlockDown_2(x_new)

            if i==0:
                data1=x_new
            else:
                data1=torch.cat((data1,x_new),1)
                if i == 3:
                    # data1 = nn.utils.spectral_norm(data1)
                    data1 = self.conv3_2(data1)
                    dataList.append(data1)
            x_new=self.DBlockDown_3(x_new)

            if i==0:
                data2=x_new
            else:
                data2=torch.cat((data2,x_new),1)
                if i == 3:
                    # data2 = nn.utils.spectral_norm(data2)
                    data2 = self.conv3_3(data2)
                    dataList.append(data2)
            x_new=self.DBlockDown_4(x_new)

            if i==0:
                data3=x_new
            else:
                data3=torch.cat((data3,x_new),1)
                if i == 3:
                    # data3 = nn.utils.spectral_norm(data3)
                    data3= self.conv3_4(data3)
                    dataList.append(data3)


        return dataList


if __name__ == "__main__":
    cds_net=conditioningStack(24)
    x=torch.randn(8,4,1,256,256)
    out=cds_net(x)
    out.reverse()
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)

    pass
