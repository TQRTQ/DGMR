import torch
import torch.nn as nn
from DBlock import DBlock,DBlockDown,DBlock3D_1,DBlock3D_2
from spectralNormalization import  SpectralNorm
from utils import space_to_depth

class TemDiscriminator(nn.Module):
    def __init__(self,):
        super(TemDiscriminator, self).__init__()

        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.DBlock3D_1 =DBlock3D_1(4, 48)
        self.DBlock3D_2 = DBlock3D_2(48, 96)
        self.DBlockDown_3 = DBlockDown(96, 192)
        self.DBlockDown_4 = DBlockDown(192, 384)
        self.DBlockDown_5 = DBlockDown(384, 768)
        self.DBlock_6 = DBlock(768, 768)
        self.sum_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(in_features=768 * 1 * 1, out_features=1)
        self.batchnorm = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.avgPool(x)
        x = space_to_depth(x,2)
        x = self.DBlock3D_1(x)
        x = self.DBlock3D_2(x)
        for i in range(0,x.shape[2]):
          x_temp=x[:,:,i,:,:]
          x_temp = self.DBlockDown_3(x_temp)
          x_temp = self.DBlockDown_4(x_temp)
          x_temp = self.DBlockDown_5(x_temp)
          x_temp = self.DBlock_6(x_temp)

          x_temp = self.sum_pool(x_temp)
          x_temp = x_temp.view(x_temp.shape[0], x_temp.shape[1])
          x_temp = x_temp * 4
          out = self.linear(x_temp)

          if i==0:
           data=out
          else:
           data=data+out

        data=self.relu(data)
        data=torch.squeeze(data)
        return data




if __name__ == "__main__":
    X_TemDri = torch.randn((20, 22, 256, 256))
    temDri = TemDiscriminator()
    outTemDri = temDri(X_TemDri)
    outTemDri = torch.squeeze(outTemDri)
    print('outTemDri.shape', outTemDri.shape)
    print('TemDiscriminator ok')