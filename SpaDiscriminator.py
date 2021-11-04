import torch
import torch.nn as nn
import random
from DBlock import  DBlock,DBlockDown,DBlockDownFirst
from spectralNormalization import SpectralNorm
from utils import space_to_depth



class SpaDiscriminator(nn.Module):
    def __init__(self,):
        super(SpaDiscriminator, self).__init__()
        self.DBlockDown_1 = DBlockDownFirst(4, 48)
        self.DBlockDown_2 = DBlockDown(48, 96)
        self.DBlockDown_3 = DBlockDown(96, 192)
        self.DBlockDown_4 = DBlockDown(192, 384)
        self.DBlockDown_5 = DBlockDown(384, 768)
        self.DBlock_6 = DBlock(768, 768)
        self.sum_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.linear =  SpectralNorm(nn.Linear(in_features = 768*1*1, out_features = 1))
        self.batchnorm = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()

    def forward(self,x):

        T_H = random.sample(range(0, 128), 1)
        T_W = random.sample(range(0, 128), 1)
        x = x[:, :, T_H[0]:T_H[0] + 128, T_W[0]:T_W[0] + 128]


        for i in range(x.shape[1]):
          x_temp=x[:,i]
          x_temp=x_temp.view(x_temp.shape[0],1,x_temp.shape[1],x_temp.shape[2])
          x_temp = space_to_depth(x_temp,2)
          x_temp=torch.squeeze(x_temp)
          x_temp = self.DBlockDown_1(x_temp)
          x_temp = self.DBlockDown_2(x_temp)
          x_temp = self.DBlockDown_3(x_temp)
          x_temp = self.DBlockDown_4(x_temp)
          x_temp = self.DBlockDown_5(x_temp)
          x_temp = self.DBlock_6(x_temp)
          x_temp = self.sum_pool(x_temp)
          x_temp = x_temp.view(x_temp.shape[0],x_temp.shape[1])
          x_temp = x_temp * 4
          out = self.linear(x_temp)

          if i==0:
           data=out
          else:
           data=data+out

        data=self.relu(data)
        data = torch.squeeze(data)
        return data


if __name__ == "__main__":
    pass
    X_SpaDri = torch.randn((20,8,256,256))
    spaDri = SpaDiscriminator()
    outSpaDri = spaDri(X_SpaDri)
    outSpaDri = torch.squeeze(outSpaDri)
    print('outSpaDri.shape', outSpaDri.shape)
    print('SpaDiscriminator ok')