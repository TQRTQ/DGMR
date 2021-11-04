import torch.nn as nn
from  utils import depth_to_space
import torch
from spectralNormalization import SpectralNorm

class outputStack(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self,):
        super(outputStack, self).__init__()
        self.BN = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(48, 4, 1))


    def forward(self, x):

        x=self.BN(x)
        x=self.relu(x)
        x=self.conv1(x)
        out=depth_to_space(x,2)

        return out


if __name__ == "__main__":

    data= torch.randn([8, 18, 48, 128, 128])
    os_net = outputStack()
    RadarPreds=[]
    for i in range(data.shape[1]):
        a=data[:,i]
        out=os_net(a)
        RadarPreds.append(out)
    pass


