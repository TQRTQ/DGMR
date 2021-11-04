import torch.nn as nn
from CStack import conditioningStack
from convGRU import ConvGRU,ConvGRUCell
from outputStack import outputStack
from LCStack import LCStack
import torch


class generator(nn.Module):
    def __init__(self,input_channel):
        super(generator, self).__init__()
        self.conditioningStack=conditioningStack(input_channel)
        self.LCStack=LCStack()
        self.ConvGRU = ConvGRU(input_dim=[768, 384, 192, 96],
                               hidden_dim=[384, 192, 96, 48],
                               kernel_sizes=3,
                               num_layers=4,
                               gb_hidden_size=[384, 192, 96, 48]
                              )

        self.outputStack=outputStack()


    def forward(self,CD_input,LCS_input):

        CD_input=torch.unsqueeze(CD_input,2)
        LCS_output = self.LCStack(LCS_input)
        CD_output = self.conditioningStack(CD_input)
        CD_output.reverse()  #listéĺş

        LCS_output=torch.unsqueeze(LCS_output,1)

        LCS_outputs=[LCS_output]*18


        for i in range(len(LCS_outputs)):
            if i==0:
               LCS_outputs_data=LCS_outputs[i]
            else:
               LCS_outputs_data = torch.cat((LCS_outputs_data , LCS_outputs[i]), 1)

        layer_output_list, last_state_list = self.ConvGRU(LCS_outputs_data,CD_output)

        RadarPreds = []
        data=layer_output_list[0]
        for i in range(data.shape[1]):
            print(i)
            temp = data[:,i]
            out = self.outputStack(temp)
            RadarPreds.append(out)

        for i in range(len(RadarPreds)):
            if  i==0:
               tempData=RadarPreds[i]
            else:
               tempData=torch.cat((tempData,RadarPreds[i]),dim=1)

        return tempData

if __name__ == "__main__":



        CD_input = torch.randn(10, 4, 256, 256)
        LCS_input = torch.randn((10, 8, 8, 8))


        g = generator(24)
        RadarPreds=g(CD_input,LCS_input)





