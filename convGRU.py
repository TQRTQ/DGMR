import torch
import torch.nn as nn
from torch.nn import init
from GBlock import GBlockUp,GBlock
from spectralNormalization import SpectralNorm


class SequenceGRU(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.conv =SpectralNorm(nn.Conv2d(input_size, input_size, kernel_size=3, padding=1, stride=1))
        self.GBlock = GBlock(input_size,input_size)
        self.GBlockUp = GBlockUp(input_size,input_size)
    def forward(self,x):

        x=self.conv(x)
        x=self.GBlock(x)
        out=self.GBlockUp(x)
        return out




class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=torch.sigmoid):

        super().__init__()
        padding = kernel_size//2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate  = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding,stride=1)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding,stride=1)
        self.out_gate    = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding,stride=1)
        self.activation = activation

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)


    def forward(self, x, prev_state=None):

        if prev_state is None:

            # get batch and spatial sizes
            batch_size = x.data.size()[0]
            spatial_size = x.data.size()[2:]

            # generate empty prev_state, if None is provided
            state_size = [batch_size, self.hidden_size] + list(spatial_size)

            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size)

            else:
                prev_state = torch.zeros(state_size)

        combined_1 = torch.cat([x, prev_state], dim=1)
        update = self.activation(self.update_gate(combined_1))
        reset = self.activation(self.reset_gate(combined_1))
        out_inputs = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self,input_dim, hidden_dim, kernel_sizes, num_layers,gb_hidden_size):
        """
        Generates a multi-layer convolutional GRU.
        :param input_size: integer. depth dimension of input tensors.
        :param hidden_sizes: integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        :param kernel_sizes: integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        :param n_layers: integer. number of chained `ConvGRUCell`.
        """

        super().__init__()

        self.input_size = input_dim
        self.input_dim =input_dim

        if type(hidden_dim) != list:
            self.hidden_sizes = [hidden_dim]*num_layers
        else:
            assert len(hidden_dim) == num_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_dim
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*num_layers
        else:
            assert len(kernel_sizes) == num_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = num_layers

        cells = nn.ModuleList()
        squenceCells=nn.ModuleList()

        for i in range(self.n_layers):

            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(self.input_dim[i], self.hidden_sizes[i], 3)

            cells.append(cell)

        self.cells = cells


        for i in range(self.n_layers):

            squenceCell = SequenceGRU(gb_hidden_size[i])

            squenceCells.append(squenceCell)

        self.squenceCells = squenceCells


    def forward(self, x, hidden):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''

        input = x
        output = []


        layer_output_list = []
        last_state_list = []
        seq_len = input.size(1)
        cur_layer_input = input

        for layer_idx in range(self.n_layers):
            output_inner=[]
            for t in range(seq_len):


               cell= self.cells[layer_idx]
               cell_hidden = hidden[layer_idx]
               squenceCell=self.squenceCells[layer_idx]

               # pass through layer

               a=cur_layer_input[:, t, :, :, :]

               upd_cell_hidden = cell(cur_layer_input[:, t, :, :, :], cell_hidden) # TODO comment
               upd_cell_hidden=squenceCell(upd_cell_hidden)

               output_inner.append(upd_cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)#每一层layer的所有18个hidden状态输出
            cur_layer_input=layer_output

            layer_output_list.append(layer_output)#所有layer层的18个hidden输出
            last_state_list.append(cell_hidden)    #最后一层的18的hidden输出

        layer_output_list = layer_output_list[-1:]
        last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list




if __name__ == "__main__":

    # Generate a ConvGRU with 3 cells
    # input_size and hidden_sizes reflect feature map depths.
    # Height and Width are preserved by zero padding within the module.

    # model = ConvGRU(input_size=8, hidden_sizes=[ 8, 16, 32, 64], kernel_sizes=[3, 3, 3, 3], n_layers=4).cuda()

    # model = nn.DataParallel(model, device_ids=[0, 1])

    hidden_state=[]
    hidden_1 = torch.rand([8, 384, 8, 8], dtype=torch.float32).cuda()
    hidden_2 = torch.rand([8, 192, 16, 16], dtype=torch.float32).cuda()
    hidden_3 = torch.rand([8, 96, 32, 32], dtype=torch.float32).cuda()
    hidden_4 = torch.rand([8, 48, 64, 64], dtype=torch.float32).cuda()

    hidden_state.append(hidden_1)
    hidden_state.append(hidden_2)
    hidden_state.append(hidden_3)
    hidden_state.append(hidden_4)


    x = torch.rand([8, 18, 768, 8, 8], dtype=torch.float32).cuda()

    model = ConvGRU(input_dim =[768, 384, 192, 96],
                    hidden_dim=[384, 192,  96, 48],
                    kernel_sizes=3,
                    num_layers=4,
                    gb_hidden_size=[384,192,96,48]
                   ).cuda()

    model = nn.DataParallel(model, device_ids=[0, 1])

    layer_output_list, last_state_list= model(x, hidden_state)

    print(layer_output_list[-1].size(),layer_output_list[-2].size())
    print(last_state_list[-1].size())
