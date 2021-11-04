import numpy as np
import torch

def depth_to_space(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    if ch % (scale_factor * scale_factor) != 0:
        raise ValueError('channel of tensor must be divisible by '
                         '(scale_factor * scale_factor).必须是整数倍.')

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    tensor = tensor.permute([0, 1, 4, 2, 5, 3])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor


def space_to_depth(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.permute([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([num, scale_factor*scale_factor,ch, new_height, new_width])
    return tensor


def w(y):

    for i in range(y.shape[0]):
      for j in range(y.shape[1]):
        for k in range(y.shape[2]):
            for h in range(y.shape[3]):
             y[i,j,k,h]=max(y[i,j,k,h]+1,24)
    return y

def Norm_1_numpy(y):
    print(type(y),'vv')
    print(y.shape,'tt')
    sum1=0
    for i in range(y.shape[0]):
      sum1=sum1+np.linalg.norm(y[i], ord=1, keepdims=True)
    return sum1/y.shape[0]
    
def Norm_1_torch(y):
    sum=0
    for i in range(y.shape[0]):
      sum=sum+torch.max(torch.norm(y[i],  p=1, dim=0))
    return sum/y.shape[0]
    

def Norm_1(y):
    sum=0
    for i in range(y.shape[0]):
      sum=sum+np.linalg.norm(y[i], ord=1, keepdims=True)
    return sum/y.shape[0]

if __name__ == "__main__":
    a=np.random.randint(12,30,(4,5,5))
    print(a,'a')
    print(w(a),'w(a)')