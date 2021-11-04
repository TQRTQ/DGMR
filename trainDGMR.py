# coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import numpy as np
import os
import random
from generator import generator
from TemDiscriminator import TemDiscriminator
from SpaDiscriminator import SpaDiscriminator
from utils import w,Norm_1_numpy,Norm_1_torch

cuda = True if torch.cuda.is_available() else False


# 创建文件夹
if not os.path.exists('./img'):
    os.mkdir('./img')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out


batch_size = 128
num_epoch = 100
z_dimension = 100
# 图像预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
])



# mnist dataset mnist数据集下载
mnist = datasets.MNIST(
    root='./data/', train=True, transform=img_transform, download=True
)

# data loader 数据载入
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True
)



SDis=SpaDiscriminator()
TDis=TemDiscriminator()
G = generator(24)
RELU = nn.ReLU()

if torch.cuda.is_available():
    SDis = SDis.cuda()
    TDis = TDis.cuda()
    G = G.cuda()


# 首先需要定义loss的度量方式  （二分类的交叉熵）
# 其次定义 优化函数,优化函数的学习率为0.0003
criterion = nn.BCEWithLogitsLoss()  # 是单目标二分类交叉熵函数
SDis_optimizer = torch.optim.Adam(SDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
TDis_optimizer = torch.optim.Adam(TDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(),betas=(0.0, 0.999), lr=0.00005)

BATCHSIZE=16
M=4
N=22
H=256
W=256
Lambda=20
num_epoch=5
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ##########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    print('第'+str(epoch)+'次迭代')
    for i, (img, _) in enumerate(dataloader):

        X_real_first_half = Variable(torch.randn(BATCHSIZE, M, H, W)).cuda()
        X_real_second_half = Variable(torch.randn(BATCHSIZE,N-M,H,W)).cuda()
        X_real_whole = Variable(torch.randn(BATCHSIZE,N, H, W)).cuda()

        X_real_first_half = Variable(X_real_first_half).cuda()

        num_img = X_real_second_half.size(0)

        X_real_second_half = Variable(X_real_second_half).cuda()  # 将tensor变成Variable放入计算图中
        S = random.sample(range(0, 18), 8)
        S.sort()
        X_real_second_half_DS=X_real_second_half[:,S]

        real_label = Variable(torch.ones (num_img)).cuda()  # 定义真实的图片label为1
        fake_label = Variable(torch.zeros(num_img)).cuda()  # 定义假的图片的label为0

        real_out_DS = SDis(X_real_second_half_DS)  # 将真实图片放入判别器中

        X_real_whole=torch.squeeze(X_real_whole)
        real_out_DT = TDis(X_real_whole)


        ds_loss_real = criterion(real_out_DS, real_label)
        dt_loss_real = criterion(real_out_DT, real_label)

        real_scores_DS = real_out_DS
        real_scores_DT = real_out_DT

        # 计算假的图片的损失
        # z = Variable(torch.randn(16 ,8 ,8 ,8)).cuda()
        z = Variable(Tensor(np.random.normal(0, 1, (16 ,8 ,8 ,8))))



        # fake_img = G(X_real_first_half,z).detach()
        fake_img = G(X_real_first_half,z).detach()
        fake_img2 = fake_img[:,S] #随机抽取8个

        fake_out_DS = SDis(fake_img2)

        print(type(fake_img))
        print(fake_img.shape)

        fake_img1=torch.cat((fake_img,X_real_first_half),dim=1)
        fake_out_TS = TDis(fake_img1)


        ds_loss_fake = criterion(fake_out_DS, fake_label)
        dt_loss_fake = criterion(fake_out_TS, fake_label)

        S_d_loss = RELU(1 - ds_loss_real) + RELU(1 + ds_loss_fake)
        T_d_loss = RELU(1 - dt_loss_real) + RELU(1 + dt_loss_fake)


        fake_scores_DS = fake_out_DS
        fake_scores_DT = fake_out_TS

        # 损失函数和优化
        d_loss = T_d_loss+S_d_loss   # 损失包括判真损失和判假损失


        SDis_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        TDis_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        SDis_optimizer.step()
        TDis_optimizer.step()  # 更新参数

        # z = Variable(torch.randn(16 ,8 ,8 ,8)).cuda() # 得到随机噪声
        z = Variable(Tensor(np.random.normal(0, 1, (16 ,8 ,8 ,8))))

        fake_img = G(X_real_first_half,z)  # 随机噪声输入到生成器中，得到一副假的图片


         # 经过判别器得到的结果
        fake_img1 = fake_img[:, S]
        fake_img2=torch.cat((fake_img,X_real_first_half),dim=1)

        output_DS = SDis(fake_img1)
        output_DT = TDis(fake_img2)

        # 得到的假的图片与真实的图片的label的loss

        dt_g_loss = criterion(output_DT, real_label)
        ds_g_loss = criterion(output_DS, real_label)

        # X_real_whole=w(X_real_whole)

        r_loss_sum=0
        for i in range(fake_img.shape[0]):
          
          result=torch.mul((fake_img[i] - X_real_second_half[i]), X_real_second_half[i])
         
          #   result=result.detach().cpu().numpy()

          r_loss = (1 / H * W * N) * Lambda * Norm_1_torch(result)
          r_loss_sum=r_loss_sum+r_loss
        # r_loss_sum = torch.from_numpy(r_loss_sum)
        g_loss_sum=dt_g_loss+ds_g_loss-r_loss_sum/fake_img.shape[0]


        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0

        g_loss_sum.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

       
        # # 打印中间的损失
        # if (i + 1) % 100 == 0:
        #     print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
        #           'D real: {:.6f},D fake: {:.6f}'.format(
        #         epoch, num_epoch, d_loss.data.item(), g_loss_sum.data.item(),
        #         real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
        #     ))
        # if epoch == 0:
        #     real_images = to_img(real_img.cpu().data)
        #     save_image(real_images, './img/real_images.png')
    # fake_images = to_img(fake_img.cpu().data)
    # save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))
print('....................................')
# 保存模型
torch.save(G.state_dict(), './generator.pth')  
torch.save(SDis.state_dict(), './SpaDiscriminator.pth')
torch.save(TDis.state_dict(), './TemDiscriminator.pth')