import random
L1 = random.sample(range(1, 10), 5)
L1.sort()
print(L1)

import torch
import torch.nn as nn
import torch.nn as nn
from torch.autograd import Variable
fake_label = Variable(torch.zeros(20)).cuda()

out = torch.randn((20,1)).cuda()
out=torch.squeeze(out)
criterion = nn.BCELoss()

result=criterion(fake_label,out)
print(result)