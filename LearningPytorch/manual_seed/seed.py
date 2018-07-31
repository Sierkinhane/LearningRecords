import torch
import numpy as np 
import random
import tensorflow
# 如果设置manual seed为一个固定数值，则每次得到结果都一样，但是每次里面random的值不一定相同
manual_seed = np.random.randint(10)
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

for i in range(3):
    a = random.randint(1,10)
    print(a)
    b = np.random.rand(10)
    print(b)
    c = torch.rand(10)
    print(c)