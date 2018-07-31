import torch
import torch.nn.functional as F

a = torch.ones([2, 2])
b = torch.ones([2, 2])
c = torch.cat([a, b], 0) # column stack
d = torch.cat([a, b], 1) # row stack
# print(a)
# tensor([[ 1.,  1.]])
# print('{0}:\n'.format(d))
# tensor([[ 1.,  1.],
        # [ 1.,  1.]]):
# print('{0}:\n'.format(c))
# tensor([[ 1.,  1.,  1.,  1.]]):

e = torch.arange(1., 11., 2)
# print(a)
# tensor([ 1.,  3.,  5.,  7.,  9.])

# torch.normal
f = torch.normal(e, 1) # 表示生成的第一个数由均值为e[i]标准差为1的正态分布中随机生成
# tensor([ 0.3131,  2.5689,  5.9485,  7.2890,  8.5549])
g = torch.normal(b, 2)
# tensor([[-3.7400,  2.1373],
#         [ 3.2528,  0.7294]])
# print(g)
# print(f)

# 对于单行tensor,直接在行末尾拼接，没有有列数拼接
h = torch.zeros(10)
i = torch.zeros(10)
j = torch.cat([h,i])
# tensor([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
#print(j)

k = torch.arange(0, 1, 0.3).reshape(2, 2)
# tensor([[ 0.0000,  0.3000],
#         [ 0.6000,  0.9000]])
print(k)
print(F.softmax(k, 1))
print(torch.max(k,0))
# (tensor([ 0.6000,  0.9000]), tensor([ 1,  1]))
print(torch.max(k,0)[1]) # 返回各列最大值元素的下标
# tensor([ 1,  1])
print(torch.max(k,1)[1])
# tensor([ 1,  1])

#print(torch.rand(2))