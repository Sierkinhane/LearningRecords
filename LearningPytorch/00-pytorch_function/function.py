import torch
import numpy as np
import torch.nn.functional as F

#a = torch.randn(1, 3, 1)
a = torch.linspace(-1, 1, 5)
# squeeze 对数据的维度进行压缩
b = torch.squeeze(a) # 把维数为一的维度去掉
print(
	"a's type is:", type(a),
	'\nsize :', a.shape
) # torch.size([5])
print('b:', b)
c = torch.squeeze(a, 0) # 若第二个维度维数为一去掉，不唯一不做任何操作
print('c:', c)

# unsqueeze 对数据的维度进行扩充
d = torch.unsqueeze(a, dim=0) # 在第零个维度加上一个维数
e = a.unsqueeze(dim=1)
print('d:', d.size())
print('e:', e) # torch.size([5,1])

f = np.linspace(-1, 1, 5) # shape=(5,) 并不是一行五列，它只是一个数组，并不是矩阵
print('f:', f.size)

# torch.normal(means=tensor, std=scatter)
h = torch.FloatTensor([[1],[1]])
g = torch.normal(h,1)
