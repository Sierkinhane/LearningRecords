import torch

a = torch.FloatTensor([[[[3, 3, 3],
					    [1, 1, 1],
					    [2, 2, 2]],
					    [[3, 3, 3],
					    [1, 1, 1],
					    [2, 2, 2]]]])
print(a.size())
# x.view(d, -1) -- d 在d维以后看做一维
print(a.view(a.size(1), -1))#.squeeze())