import numpy as np 

a = np.arange(10).reshape(2, 5)
b = np.arange(10).reshape(5, 2)
# print(a)
# print(b)

# print(a<5)
# print(a**2) # 对应元素相乘
# print(a.dot(b)) # 矩阵乘法
# print(np.dot(a,b))

c = np.random.random((4, 3))
# print(c)

# print("sum =",np.sum(c,axis=1))
# # sum = [ 1.96877324  2.43558896]

# print("min =",np.min(c,axis=0))
# # min = [ 0.23651224  0.41900661  0.36603285  0.46456022]

# print("max =",np.max(c,axis=1))
# # max = [ 0.84869417  0.9043845 ]

# print(np.argmax(c, axis=1))
# print(np.argmin(c, axis=0))

# print(c.mean()) # 平均值
# print(np.mean(c))
# print(np.average(c))

# print(np.median(c)) # 中位数

b = np.arange(1,13).reshape(4,3)
# print(b)
# print(np.cumsum(b)) # 返回一个累加的数组
# print(np.diff(c)) # 返回一个累差的数组, 后一项与前一项之差

# print(np.sort(c)) # 排序

# # 转置
# print(np.transpose(b))
# print(b.T)

# print(np.clip(b, 5, 9)) # 小于5变成5, 大于9变成9

# # 展开成一行
# print(b.flatten())

# squeeze 去除维数为1的维
d = np.array([[[1, 2, 3]]])
# print(d.ndim)
print(d.shape)
e = np.squeeze(d, axis=0)
print(e.shape)
print(e[:, np.newaxis].shape) # 添加维数为1的维