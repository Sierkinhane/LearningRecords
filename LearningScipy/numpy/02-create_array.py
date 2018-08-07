import numpy as np 

a = np.array([1, 2, 4]) # list
# print(a)
a = np.array([1, 2, 4], dtype=np.int) # 指定类型: int32, int64, float32, float64
# print(a.dtype)

# create specific data
b = np.array([[1, 2, 3], [3, 4, 5]])
# print(b)

c = np.zeros((10, 1))
# print(c)
d = np.ones((10, 1))
# print(d)
e = np.concatenate((c,d), axis=0) # axis=1
# print(e)

# 创建全空数组，其实每个值都是接近于零的数
f = np.empty((10, 2))
# print(f)

# 创建连续数组
g = np.arange(10, 20, 1) # 10-19 步长为1
# print(g)
# print(np.arange(12).reshape(3, 4))

# 创建线段数据 间隔相等
h = np.linspace(1, 5, 10)
print(h)