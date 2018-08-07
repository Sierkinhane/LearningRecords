import numpy as np 

array = np.array([[[1, 2, 3],
				  [4, 5, 6]]], dtype=np.float32)
print(array)

# 维度
print('number of dim:{0}'.format(array.ndim))
# 行数与列数
print('shape is:{0}'.format(array.shape))
# 元素个数
print('size is:{}'.format(array.size))