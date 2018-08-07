import numpy as np 

a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.split(a, 2, axis=0))
print(np.split(a, 3, axis=1))

# 不等量分割
print(np.array_split(a, 2, axis=1))

print(np.vsplit(a, 2)) #等于 print(np.split(a, 2, axis=0))
print(np.hsplit(a, 3)) #等于 print(np.split(a, 3, axis=1))