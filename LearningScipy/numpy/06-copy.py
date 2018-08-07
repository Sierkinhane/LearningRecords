import numpy as np 

# 深拷贝
a = np.array([1, 2, 3, 4])
b = a
c = b
a[0] = 2
print(b is a) #True
print(c is a) #True

# 浅拷贝
d = a.copy()
a[0] = 2
print(d is a) #False