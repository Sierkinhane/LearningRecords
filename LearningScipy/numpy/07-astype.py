import numpy as np 

a = np.array([1, 2, 3], dtype=np.int32)
print(a.dtype)

b = a.astype(np.int64)

print(a.dtype)
print(b.dtype)

a[0] = 2
print(a)
print(b)