import numpy as np 

x = np.linspace(-1, 1, 10)[np.newaxis, :]
x_ = np.linspace(-1, 1, 10)[:, np.newaxis]
print(x)
print(x_)