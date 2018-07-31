"""
dependencies:
torch: 0.4.0
numpy
"""

import torch
import torch.nn.functional as F 
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200) # x data (tensor), shape=(100, 1)
x = Variable(x)
x_np = x.data.numpy() # numpy array for plotting

# following are popular activation functions
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy() 
y_softplus = F.softplus(x).data.numpy()
# y_softmax = F.softmax(x) # softmax is special kind of activation function, it is about probability

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6)) # width, height
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='green', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='green', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='green', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
