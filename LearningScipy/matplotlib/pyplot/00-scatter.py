import matplotlib.pyplot as plt
import torch
import numpy as np 
import torch.nn.functional as F 

x0 = np.random.random(10).reshape([5, 2])
x0 = torch.from_numpy(x0)
x1 = -x0
y1 = torch.zeros(5)
y2 = torch.ones(5)
x = torch.cat([x0, x1], 0)
y = torch.cat([y1, y2])
print(x)
print(y)
# s is the marker's size
# RdYlGn red--yellow-green
# c 按照y的变化进行颜色的渐变

# 动态画图start
plt.ion()

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], s=100, c=y.data.numpy(), cmap='RdYlBu')
plt.text(0.6,0.6, 'hialice', size=20, color='red')
plt.text(0.0, 0.6, 'hi', fontdict={'size': 20, 'color': 'red'})
plt.legend(loc='best') # 图例
# 动态画图end
plt.ioff()
plt.show()

#plt.cla()   # Clear axis
#plt.clf()   # Clear figure
#plt.close() # Close a figure window