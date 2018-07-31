"""
dependencies:
torch: 0.4.0
numpy
"""

import torch
import numpy as np 
import torch.nn.functional as F
import matplotlib.pyplot as plt 

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1) # 100, 2
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

#             x                        y
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

class Net(torch.nn.Module):
	"""docstring for Net"""
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden = torch.nn.Linear(n_feature, n_hidden)
		self.out = torch.nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x = F.relu(self.hidden(x))
		x = self.out(x)

		return x

net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss() # the target label is Not an one-hotted

plt.ion()

for t in range(100):
	out = net(x) # forward()
	loss = loss_func(out, y) # must be (nn.output, target), the target label is Not one-hotted

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if t % 2 == 0:
		plt.cla()
		# print(out)
		# softmax(out) -- 转化成概率和为1
		# torch.max(x, 1) -- 返回每行值最大的元素和下标 
		prediction = torch.max(F.softmax(out), 1)[1]
		pred_y = prediction.data.numpy().squeeze() # 神经网络预测的结果
		target_y = y.data.numpy()
		# c=pred_y，点的颜色变化随着c的分布而变化，刚开始神经网络预测的值并不准确，所以全部点的颜色一样，
		# 随着网络一步一步的训练，预测的值越来越接近标签，[0, 1], 所以plot的点就以红绿两种颜色分开
		plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
		accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
		plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
		plt.pause(0.1)

plt.ioff()
plt.show()

		
