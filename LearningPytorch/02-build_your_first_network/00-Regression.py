"""
dependencies:
torch: 0.4.0
numpy
"""
import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from torch.autograd import Variable

# torch.manual_seed(1)  #为CPU设置种子用于生成随机数，以使得结果是确定的 

# x data(tensor), size = tensor([100, 1])
# if not use torch.unsqueeze, it just generates a tensor whose size is tensor([100])
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
noise = 0.2*torch.rand(x.size()) # noise y data(tensor) shape=(100, 1)
y = x.pow(2) + noise

# torch can only train on Variable, so convert them to Variable
# the code below is deprecated in pytorch 0.4.0. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy(), c='g')
# plt.show()

class Net(torch.nn.Module):
	"""docstring for NeT"""
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer
		self.predict = torch.nn.Linear(n_hidden, n_output) # output layer

	def forward(self, x):
		x = F.relu(self.hidden(x)) # activation function for hidden layer
		x = self.predict(x) # liner output
		return x

net = Net(n_feature=1, n_hidden=10, n_output=1) # define the network
print(net) # net architecture

# stochastic gradient descent algorithm
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
# this is for regression mean squared loss
loss_func = torch.nn.MSELoss()

plt.ion() # something about plotting

for i in range(200):
	prediction = net(x) # input x and predict based on x

	loss = loss_func(prediction, y) 

	optimizer.zero_grad() # clear gradients for next train
	loss.backward() # backpropagation, compute gradients
	optimizer.step() # apply gradients

	if i % 5 == 0:
		plt.cla()
		plt.scatter(x.data.numpy(), y.data.numpy())
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
		plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color': 'red'})
		plt.pause(0.1)
for name, param in net.named_parameters():
	print(name, param.size(), param)
plt.ioff()
plt.show()


