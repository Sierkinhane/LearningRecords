import torch
import torch.nn.functional as F 
from collections import OrderedDict
# build networks the first way
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
net1 = Net(2, 10, 2)
print(net1)

# build networks the second way
net2 = torch.nn.Sequential(
	torch.nn.Linear(2, 10),
	torch.nn.ReLU(),
	torch.nn.Linear(10, 2)
	)
print(net2)

net3 = torch.nn.Sequential(OrderedDict([
	('Linear', torch.nn.Linear(1,10)),
	('ReLU', torch.nn.ReLU()),
	('Predict', torch.nn.Linear(10, 1))
	]))
print(net3)


		