"""
dependencies:
torch: 0.4.0
matplotlib
"""

import torch
import matplotlib.pyplot as plt 

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

def save():
	
	# build networks
	net1 = torch.nn.Sequential(
		torch.nn.Linear(1, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 1)
	)

	optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
	loss_func = torch.nn.MSELoss()

	for t in range(100):
		prediction = net1(x)
		loss = loss_func(prediction, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# plot result
	plt.figure(1, figsize=(6, 5))
	#plt.subplot(131)
	plt.title('Net1')
	plt.scatter(x.data.numpy(), y.data.numpy())
	plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
	plt.show()
	# two ways to save the net
	torch.save(net1, 'net.pth') # save entire net
	torch.save(net1.state_dict(), 'net_params.pkl') # save only the parameters

def restore_net():
	# restore entire net1 to net2
	net2 = torch.load('net.pkl')
	prediction = net2(x)

		# plot result
	plt.figure(1, figsize=(6, 5))
	#plt.subplot(131)
	plt.title('Net1')
	plt.scatter(x.data.numpy(), y.data.numpy())
	plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
	plt.show()

def restore_params():
	# restore only the parameters in net1 to net2
	net3 = torch.nn.Sequential(
		torch.nn.Linear(1, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 1)
		)
	net3.load_state_dict(torch.load('net_params.pkl'))
	prediction = net3(x)

		# plot result
	plt.figure(1, figsize=(6, 5))
	#plt.subplot(131)
	plt.title('Net1')
	plt.scatter(x.data.numpy(), y.data.numpy())
	plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
	plt.show()

# save net1
save()

# restore entire net (may slow)
restore_net()

# restore only the parameters
restore_params()