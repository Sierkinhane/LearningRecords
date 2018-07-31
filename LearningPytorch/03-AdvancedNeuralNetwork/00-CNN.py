"""
dependencies:
torch: 0.4.0
torchvision
matplotlib
"""

import os 
import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision
import matplotlib.pyplot as plt 

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 16
LR = 0.001
DOWNLOAD_MNIST = False

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
	# not mnist dir or mnist is empty dir
	DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
	root='./mnist/',
	train=True,
	transform=torchvision.transforms.ToTensor(),
	download=DOWNLOAD_MNIST
	)

# plot one example
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[4].numpy(), cmap='gray')
# print(train_data.train_labels[4])
# plt.show()
def show_batch(loader):
	for epoch in range(1):
		for step, (batch_x, batch_y) in enumerate(loader):
			# train your own data...
			print('epoch: ', epoch, '\nStep: ', step, '\nbatch x: ', batch_x.numpy(), '\nbatch y: ', batch_y.size())

# Data loader for easy mini-batch return in training,
# the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# batch x:  torch.Size([5, 1, 28, 28]) 
# batch y:  torch.Size([5])

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# print(test_data.test_data.size()) --- [10000, 28, 28]
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:500]/255.
# print(test_x.size()) # shape from (2000, 28, 28) to (2000, 1, 28, 28)
test_y = test_data.test_labels[:500]

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(		# input shape (1, 28, 28)
			nn.Conv2d(					
				in_channels=1,    		# input channels
				out_channels=16,		# num filters
				kernel_size=5,			# kernel size
				stride=1,				# filter step/movement
				padding=2,				# if want same height and width of this image after conv2d, padding=(kernel_size-1)/2 if stride=1
				),						# output shape (16, 28, 28)
			nn.ReLU(),					# activation
			nn.MaxPool2d(kernel_size=2)	# max pooling, output shape (16, 14, 14)
			)
		self.conv2 = nn.Sequential(		# input shape (16, 14, 14)
			nn.Conv2d(16, 32, 5, 1, 2),	# output shape (32, 14, 14)
			nn.ReLU(),					
			nn.MaxPool2d(2)				# output shape (32, 7, 7)
			)
		self.out = nn.Linear(7*7*32, 10)# fully connected layer, output 10 classes

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)		# flatten the output of conv2 to (batch_size, 32*7*7)
		output = self.out(x)

		return output, x 				# return x for visualization
if __name__ == '__main__':
	
	cnn = CNN()
	print(cnn)

	optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
	loss_func = nn.CrossEntropyLoss() 		# the target label is not one-hotted

	# following function (plot with labels) is for visualization

	# plt.ion()
	# training and testing
	loss_his = []
	for epoch in range(EPOCH):
		for step, (b_x, b_y) in enumerate(train_loader):
			# print(step)
			output = cnn(b_x)[0]
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_his.append(loss.data.numpy())

			if step % 50 == 0:
				#plt.cla()
				test_output, last_layer = cnn(test_x) # test_output shape (2000, 10)
				print(test_output.size())
				pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
				accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
				print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

	# print 10 predictions from test data
	test_output, _ = cnn(test_x[:10])
	pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
	print(pred_y, 'prediction number')
	print(test_y[:10].numpy(), 'real number')

	for name, param in cnn.named_parameters():
		print(name, param.size())
	# conv1.0.weight torch.Size([16, 1, 5, 5])
	# conv1.0.bias torch.Size([16])
	# conv2.0.weight torch.Size([32, 16, 5, 5])
	# conv2.0.bias torch.Size([32])
	# out.weight torch.Size([10, 1568])
	# out.bias torch.Size([10])

	# for i, l_h in enumerate(loss_his):
	# 	plt.plot(l_h, label='loss')
	# 	plt.legend(loc='best')
	# # plt.ioff()
	# plt.show()
		
	# train_loader one batch:
	# size: 5, 1, 28, 28
	# [[[[0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    ...
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]]]
	#  [[[0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    ...
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]]]
	#  [[[0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    ...
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]]]
	#  [[[0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    ...
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]]]
	#    ...
	#  [[[0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    ...
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]
	#    [0. 0. 0. ... 0. 0. 0.]]]]