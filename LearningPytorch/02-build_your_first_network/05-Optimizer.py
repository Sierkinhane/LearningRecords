"""
dependencies:
torch: 0.4.0
matplotlib
"""

import torch
import torch.utils.data as Data 
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np 
import random

#torch.manual_seed(1)

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(x.size()))

# put data into torch dataset
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
    )

# default networks
class Net(torch.nn.Module):
    """docstring for Net"""
    def __init__(self, n_feature=1, n_hidden=20, n_output=1):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)

        return x

if __name__ == '__main__':
    
    # different nets
    SGD_net = Net()
    Momentum_net = Net()
    RMSprop_net = Net()
    Adam_net = Net()
    nets = [SGD_net, Momentum_net, RMSprop_net, Adam_net]

    # diferent optimizers
    opt_SGD = torch.optim.SGD(SGD_net.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(Momentum_net.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(RMSprop_net.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(Adam_net.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []] # record loss

    # training
    for epoch in range(EPOCH):
        print('Epoch：', epoch)
        for step, (b_x, b_y) in enumerate(loader):
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)
                #print(output.shape)
                loss = loss_func(output, b_y)
                opt.zero_grad() # clear gradients for next train
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.ylim(0, 0.2)
    plt.show()


