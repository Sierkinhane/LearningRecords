"""
dependencies:
torch: 0.4.0
"""
import torch 
import torch.utils.data as Data 


torch.manual_seed(1) # s设定随机种子，使得每一次结果都一样

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
	dataset=torch_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,
	num_workers=2,
	)

def show_batch():
	for epoch in range(3):
		for step, (batch_x, batch_y) in enumerate(loader):
			# train your own data...
			print('epoch: ', epoch, '\nStep: ', step, '\nbatch x: ', batch_x.data.numpy(), '\nbatch y: ', batch_y.data.numpy())

if __name__ == '__main__':
	show_batch()