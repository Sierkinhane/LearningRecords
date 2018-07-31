"""
dependencies:
torch: 0.4.0
numpy
"""

import torch
import numpy as np

# details about math operation in torch can be found in http://pytorch.org/docs/torch.html#math-operations

# convert numpy to tensor or vise versa(相反的)
np_data = np.arange(6).reshape((2, 3))
tensor_data = torch.from_numpy(np_data)
tensor2numpy = tensor_data.numpy()
print(
	'numpy array:', np_data,
	'\ntorch tensor:', tensor_data,
	'\ntensor to array:', tensor2numpy
)

# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data) # 32-bit floating point
print(
	'abs',
	'\nnumpy:', np.abs(data),
	'\ntensor:', torch.abs(tensor)
)

# sin
print(
	'sin',
	'\nnumpy:', np.sin(data),
	'\ntorch:', torch.sin(tensor)
)

# mean
print(
	'mean',
	'\nnumpy mean:', np.mean(data),
	'\ntorch mean:', torch.mean(tensor)
)

# matrix multiplication
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data) # 32-bit floating point
# correct method
print(
	'matrix multiplication',
	'\nnumpy:', np.matmul(data, data),
	'\ntorch:', torch.mm(tensor, tensor)
)

# incorrect method
data = np.array(data)
print(
	'matrix multiplication',
	'\nnumpy:', data.dot(data),
	#'\ntorch:', torch.dot(tensor.dot(tensor)) #error
)