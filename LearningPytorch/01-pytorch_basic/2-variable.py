"""
dependencies:
torch: 0.4.0
numpy
"""

import torch
from torch.autograd import Variable

# Variable in torch is to build a computational graph,
# but this graph is dynamic compared with a static graph in tensorflow or theano.
# so torch does not have placeholder, torch can just pass variable to the computational graph.

tensor = torch.FloatTensor([[1, 2], [3, 4]]) # build a tensor
variable = Variable(tensor, requires_grad=True) # build a variable, usually for compute grdients

print(tensor.dtype)
print(variable.dtype)

# till now the tensor and variable seem the same.
# However, the variable is a part of the graph, it's a part of the auto-gradient.

t_out = torch.mean(tensor*tensor) # x^2
v_out = torch.mean(variable*variable) # x^2
print(t_out)
print(v_out)
#for i in range(10):
v_out.backward(retain_graph=True) # backpropagation from v_out
# v_out =  1/4 * sum(variable*variable)
# the gradient w.r.t the variable. d(v_out)/d(variable) = d(x^2) = 1/4 * 2 * variable = variable/2
print('grad:', variable.grad)
'''
tensor([[ 0.5000,  1.0000],
        [ 1.5000,  2.0000]])
'''
print(variable) # this is data in variable format
print(variable.data) # this is data in tensor format
print(variable.data.numpy()) # this is data in numpy format
