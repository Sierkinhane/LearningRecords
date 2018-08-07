import torch 

tensor = torch.randn(3, 4, dtype=torch.float64)
long_tensor = tensor.float() # float, int, long, double, char, byte, short, half

print(tensor.dtype)
print(long_tensor.dtype)

# 默认生成torch.FloatTensor--torch.float32
tensor = torch.Tensor(3, 5)
assert isinstance(tensor, torch.FloatTensor)
tensor = torch.rand(3, 5)
assert isinstance(tensor, torch.FloatTensor)
tensor = torch.randn(3, 5)
assert isinstance(tensor, torch.FloatTensor)
print(tensor.dtype)

# type()
int_tensor = tensor.type(torch.IntTensor)
print(int_tensor)

# type_as
tensor_1 = torch.FloatTensor(5)
tensor_2 = torch.IntTensor([1, 2])
tensor_1 = tensor_1.type_as(tensor_2)
print(tensor_1)