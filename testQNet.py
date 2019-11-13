import PytorchDeepQ 
import torch
import torch.nn as nn
loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
print(input,"\n",target)
output = loss(input, target)
output.backward()