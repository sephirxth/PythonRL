import torch
x = torch.IntTensor([[1,1,1],[2,2,2],[3,3,3]])
print(x)
print((x, x, x))

print("connect dimension = 0", torch.cat((x, x, x), 0))

print("connect dimension = 1", torch.cat((x, x, x), 1))
