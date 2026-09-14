import torch
x = torch.rand(4, 3,32,32)
print(x.shape)  # torch.Size([4, 3, 32, 32])
print(x[0].shape)  # torch.Size([3, 32, 32])
print(x.ndim)  # 4
print(x.numel())  # 12288
print(x.device)   #cpu
print(x.dtype)  # torch.float32


x = x.permute(0,3,2,1) 

print(x.shape)  # torch.Size([4, 32, 32, 3])

x = x.reshape(4,-1)  # torch.Size([4, 3072])

w = torch.rand(3072,128)
print(w.shape)  # torch.Size([3072, 128])


res = x @ w  # torch.Size([4, 128])

print(res.mean(dim=1))  # tensor([0.5003, 0.4989, 0.5001, 0.4995])

print(res.argmax(dim=1))  # tensor([ 64,  63,  63,  63])
import torch.nn as nn
criterion = nn.CrossEntropyLoss()