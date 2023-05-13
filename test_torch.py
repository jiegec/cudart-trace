import torch

x = torch.tensor([1., 2.]).cuda()
y = torch.tensor([3., 4.]).cuda()
z = x * y
