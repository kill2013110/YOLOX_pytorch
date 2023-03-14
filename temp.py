import torch
a = torch.randn([1,8,3,3])
b = a.chunk(4, dim=1)
print('asd')
