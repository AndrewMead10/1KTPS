import torch

x = torch.ones(5, 5)

x = torch.tril(x).unsqueeze(0).unsqueeze(0)

# print(x.shape)

# print(x)

# print(x.index_select(2, torch.arange(0, 3)))

# print(x.index_select(2, torch.tensor([2])))

index = torch.arange(0, 5, dtype=torch.long)

# dtype of index
print(index.dtype)
