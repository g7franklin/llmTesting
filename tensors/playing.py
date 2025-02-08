import torch
import numpy as np

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0


t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
