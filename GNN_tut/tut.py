import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[2, 1, 3],
                           [0, 0, 2]], dtype=torch.long)
x = torch.tensor([[1], [1], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)

