from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch_geometric.data.separate import separate


class GraphDataset(Dataset):
    def __init__(self, path):
        self.data, self.slices = torch.load(path, weights_only=False)
        self.num_graphs = int(self.slices["x"].numel() - 1)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        return separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )