from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
import torch
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
import networkx as nx

def add_degree_feature(data):
    # First row contains source nodes of each edge
    row = data.edge_index[0]

    # Compute per-node degree and reshape to [num_nodes, 1]
    deg = degree(row, data.num_nodes).view(-1, 1).float()
    deg_norm = deg / max(data.num_nodes - 1, 1)
    log_deg = torch.log1p(deg)

    x = data.x
    if x.dim() == 1:
        x = x.view(-1, 1).float()
    data.x = torch.cat([x, deg_norm, log_deg], dim=1)
    return data

def add_mean_neighbor_degree(data):
    """
    Appends one feature channel: mean degree of each node's neighbors.
    """
    row, col = data.edge_index
    N = data.num_nodes
    deg = degree(row, N, dtype=torch.float) 
    neigh_deg_per_edge = deg[col]

    neigh_sum = torch.zeros(N, dtype=torch.float, device=deg.device)
    neigh_cnt = torch.zeros(N, dtype=torch.float, device=deg.device)
    neigh_sum.index_add_(0, row, neigh_deg_per_edge)
    neigh_cnt.index_add_(0, row, torch.ones_like(neigh_deg_per_edge))
    mean_neigh_deg = (neigh_sum / neigh_cnt.clamp(min=1)).view(-1, 1)

    max_neigh_deg = torch.full((N,), -1e9, dtype=torch.float, device=deg.device)
    max_neigh_deg = max_neigh_deg.scatter_reduce(
        0, row, neigh_deg_per_edge, reduce="amax", include_self=True
    )
    max_neigh_deg = torch.where(
        max_neigh_deg < -1e8, torch.zeros_like(max_neigh_deg), max_neigh_deg
    ).view(-1, 1)      
    
    x = data.x
    if x.dim() == 1:
        x = x.view(-1, 1).float()
    data.x = torch.cat([x, torch.log1p(mean_neigh_deg), torch.log1p(max_neigh_deg)], dim=1)
    return data

def add_triangle_count_feature(data):
    G = to_networkx(data, to_undirected=True)
    triangles = nx.triangles(G)
    N = data.num_nodes
    tri_feat = torch.tensor([triangles[i] for i in range(N)], dtype=torch.float).view(-1, 1)
    x = data.x
    if x.dim() == 1:
        x = x.view(-1, 1).float()
    data.x = torch.cat([x, tri_feat], dim=1)
    return data

def add_core_number_feature(data):
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    core_dict = nx.core_number(G)
    N = data.num_nodes
    core_feat = torch.tensor([core_dict[i] for i in range(N)], dtype=torch.float).view(-1, 1).to(data.x.device)
    data.x = torch.cat([data.x, core_feat], dim=1)
    return data