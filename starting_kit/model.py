import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, MLP
import os
from torch_geometric.utils import degree


# edge_index = tensor([
#  [0, 2, 3],   # sources
#  [1, 1, 0],   # targets
# ])

FEATURE_COUNT = 6

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(GIN, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels

            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = MLP([hidden_channels] * 3 + [3])

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        logits = self.classifier(x)
        return logits



class Model:
    def __init__(self, model_dir="./"):
        self.device = torch.device("cpu")
        self.net = GIN(FEATURE_COUNT, 128, 4).to(self.device)

        if model_dir is not None:
            path = os.path.join(model_dir, "model.pt")
            if os.path.exists(path):
                self.net.load_state_dict(torch.load(path, map_location=self.device))

        self.net.eval()
    
    def add_degree_feature(self, data):
        """
        Appends degree and normalized degree as features
        """
        # First row contains source nodes of each edge
        row = data.edge_index[0]
        # Compute per-node degree and reshape to [num_nodes, 1]
        deg = degree(row, data.num_nodes).view(-1, 1).float()
        deg_norm = deg / max(data.num_nodes - 1, 1)
        log_deg = torch.log1p(deg)
        # Get existing node features
        x = data.x
        # Error handling if node features do not exist
        if x is None:
            x = torch.ones((data.num_nodes, 1), dtype=torch.float)
        elif x.dim() == 1:
            x = x.view(-1, 1).float()
        else:
            x = x.float()
        # Append degree as a feature
        data.x = torch.cat([x, deg, deg_norm, log_deg], dim=1)
        return data
    
    def add_mean_neighbor_degree(self, data):
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

        if x is None:
            x = torch.ones((N, 1), dtype=torch.float, device=deg.device)
        elif x.dim() == 1:
            x = x.view(-1, 1).float()
        else:
            x = x.float()
        data.x = torch.cat([x, mean_neigh_deg, max_neigh_deg], dim=1)
        return data

    
    def repair_mis(self, mis_pred, edge_index, mis_scores):
       
        source, target = edge_index
        mis = mis_pred.clone()

        changed = True
        while changed:
            changed = False
            conflicts = (mis[source] == 1) & (mis[target] == 1) # both cannot be 1-1 at the same time
            if conflicts.any():
                rr = source[conflicts]
                cc = target[conflicts]
                # drop lower score endpoint (take higher prob endpoint)
                drop_r = mis_scores[rr] <= mis_scores[cc]
                drop_nodes = torch.where(drop_r, rr, cc)
                mis[drop_nodes] = 0
                changed = True
        return mis

    def predict(self, data):
        data = self.add_mean_neighbor_degree(self.add_degree_feature(data))
        x = data.x.float()

        x = x.float().to(self.device)
        edge_index = data.edge_index.to(self.device)

        

        with torch.no_grad():
            out = self.net(x, edge_index)

        mis = (out[:, 0] > 0).long()
        mvc = (out[:, 1] > 0).long()
        mc  = (out[:, 2] > 0).long()

        mis = self.repair_mis(mis, edge_index, out[:, 0])

        return {
            "mis": mis.long().cpu(),
            "mvc": mvc.cpu(),
            "mc": mc.cpu(),
        }

