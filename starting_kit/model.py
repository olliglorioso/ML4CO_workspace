import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, MLP
import os
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
import networkx as nx



# edge_index = tensor([
#  [0, 2, 3],   # sources
#  [1, 1, 0],   # targets
# ])
FEATURE_COUNT = 7 
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        print("Feature count", in_channels, ", hidden channels", hidden_channels, ", num layers", num_layers)
        super(GIN, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels

            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Identity(),
                nn.Linear(hidden_channels, hidden_channels)
            )

            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.mis_head = MLP([hidden_channels, hidden_channels, hidden_channels, 1])
        self.mvc_head = MLP([hidden_channels, hidden_channels, hidden_channels, 1])
        self.mc_head = MLP([hidden_channels, hidden_channels, hidden_channels, 1])

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        mis_logit = self.mis_head(x)
        mvc_logit = self.mvc_head(x)
        mc_logit = self.mc_head(x)
        return torch.cat([mis_logit, mvc_logit, mc_logit], dim=-1)

class Model:
    def __init__(self, model_dir="./", feature_count = FEATURE_COUNT, hidden_channels = HIDDEN_CHANNELS, num_layers = NUM_LAYERS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = GIN(feature_count, hidden_channels, num_layers).to(self.device)

        if model_dir is not None:
            path = os.path.join(model_dir, "model.pt")
            if os.path.exists(path):
                self.net.load_state_dict(torch.load(path, map_location=self.device))

        self.net.eval()
    
    def add_degree_feature(self, data):
        row = data.edge_index[0]
        deg = degree(row, data.num_nodes).view(-1, 1).float()
        deg_norm = deg / max(data.num_nodes - 1, 1)
        log_deg = torch.log1p(deg)
        x = data.x
        x = x.float()
        if x.dim() == 1:
            x = x.view(-1, 1).float()
        data.x = torch.cat([x, deg_norm, log_deg], dim=1)
        
        
        return data
    
    def add_mean_neighbor_degree(self, data):
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
        
        x = data.x.float()
        x = x.float()
        if x.dim() == 1:
            x = x.view(-1, 1).float()
            
        data.x = torch.cat([x, torch.log1p(mean_neigh_deg), torch.log1p(max_neigh_deg)], dim=1)
        return data

    def add_triangle_count_feature(self, data):
        G = to_networkx(data, to_undirected=True)
        triangles = nx.triangles(G)
        N = data.num_nodes
        tri_feat = torch.tensor([triangles[i] for i in range(N)], dtype=torch.float).view(-1, 1)
        x = data.x
        x = x.float()
        if x.dim() == 1:
            x = x.view(-1, 1).float()
        data.x = torch.cat([x, tri_feat], dim=1)
        return data

    def rollout_search_mis(self, logits, edge_index, num_rollouts=64):
        probs = torch.sigmoid(logits)
        candidates = torch.bernoulli(probs.repeat(num_rollouts, 1)).to(self.device)
        
        best_mis = None
        best_size = -1
        row, col = edge_index
        
        for i in range(num_rollouts):
            mask = candidates[i]
            
            changed = True
            while changed:
                changed = False
                conflicts = (mask[row] == 1) & (mask[col] == 1)
                if conflicts.any():
                    rr = row[conflicts]
                    cc = col[conflicts]
                    drop_r = logits[rr] <= logits[cc]
                    drop_nodes = torch.where(drop_r, rr, cc)
                    mask[drop_nodes] = 0
                    changed = True
                    
            changed = True
            while changed:
                changed = False
                sel_neighbors = torch.zeros_like(mask)
                sel_neighbors.index_add_(0, row, mask[col])
                
                available = (mask == 0) & (sel_neighbors == 0)
                if available.any():
                    avail_logits = logits.clone()
                    avail_logits[~available] = -float('inf')
                    best_node = avail_logits.argmax()
                    mask[best_node] = 1
                    changed = True
                    
            size = mask.sum().item()
            if size > best_size:
                best_size = size
                best_mis = mask.clone()
                
        return best_mis
    
    def rollout_search_mc(self, logits, edge_index, num_nodes, num_rollouts=64):
        adj = torch.ones((num_nodes, num_nodes), device=self.device)
        adj.fill_diagonal_(0)
        adj[edge_index[0], edge_index[1]] = 0
        complement_edge_index = adj.nonzero().t()
        
        return self.rollout_search_mis(logits, complement_edge_index)
    
    def rollout_search_mvc(self, logits, edge_index, num_rollouts=64):
        mis_equivalent = self.rollout_search_mis(-logits, edge_index, num_rollouts)
        return 1 - mis_equivalent    
    
    def build_features(self, data):
        for fn in [
            self.add_degree_feature,
            self.add_mean_neighbor_degree,
            self.add_triangle_count_feature,
        ]:
            data = fn(data)
        return data
    

    def predict(self, data):
        data = self.build_features(data)
        x = data.x.float().to(self.device)
        edge_index = data.edge_index.to(self.device)

        with torch.no_grad():
            out = self.net(x, edge_index)
            
        mis_logits = out[:, 0]
        mvc_logits = out[:, 1]
        mc_logits  = out[:, 2]

        mis = self.rollout_search_mis(mis_logits, edge_index, num_rollouts=64)
        mvc = self.rollout_search_mvc(mvc_logits, edge_index, num_rollouts=64)
        mc = self.rollout_search_mc(mc_logits, edge_index, data.num_nodes, num_rollouts=64)
        

        return {
            "mis": mis.long().cpu(),
            "mvc": mvc.cpu(),
            "mc": mc.cpu(),
        }