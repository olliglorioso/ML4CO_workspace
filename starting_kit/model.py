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

FEATURE_COUNT = 8
HIDDEN_CHANNELS = 128
NUM_LAYERS = 4

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
                nn.Identity(),
                nn.Linear(hidden_channels, hidden_channels)
            )

            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Identity(),
            MLP([hidden_channels] * 3 + [3]),
        )

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
        self.net = GIN(FEATURE_COUNT, HIDDEN_CHANNELS, NUM_LAYERS).to(self.device)

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
        data.x = torch.cat([x, deg, deg_norm, log_deg], dim=1)
        
        x = x.float()
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

            
        data.x = torch.cat([x, mean_neigh_deg, max_neigh_deg], dim=1)
        return data
    
    def add_core_number_feature(self, data):
        G = to_networkx(data, to_undirected=True)
        core = nx.core_number(G)
        N = data.num_nodes
        core_feat = torch.tensor([core[i] for i in range(N)], dtype=torch.float).view(-1, 1)
        x = data.x
        x = x.float()
        data.x = torch.cat([x, core_feat], dim=1)
        return data

    def add_triangle_count_feature(self, data):
        G = to_networkx(data, to_undirected=True)
        triangles = nx.triangles(G)
        N = data.num_nodes
        tri_feat = torch.tensor([triangles[i] for i in range(N)], dtype=torch.float).view(-1, 1)
        x = data.x
        x = x.float()
        data.x = torch.cat([x, tri_feat], dim=1)
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
    
    def repair_mc(self, mc_pred, edge_index, mc_scores):
        row, col = edge_index
        mc = mc_pred.clone()

        changed = True
        while changed:
            changed = False
            sel_mask = mc.bool()
            clique_size = sel_mask.sum().item()
            if clique_size < 2:
                break

            both_selected = sel_mask[row] & sel_mask[col]
            intra_deg = torch.zeros(mc.numel(), dtype=torch.long, device=mc.device)
            intra_deg.index_add_(
                0, row[both_selected],
                torch.ones(both_selected.sum(), dtype=torch.long, device=mc.device)
            )

            violations = sel_mask & (intra_deg < clique_size - 1)
            if violations.any():
                scores = mc_scores.clone()
                scores[~violations] = float("inf")
                mc[scores.argmin()] = 0
                changed = True

        return mc

    def repair_mvc(self, mvc_pred, edge_index, mvc_scores):
        row, col = edge_index
        mvc = mvc_pred.clone()

        uncovered = (mvc[row] == 0) & (mvc[col] == 0)
        if uncovered.any():
            rr = row[uncovered]
            cc = col[uncovered]
            pick_r = mvc_scores[rr] >= mvc_scores[cc]
            add_nodes = torch.where(pick_r, rr, cc)
            mvc[add_nodes] = 1
        return mvc
    
    def build_features(self, data):
        for fn in [
            self.add_degree_feature,
            self.add_mean_neighbor_degree,
            self.add_core_number_feature,
            self.add_triangle_count_feature,
        ]:
            data = fn(data)
        return data
    

    def predict(self, data):
        data = self.build_features(data)
        x = data.x.float()

        x = x.float().to(self.device)
        edge_index = data.edge_index.to(self.device)


        with torch.no_grad():
            out = self.net(x, edge_index)

        mis = (out[:, 0] > 0).long()
        mvc = (out[:, 1] > 0).long()
        mc  = (out[:, 2] > 0).long()

        mis = self.repair_mis(mis, edge_index, out[:, 0])
        mvc = self.repair_mvc(mvc, edge_index, out[:, 1])
        mc  = self.repair_mc(mc, edge_index, out[:, 2])

        return {
            "mis": mis.long().cpu(),
            "mvc": mvc.cpu(),
            "mc": mc.cpu(),
        }
