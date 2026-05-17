import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import subgraph



# edge_index = tensor([
#  [0, 2, 3],   # sources
#  [1, 1, 0],   # targets
# ])

FEATURE_COUNT = 7
HIDDEN_CHANNELS = 64
NUM_LAYERS = 4
K_H = 4

class GATv2Net(nn.Module):
    def __init__(self, in_channels=FEATURE_COUNT, hidden_channels=HIDDEN_CHANNELS, num_layers=NUM_LAYERS, heads=K_H, dropout=0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads

            conv = GATv2Conv(
                in_dim,
                hidden_channels,
                heads=heads,
                concat=True,
                dropout=dropout
            )

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 3)
        )

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        return self.classifier(x)


class Model:
    def __init__(self, model_dir="./", feature_count = 7, hidden_channels = 64, num_layers = 4, dropout=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = GATv2Net(feature_count, hidden_channels, num_layers, dropout=dropout).to(self.device)

        if model_dir is not None:
            path = os.path.join(model_dir, "model.pt")
            if os.path.exists(path):
                self.net.load_state_dict(torch.load(path, map_location=self.device), strict=False)

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

    def add_core_number_feature(self, data):
        G = to_networkx(data, to_undirected=True)
        G.remove_edges_from(nx.selfloop_edges(G))
        core_dict = nx.core_number(G)
        N = data.num_nodes
        core_feat = torch.tensor([core_dict[i] for i in range(N)], dtype=torch.float).view(-1, 1).to(data.x.device)
        data.x = torch.cat([data.x, core_feat], dim=1)
        return data

    def grasp_mis(self, logits, edge_index, num_candidates=64, weights=None):
        probs = torch.sigmoid(logits)
        candidates = torch.bernoulli(probs.repeat(num_candidates, 1)).to(self.device)
        if weights is not None:
            weights = weights.float().to(self.device)
            node_scores = logits + torch.log1p(weights)
        else:
            node_scores = logits
        
        best_mis = None
        best_value = -1
        row, col = edge_index
        
        for i in range(num_candidates):
            mask = candidates[i]
            
            changed = True
            while changed:
                changed = False
                conflicts = (mask[row] == 1) & (mask[col] == 1)
                if conflicts.any():
                    rr = row[conflicts]
                    cc = col[conflicts]
                    drop_r = node_scores[rr] <= node_scores[cc]
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
                    avail_scores = node_scores.clone()
                    avail_scores[~available] = -float('inf')
                    best_node = avail_scores.argmax()
                    mask[best_node] = 1
                    changed = True
                    
            value = (mask.float() * weights).sum().item() if weights is not None else mask.sum().item()
            if value > best_value:
                best_value = value
                best_mis = mask.clone()
                
        return best_mis


    def build_features(self, data):
        for fn in [
            self.add_degree_feature,
            self.add_mean_neighbor_degree,
            self.add_triangle_count_feature,
            self.add_core_number_feature,
        ]:
            data = fn(data)
        return data
        

    def recursive_basic_mc(self, logits, edge_index, num_nodes, num_candidates=64):
        adj = torch.ones((num_nodes, num_nodes), device=self.device)
        adj.fill_diagonal_(0)
        adj[edge_index[0], edge_index[1]] = 0
        complement_edge_index = adj.nonzero().t()
        
        return self.grasp_mis(logits, complement_edge_index)        

    def get_complement(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        device = self.device
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        adj[edge_index[0], edge_index[1]] = True
        comp_adj = ~adj
        comp_adj.fill_diagonal_(False)
        return comp_adj.nonzero(as_tuple=False).t().long().contiguous()
    
    def recursive_basic_mis(self, data): #https://arxiv.org/pdf/1810.10659
        N = data.num_nodes
        global_labels = torch.full((N,), -1, dtype=torch.long, device=self.device)
        
        curr_x = data.x.float().to(self.device)
        curr_edge_index = data.edge_index.to(self.device)
        
        curr_mapping = torch.arange(N, device=self.device)

        while curr_mapping.numel() > 0:
            with torch.no_grad():
                out = self.net(curr_x, curr_edge_index)
                scores = out[:, 0]

            v_sorted = torch.argsort(scores, descending=True)
            step_labeled_mask = torch.zeros(len(curr_mapping), dtype=torch.bool, device=self.device)
            
            for i in v_sorted:
                idx = i.item()
                
                if step_labeled_mask[idx]:
                    break
                    
                global_labels[curr_mapping[idx]] = 1
                step_labeled_mask[idx] = True
                row, col = curr_edge_index
                neighbors = col[row == idx]
                
                global_labels[curr_mapping[neighbors]] = 0
                step_labeled_mask[neighbors] = True

            remaining_mask = ~step_labeled_mask
            if not remaining_mask.any():
                break
                
            remaining_indices = torch.where(remaining_mask)[0]
            
            new_edge_index, _ = subgraph(
                remaining_indices, 
                curr_edge_index, 
                relabel_nodes=True, 
                num_nodes=len(curr_mapping)
            )
            
            curr_x = curr_x[remaining_indices]
            curr_edge_index = new_edge_index
            curr_mapping = curr_mapping[remaining_indices]
        global_labels[global_labels == -1] = 1
        return global_labels

    def tree_search_mis(self, data, time_budget=5.0, M=16, max_queue_size=256):
        N = data.num_nodes
        best_labels = self.recursive_basic_mis(data)
        weights = data.x[:, 0].float().to(self.device)
        best_value = (best_labels.float() * weights).sum().item()
        queue = [(
            data.x.float().to(self.device),
            data.edge_index.to(self.device),
            torch.arange(N, device=self.device),
            torch.full((N,), -1, dtype=torch.long, device=self.device),
        )]
        end_time = time.time() + time_budget
        while queue and time.time() < end_time:
            pop_idx = torch.randint(len(queue), (1,)).item()
            curr_x, curr_edge_index, curr_mapping, base_labels = queue.pop(pop_idx)
            for m in range(M):
                if time.time() >= end_time:
                    break
                labels = base_labels.clone()
                step_labeled_mask = torch.zeros(curr_mapping.numel(), dtype=torch.bool, device=self.device)
                with torch.no_grad():
                    out = self.net(curr_x, curr_edge_index)
                    scores = out[:, 0]
                if m > 0:
                    scores = scores + torch.randn_like(scores) * (0.01 * m)
                v_sorted = torch.argsort(scores, descending=True)
                row, col = curr_edge_index
                for i in v_sorted:
                    idx = i.item()
                    if step_labeled_mask[idx]:
                        break
                    labels[curr_mapping[idx]] = 1
                    step_labeled_mask[idx] = True
                    neighbors = col[row == idx]
                    labels[curr_mapping[neighbors]] = 0
                    step_labeled_mask[neighbors] = True
                remaining_mask = ~step_labeled_mask
                if not remaining_mask.any():
                    value = (labels.float() * weights).sum().item()
                    if value > best_value:
                        best_value = value
                        best_labels = labels.clone()
                else:
                    remaining_indices = torch.where(remaining_mask)[0]
                    new_edge_index, _ = subgraph(
                        remaining_indices,
                        curr_edge_index,
                        relabel_nodes=True,
                        num_nodes=len(curr_mapping)
                    )
                    queue.append((
                        curr_x[remaining_indices],
                        new_edge_index,
                        curr_mapping[remaining_indices],
                        labels,
                    ))
                    if len(queue) > max_queue_size:
                        queue.pop(0)

        return best_labels
    
    def grasp_mc(self, logits, edge_index, num_nodes, num_candidates=128, weights=None):
        adj = torch.ones((num_nodes, num_nodes), device=self.device)
        adj.fill_diagonal_(0)
        adj[edge_index[0], edge_index[1]] = 0
        complement_edge_index = adj.nonzero().t()
        return self.grasp_mis(logits, complement_edge_index, num_candidates, weights)

    def predict(self, data):
        data = self.build_features(data)
        x = data.x.float().to(self.device)
        edge_index = data.edge_index.to(self.device)

        with torch.no_grad():
            out = self.net(x, edge_index)

        mis_logits = out[:, 0]
        mvc_logits = out[:, 1]
        mc_logits = out[:, 2]
        weights = x[:, 0]
        

        mis = self.grasp_mis(mis_logits, edge_index, num_candidates=512, weights=weights)
        mvc = 1 - self.grasp_mis(-mvc_logits, edge_index, num_candidates=512, weights=weights)
        mc = self.grasp_mc(mc_logits, edge_index, data.num_nodes, num_candidates=512, weights=weights)

        return {
            "mis": mis.long().cpu(),
            "mvc": mvc.cpu(),
            "mc": mc.cpu(),
        }
