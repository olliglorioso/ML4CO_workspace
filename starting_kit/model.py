import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, MLP
import os
import time
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.utils import subgraph
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import SAGEConv


# edge_index = tensor([
#  [0, 2, 3],   # sources
#  [1, 1, 0],   # targets
# ])

FEATURE_COUNT = 7
HIDDEN_CHANNELS = 64
NUM_LAYERS = 4
K_H = 8


def add_clustering_coefficient_feature(data):
    G = to_networkx(data, to_undirected=True)

    clustering_dict = nx.clustering(G)

    clustering = torch.tensor(
        [clustering_dict[i] for i in range(data.num_nodes)],
        dtype=torch.float
    ).view(-1, 1)

    x = data.x.float()

    if x.dim() == 1:
        x = x.view(-1, 1)

    data.x = torch.cat([x, clustering], dim=1)

    return data



def add_eigenvector_feature(data):
    # 转成 networkx graph
    G = to_networkx(data, to_undirected=True)

    # 计算 eigenvector centrality
    eigen_dict = nx.eigenvector_centrality(
        G,
        max_iter=500,
        tol=1e-6
    )

    # 转成 tensor
    eigen = torch.tensor(
        [eigen_dict[i] for i in range(data.num_nodes)],
        dtype=torch.float
    ).view(-1, 1)

    # 可选：log normalize
    eigen = torch.log1p(eigen)

    x = data.x
    x = x.float()

    if x.dim() == 1:
        x = x.view(-1, 1).float()

    data.x = torch.cat([x, eigen], dim=1)

    return data

def add_degree_feature(data):
    row = data.edge_index[0]

    deg = degree(row, data.num_nodes).view(-1, 1).float()

    # normalized degree
    deg_norm = deg / max(data.num_nodes - 1, 1)

    # log normalized degree
    log_norm_deg = torch.log1p(deg_norm)

    x = data.x
    x = x.float()

    if x.dim() == 1:
        x = x.view(-1, 1).float()

    data.x = torch.cat([x, log_norm_deg], dim=1)

    return data

def add_mean_neighbor_degree(data):
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

    data.x = torch.cat([x, torch.log1p(mean_neigh_deg)], dim=1)
    return data


def add_triangle_count_feature(data):
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

def add_core_number_feature(data):
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    core_dict = nx.core_number(G)
    N = data.num_nodes
    core_feat = torch.tensor([core_dict[i] for i in range(N)], dtype=torch.float).view(-1, 1).to(data.x.device)
    x = data.x
    if x.dim() == 1:
        x = x.view(-1, 1).float()
    data.x = torch.cat([x, core_feat], dim=1)
    return data



def add_max_neighbor_degree(data):
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

    data.x = torch.cat([x, torch.log1p(max_neigh_deg)], dim=1)
    return data

#all_features = [add_degree_feature, add_core_number_feature, add_clustering_coefficient_feature, add_mean_neighbor_degree, add_max_neighbor_degree]
all_features = [add_degree_feature, add_core_number_feature]

def build_features(data):
    for feature in all_features:
        data = feature(data)
    return data


class GraphSAGENet(nn.Module):
    def __init__(
        self,
        features_idx=[],
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        dropout=0.2,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.features_idx = [0] + features_idx
        in_channels = len(self.features_idx)

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_dim, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 3),
        )

    def forward(self, x, edge_index):
        x = x[:, self.features_idx]

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0:   # 第一层dim可能不一致
                x = x + h


        return self.classifier(x)




class GATv2Net(nn.Module):
    def __init__(self, features_idx=[], hidden_channels=HIDDEN_CHANNELS, num_layers=NUM_LAYERS, heads=K_H, dropout=0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.dropout = dropout
        self.features_idx = [0] + features_idx
        in_channels = len(self.features_idx)

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
            # residual projection（关键）
            if in_dim != hidden_channels * heads:
                self.projs.append(nn.Linear(in_dim, hidden_channels * heads))
            else:
                self.projs.append(nn.Identity())

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 3)
        )

    def forward(self, x, edge_index):
        x = x[:, self.features_idx]
        for conv, norm, proj in zip(self.convs, self.norms, self.projs):
            h = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training) # dropout
            x = x + proj(h)

        return self.classifier(x)




class GIN(nn.Module):
    def __init__(self, features_idx, hidden_channels, num_layers):
        super(GIN, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.features_idx = [0] + features_idx
        in_channels = len(self.features_idx)

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels

            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels), #
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
        x = x[:, self.features_idx]
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        mis_logit = self.mis_head(x)
        mvc_logit = self.mvc_head(x)
        mc_logit = self.mc_head(x)
        return torch.cat([mis_logit, mvc_logit, mc_logit], dim=-1)

class Model:
    def __init__(self, model_dir="./", feature_count = 7, hidden_channels = 128, num_layers = 4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = GIN([], hidden_channels, num_layers).to(self.device)

        if model_dir is not None:
            path = os.path.join(model_dir, "model.pt")
            if os.path.exists(path):
                ckpt = torch.load(path, map_location="cpu")
                hiddens = ckpt["hidden_channels"]
                self.features = ckpt["features"]
                layers = ckpt["num_layers"]
                model_type = ckpt["model_type"]
                dropout = ckpt["dropout"]
                if model_type == "GIN":
                    self.net = GIN(self.features, hiddens, layers).to(self.device)
                elif model_type == "GAT":
                    self.net = GATv2Net(self.features, hiddens, layers, ckpt["heads"], dropout).to(self.device)
                elif model_type == "GSAGE":
                    self.net = GraphSAGENet(self.features, hiddens, layers, dropout).to(self.device)
                self.net.load_state_dict(ckpt["model_state_dict"], strict=False)

        self.net.eval()


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

    def grasp_mc2(
        self,
        logits,
        edge_index,
        num_nodes,
        num_candidates=128,
        weights=None,
    ):
        """
        Max Clique via MIS on complement graph
        """

        device = self.device

        row, col = edge_index

        # =====================================================
        # build adjacency matrix (bool)
        # =====================================================

        adj = torch.zeros(
            (num_nodes, num_nodes),
            dtype=torch.bool,
            device=device
        )

        adj[row, col] = True
        adj[col, row] = True

        # no self-loop
        adj.fill_diagonal_(True)

        # complement
        comp_adj = ~adj

        complement_edge_index = comp_adj.nonzero().t()

        return self.grasp_mis(
            logits,
            complement_edge_index,
            num_candidates=num_candidates,
            weights=weights,
        )
    def grasp_mvc(
        self,
        logits,
        edge_index,
        num_nodes,
        num_candidates=128,
        weights=None,
        alpha=0.35,
        beta=0.15,
    ):
        """
        Optimized GRASP-style MVC solver.

        Features:
        - randomized construction
        - uncovered-edge repair
        - degree-aware scoring
        - weight-aware objective
        - redundant-node pruning
        - randomized local search

        Args:
            logits: [N]
            edge_index: [2, E]
            num_nodes: int
            weights: [N] or None
        """

        device = self.device

        row, col = edge_index

        # =========================================================
        # Degree heuristic
        # =========================================================

        deg = torch.bincount(
            torch.cat([row, col]),
            minlength=num_nodes
        ).float().to(device)

        # normalized degree
        deg_score = torch.log1p(deg)

        # =========================================================
        # Base node scores
        # =========================================================

        logits = logits.float()

        if weights is not None:

            weights = weights.float().to(device)

            # prefer:
            # high logit
            # high degree
            # low weight
            node_scores = (
                logits
                + alpha * deg_score
                - beta * torch.log1p(weights)
            )

        else:

            node_scores = (
                logits
                + alpha * deg_score
            )

        # =========================================================
        # Sampling initialization
        # =========================================================

        probs = torch.sigmoid(node_scores)

        probs = probs.clamp(0.05, 0.95)

        candidates = torch.bernoulli(
            probs.repeat(num_candidates, 1)
        ).to(device)

        best_cover = None
        best_value = float("inf")

        # =========================================================
        # Main GRASP loop
        # =========================================================

        for i in range(num_candidates):

            mask = candidates[i].clone()

            # =====================================================
            # REPAIR:
            # cover all uncovered edges
            # =====================================================

            while True:

                uncovered = (
                    (mask[row] == 0)
                    & (mask[col] == 0)
                )

                if not uncovered.any():
                    break

                rr = row[uncovered]
                cc = col[uncovered]

                # choose better endpoint
                choose_r = node_scores[rr] >= node_scores[cc]

                chosen = torch.where(
                    choose_r,
                    rr,
                    cc
                )

                mask[chosen] = 1

            # =====================================================
            # FAST REDUNDANCY CHECK
            # =====================================================

            changed = True

            while changed:

                changed = False

                selected = torch.where(mask == 1)[0]

                if len(selected) == 0:
                    break

                # randomized pruning order
                perm = selected[
                    torch.randperm(len(selected), device=device)
                ]

                for v in perm:

                    mask[v] = 0

                    uncovered = (
                        (mask[row] == 0)
                        & (mask[col] == 0)
                    )

                    # invalid removal
                    if uncovered.any():
                        mask[v] = 1
                    else:
                        changed = True

            # =====================================================
            # LOCAL SEARCH
            # try removing low-value nodes
            # =====================================================

            selected = torch.where(mask == 1)[0]

            if len(selected) > 0:

                # low score first
                _, order = torch.sort(node_scores[selected])

                selected = selected[order]

                for v in selected:

                    mask[v] = 0

                    uncovered = (
                        (mask[row] == 0)
                        & (mask[col] == 0)
                    )

                    if uncovered.any():
                        mask[v] = 1

            # =====================================================
            # OBJECTIVE
            # =====================================================

            if weights is not None:
                value = (
                    mask.float() * weights
                ).sum().item()
            else:
                value = mask.sum().item()

            # =====================================================
            # BEST
            # =====================================================

            if value < best_value:

                best_value = value

                best_cover = mask.clone()

        return best_cover


    def get_complement(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        device = self.device
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        adj[edge_index[0], edge_index[1]] = True
        comp_adj = ~adj
        comp_adj.fill_diagonal_(False)
        return comp_adj.nonzero(as_tuple=False).t().long().contiguous()

    def grasp_mc2(self, logits, edge_index, num_nodes, num_candidates=128, weights=None):
        adj = torch.ones((num_nodes, num_nodes), device=self.device)
        adj.fill_diagonal_(0)
        adj[edge_index[0], edge_index[1]] = 0
        complement_edge_index = adj.nonzero().t()
        return self.grasp_mis(logits, complement_edge_index, num_candidates, weights)

    def grasp_mc(
        self,
        logits,
        edge_index,
        num_nodes,
        num_candidates=128,
        weights=None,
        alpha=0.25,
    ):
        """
        Direct GRASP Max Clique

        Features:
        - direct clique construction
        - candidate intersection
        - degree-aware scoring
        - randomized greedy
        - local expansion
        """

        device = self.device

        row, col = edge_index

        # =====================================================
        # Build adjacency matrix
        # =====================================================

        adj = torch.zeros(
            (num_nodes, num_nodes),
            dtype=torch.bool,
            device=device
        )

        adj[row, col] = True
        adj[col, row] = True

        # =====================================================
        # Degree heuristic
        # =====================================================

        deg = adj.sum(dim=1).float()

        logits = logits.float()

        if weights is not None:

            weights = weights.float().to(device)

            node_scores = (
                logits
                + alpha * torch.log1p(deg)
                + torch.log1p(weights)
            )

        else:

            node_scores = (
                logits
                + alpha * torch.log1p(deg)
            )

        # =====================================================
        # Sampling probs
        # =====================================================

        probs = torch.sigmoid(node_scores)
        probs = probs.clamp(0.05, 0.95)

        best_clique = None
        best_value = -1

        # =====================================================
        # Main GRASP loop
        # =====================================================

        for _ in range(num_candidates):

            clique = torch.zeros(
                num_nodes,
                dtype=torch.bool,
                device=device
            )

            # initially all nodes available
            candidates = torch.ones(
                num_nodes,
                dtype=torch.bool,
                device=device
            )

            # =================================================
            # Clique construction
            # =================================================

            while candidates.any():

                cand_nodes = torch.where(candidates)[0]

                cand_scores = node_scores[cand_nodes]

                # randomized greedy sampling
                cand_probs = torch.softmax(
                    cand_scores,
                    dim=0
                )

                idx = torch.multinomial(
                    cand_probs,
                    1
                ).item()

                v = cand_nodes[idx]

                # add to clique
                clique[v] = True

                # update candidates:
                # must connect to v
                candidates &= adj[v]

                # cannot reselect
                candidates[v] = False

            # =================================================
            # Local expansion
            # =================================================

            improved = True

            while improved:

                improved = False

                non_clique = torch.where(~clique)[0]

                for v in non_clique:

                    clique_nodes = torch.where(clique)[0]

                    if len(clique_nodes) == 0:
                        continue

                    # v connects to all clique nodes
                    if adj[v, clique_nodes].all():

                        clique[v] = True
                        improved = True

            # =================================================
            # Objective
            # =================================================

            if weights is not None:

                value = (
                    clique.float() * weights
                ).sum().item()

            else:

                value = clique.sum().item()

            # =================================================
            # Best
            # =================================================

            if value > best_value:

                best_value = value
                best_clique = clique.clone()

        return best_clique


    def predict(self, data):
        data = build_features(data)
        x = data.x.float().to(self.device)
        edge_index = data.edge_index.to(self.device)


        with torch.no_grad():
            out = self.net(x, edge_index)

        mis_logits = out[:, 0]
        mvc_logits = out[:, 1]
        mc_logits = out[:, 2]
        weights = x[:, 0]


        mis = self.grasp_mis(mis_logits, edge_index, num_candidates=256, weights=weights)
        #mvc = 1 - self.grasp_mis(mvc_logits, edge_index, num_candidates=256, weights=weights)
        mvc = self.grasp_mvc(mvc_logits, edge_index, data.num_nodes, num_candidates=256, weights=weights)
        mc = self.grasp_mc(mc_logits, edge_index, data.num_nodes, num_candidates=256, weights=weights)

        return {
            "mis": mis.long().cpu(),
            "mvc": mvc.cpu(),
            "mc": mc.cpu(),
        }
