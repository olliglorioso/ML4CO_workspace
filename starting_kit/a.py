import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from pathlib import Path
import itertools
import csv

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
import networkx as nx
# from features import add_degree_feature, add_mean_neighbor_degree, add_triangle_count_feature, add_core_number_feature
from model import Model
import copy

RESULTS_CSV = Path("models/hpo_results.csv")

def get_prepared_data(dataset, limit=1000, train_split=0.8):
    from model import build_features
    all_graphs = [build_features(dataset[i]) for i in range(min(limit, len(dataset)))]
    split = int(train_split * len(all_graphs))

    return all_graphs[:split], all_graphs[split:]

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        y = torch.stack([batch.mis_sol, batch.mvc_sol, batch.cli_sol], dim=-1).float()

        optimizer.zero_grad()
        logits = model(batch.x.float(), batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y = torch.stack([batch.mis_sol, batch.mvc_sol, batch.cli_sol], dim=-1).float()
            logits = model(batch.x.float(), batch.edge_index)
            # Reduction='sum' ensures exact loss calculation across varying graph sizes
            total_val_loss += F.binary_cross_entropy_with_logits(logits, y, reduction='sum').item()
    return total_val_loss / len(loader.dataset)


def run_experiment(config, train_graphs, val_graphs, device):
    features_idx = config['features']
    features_str = '_'.join(map(str,features_idx))
    in_channels = len(features_idx) + 1

    # Model Selection
    if config['model_type'] == "GIN":
        from model import GIN as model_cls
    elif config['model_type'] == "GAT":
        from model import GATv2Net as model_cls
    elif config['model_type'] == "GSAGE":
        from model_gsage import GraphSAGENet as model_cls
    else:
        raise ValueError(f"Unsupported model_type: {config['model_type']}")

    model_kwargs = {
        "features_idx": features_idx,
        "hidden_channels": config['hidden_channels'],
        "num_layers": config['num_layers'],
    }
    if config['model_type'] in {"GAT", "GSAGE"}:
        model_kwargs["dropout"] = config.get('dropout', 0.2)
        model_kwargs["heads"] = config.get('heads', 0.2)


    model = model_cls(**model_kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_loader = DataLoader(train_graphs, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config['batch_size'], shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0

    # ===== add smoothing =====
    smooth_val_loss = None
    beta = 0.9   # 越大越平滑（0.9~0.99都可以）

    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        # ===== EMA smoothing =====
        if smooth_val_loss is None:
            smooth_val_loss = val_loss
        else:
            smooth_val_loss = beta * smooth_val_loss + (1 - beta) * val_loss

        # scheduler 用 smooth
        scheduler.step(smooth_val_loss)

        # ===== early stopping & save based on smoothed loss =====
        if smooth_val_loss < best_val_loss:
            best_val_loss = smooth_val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_count": in_channels,
                    "hidden_channels": config['hidden_channels'],
                    "num_layers": config['num_layers'],
                    "dropout": config.get('dropout', 0.0),
                    "features": features_idx,
                    "model_type": config['model_type'],
                    "heads": config['heads'],
                },
                f"models/best_model_{config['model_type']}-h{config['hidden_channels']}-l{config['num_layers']}-d{config.get('dropout', 0.0)}_{features_str}_{config['heads']}.pt"
            )
        else:
            patience_counter += 1

        if patience_counter >= config['early_stopping']:
            break

    return best_val_loss

# --- 4. Hyperparameter Optimization Sweep ---

def hpo_sweep(dataset, device):
    """Explores combinations of hidden channels and layers."""

    import itertools
    from model import all_features
    all_combos = []
    arr = list(range(3, len(all_features) + 1))
    for k in range(1, len(arr)+1):
        all_combos += list(itertools.combinations(arr, k))
    all_combos = [[1,2] + list(a) for a in all_combos]
    all_combos = sorted(all_combos, key=len, reverse=True)
    all_combos.append([1,2])
    train_graphs, val_graphs = get_prepared_data(dataset, limit=5000)

    print(all_combos)

    search_space = {
        'hidden_channels': [16],
        'num_layers': [2,3,4],
        'lr': [3e-4],
        'batch_size': [16,32],
        'model_type': ["GAT"],
        #'model_type': ["GSAGE"],
        'epochs': [500],
        'early_stopping': [20],
        'dropout': [0.2, 0.3, 0.4, 0.5],
        'features': all_combos,
        'heads': [4,8,16],
    }

    # Generate all combinations
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_overall_loss = float("inf")
    best_config = None
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "model_type",
        "hidden_channels",
        "num_layers",
        "lr",
        "batch_size",
        "epochs",
        "early_stopping",
        "dropout",
        "features",
        "heads",
        "val_loss",
    ]

    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"🚀 Starting HPO Sweep: {len(combinations)} combinations to test.")
        print("-" * 60)

        for i, config in enumerate(combinations):
            features = config['features']
            print(
                f"Trial {i+1}/{len(combinations)} | Testing: Layers={config['num_layers']}, "
                f"Hidden={config['hidden_channels']}, Type={config['model_type']}, Dropout={config['dropout']},"
                f"Features: {features}, Epochs={config['epochs']}, Heads={config['heads']}"
            )

            val_loss = run_experiment(config, train_graphs, val_graphs, device)
            row = {"trial": i + 1, **config, "val_loss": val_loss}
            writer.writerow(row)
            f.flush()

            print(f"Result: Val Loss = {val_loss:.4f}")

            if val_loss < best_overall_loss:
                best_overall_loss = val_loss
                best_config = config

                print("🌟 New best configuration found!")
            print("-" * 30)

    print(f"Saved HPO results to {RESULTS_CSV}")
    return best_config, best_overall_loss

# --- 5. Main Execution ---

TRAIN_FILE = Path("train_data.pt")
from graph_dataset import GraphDataset
dataset = GraphDataset(TRAIN_FILE)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load and Preprocess once (saves time during HPO)
    print("📦 Preparing datasets...")

    # 2. Run HPO
    best_params, best_loss = hpo_sweep(dataset, device)

    print("\n" + "="*60)
    print("🏆 HPO SWEEP COMPLETE")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Best Config: {best_params}")
    print("="*60)

    # 3. Final model loading (optional: retrain on full data with best params)
    # save best model
    # Save the best model's state_dict

