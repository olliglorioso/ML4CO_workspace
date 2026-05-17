#!/bin/bash
#SBATCH --job-name=ml4co-readme
#SBATCH --account=rwth2072
#SBATCH --partition=c23g
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /rwthfs/rz/cluster/home/ts106370/ML4CO_workspace
mkdir -p logs starting_kit/models/slurm_runs

module purge
module load PyTorch/nvcr-25.02-py3
source .venv/bin/activate

cd starting_kit
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python - <<'PY'
import json
from pathlib import Path

notebook = json.loads(Path("README.ipynb").read_text())
out = Path("models/slurm_runs/readme_run.py")

lines = []
for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue
    for line in cell.get("source", []):
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        lines.append(line.rstrip("\n"))
    lines.append("")

out.write_text("\n".join(lines) + "\n")
print(f"Wrote {out}")
PY

python - <<'PY'
import torch
import torch_geometric
import networkx

print("torch", torch.__version__)
print("torch_geometric", torch_geometric.__version__)
print("networkx", networkx.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

srun python -u models/slurm_runs/readme_run.py
