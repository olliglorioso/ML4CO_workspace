import os
import shutil
from pathlib import Path

import torch

from model import Model
from runtime_config import load_runtime_config


def resolve_input_path(default_path):
    return os.environ.get("INPUT_FILE", default_path)


def resolve_output_path(default_path):
    return os.environ.get("OUTPUT_FILE", default_path)


def maybe_prepare_checkpoint(model_dir, model_filename):
    model_path = Path(model_dir) / model_filename
    canonical_path = Path(model_dir) / "model.pt"

    if (
        model_filename != "model.pt"
        and model_path.exists()
        and model_path.resolve() != canonical_path.resolve()
    ):
        shutil.copyfile(model_path, canonical_path)

    return model_path if model_path.exists() else canonical_path


def main():
    config = load_runtime_config()
    model_cfg = config["model"]
    path_cfg = config["paths"]

    if model_cfg.get("model_type") not in (None, "GIN"):
        raise ValueError(
            "This ingestion setup uses model.py, which supports the GIN model only. "
            f"Received model_type={model_cfg.get('model_type')!r}."
        )

    input_file = resolve_input_path(path_cfg["input_file"])
    output_file = resolve_output_path(path_cfg["prediction_file"])
    model_dir = model_cfg["model_dir"]
    model_filename = model_cfg["model_filename"]

    checkpoint_path = maybe_prepare_checkpoint(model_dir, model_filename)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Could not find checkpoint '{model_filename}' in '{model_dir}'."
        )

    model_kwargs = {
        "model_dir": model_dir,
        "feature_count": model_cfg["feature_count"],
        "hidden_channels": model_cfg["hidden_channels"],
        "num_layers": model_cfg["num_layers"],
    }

    model = Model(**model_kwargs)
    graphs = torch.load(input_file, weights_only=False)

    predictions = []
    for data in graphs:
        predictions.append(model.predict(data))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, output_file)

    print("Loaded runtime model configuration:")
    for key, value in model_cfg.items():
        print(f"  {key}: {value}")
    print(f"Predictions written to: {output_file}")


if __name__ == "__main__":
    main()
