import json
import os
from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "model": {
        "model_type": "GIN",
        "feature_count": 7,
        "hidden_channels": 128,
        "num_layers": 6,
        "dropout": 0.5,
        "model_dir": ".",
        "model_filename": "best_model_GIN-h128-l6-d0.5.pt",
    },
    "paths": {
        "input_file": "./train_data.pt",
        "prediction_file": "./output/predictions.pt",
        "score_predictions": None,
        "score_reference": None,
        "score_output": None,
    },
    "scoring": {
        "include_model_config": True,
    },
}


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _coerce_number(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if any(ch in lowered for ch in [".", "e"]):
                return float(lowered)
            return int(lowered)
        except ValueError:
            return value
    return value


def load_runtime_config(config_path=None):
    config = deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        config_path = os.environ.get("ML4CO_CONFIG_PATH")
    if config_path is None:
        default_path = Path(__file__).with_name("runtime_config.json")
        if default_path.exists():
            config_path = str(default_path)

    if config_path:
        path = Path(config_path)
        if path.exists():
            with path.open() as handle:
                file_config = json.load(handle)
            _deep_update(config, file_config)

    env_overrides = {
        ("model", "model_type"): os.environ.get("ML4CO_MODEL_TYPE"),
        ("model", "feature_count"): os.environ.get("ML4CO_FEATURE_COUNT"),
        ("model", "hidden_channels"): os.environ.get("ML4CO_HIDDEN_CHANNELS"),
        ("model", "num_layers"): os.environ.get("ML4CO_NUM_LAYERS"),
        ("model", "dropout"): os.environ.get("ML4CO_DROPOUT"),
        ("model", "model_dir"): os.environ.get("ML4CO_MODEL_DIR"),
        ("model", "model_filename"): os.environ.get("ML4CO_MODEL_FILENAME"),
        ("paths", "input_file"): os.environ.get("ML4CO_INPUT_FILE"),
        ("paths", "prediction_file"): os.environ.get("ML4CO_PREDICTION_FILE"),
        ("paths", "score_predictions"): os.environ.get("ML4CO_SCORE_PRED_FILE"),
        ("paths", "score_reference"): os.environ.get("ML4CO_SCORE_REF_FILE"),
        ("paths", "score_output"): os.environ.get("ML4CO_SCORE_OUTPUT_FILE"),
    }
    for (section, key), value in env_overrides.items():
        if value is not None:
            config[section][key] = _coerce_number(value)

    scoring_override = os.environ.get("ML4CO_INCLUDE_MODEL_CONFIG")
    if scoring_override is not None:
        config["scoring"]["include_model_config"] = _coerce_number(scoring_override)

    return config
