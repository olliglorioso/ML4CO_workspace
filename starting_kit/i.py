import os
import sys
import torch

def resolve_path(*parts):
    app_base = "/app"
    if os.path.exists(app_base):
        return os.path.join(app_base, *parts)

    local_base = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(local_base, *parts)

INPUT_FILE = "./train_data.pt"
SUBMISSION_DIR = "."
OUTPUT_FILE = "./predictions.pt"


def main():
    sys.path.append(SUBMISSION_DIR)
    from model import Model

    # Load model
    model = Model(model_dir=SUBMISSION_DIR)

    # Load graphs (now a list of Data objects)
    graphs = torch.load(INPUT_FILE, weights_only=False)

    predictions = []
    for data in graphs:
        pred = model.predict(data)
        predictions.append(pred)

    torch.save(predictions, OUTPUT_FILE)


if __name__ == "__main__":
    main()
