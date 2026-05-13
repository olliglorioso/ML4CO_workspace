import os
import sys
import torch

INPUT_FILE = os.path.join(".", "train_data.pt")
SUBMISSION_DIR = os.path.join(".", "model.pt")
OUTPUT_FILE = os.path.join(".", "output", "predictions.pt")


def main():
    from model import Model

    # Load model
    model = Model(model_dir=".")
    graphs = torch.load(INPUT_FILE, weights_only=False)

    predictions = []
    for data in graphs:
        pred = model.predict(data)
        predictions.append(pred)

    torch.save(predictions, OUTPUT_FILE)


if __name__ == "__main__":
    main()