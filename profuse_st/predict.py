"""Predict expression on a section without requiring observed expression."""

import argparse
from pathlib import Path

import torch

from .data import prepare_slide
from .engine import device_for, forward, save_prediction
from .model import ProFuseST


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True, help="Root containing section feature directories")
    p.add_argument("--slides", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = device_for(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = ProFuseST(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name in args.slides:
            path = output / f"{name}.npz"
            if path.exists():
                raise FileExistsError(path)
            slide = prepare_slide(args.data_root, name, k=checkpoint["num_neighbors"], require_expression=False)
            prediction = forward(model, slide, checkpoint["gene_embeddings"].to(device), device)
            save_prediction(path, slide, prediction.cpu().numpy(), checkpoint["genes"])
            print(path)


if __name__ == "__main__":
    main()
