"""Shared slide-level forward pass and output serialization."""

import json
from pathlib import Path

import numpy as np
import torch


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def forward(model, slide, embeddings, device):
    return model(
        slide["features"].to(device), embeddings,
        spot_coords=slide["coords"].to(device),
        global_features=slide["global_feature"].to(device),
        graph={k: v.to(device) for k, v in slide["graph"].items()},
    )["y_hat"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_prediction(path, slide, prediction, genes):
    np.savez_compressed(path, prediction=prediction,
                        genes=np.asarray(genes), barcodes=np.asarray(slide["barcodes"]),
                        coords=slide["coords"].numpy(), slide_id=slide["slide_id"])
