"""Train one explicitly defined fold and evaluate the selected checkpoint."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from .data import gene_embeddings, prepare_slide, select_rows, load_slide
from .engine import device_for, forward, save_prediction, write_json
from .loss import expression_loss
from .metrics import expression_metrics
from .model import ProFuseST


def validate_split(manifest, root, feature_root=None):
    """Enforce disjoint physical spots, including within-slide splits."""
    sets = {}
    selections = manifest.get("spot_indices", {})
    for group in ("train", "val", "test"):
        slides = manifest.get(group)
        if not isinstance(slides, list) or not slides or len(set(slides)) != len(slides):
            raise ValueError(f"{group} must be a nonempty list of unique slide IDs")
        identities = set()
        for name in slides:
            slide = load_slide(root, str(name), feature_root)
            selection = selections.get(group, {}).get(str(name))
            rows = select_rows(slide, selection)
            identities.update((str(name), slide["barcodes"][i]) for i in rows.tolist())
        sets[group] = identities
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise ValueError(f"Overlapping physical spots in {a} and {b}")


@torch.inference_mode()
def validation_mse(model, slides, embeddings, device):
    model.eval()
    total, count = 0.0, 0
    for slide in slides:
        prediction = forward(model, slide, embeddings, device)
        target = slide["expression"].to(device)
        total += float((prediction - target).square().sum())
        count += target.numel()
    return total / count


def train(config, data_root, embedding_path, manifest, output, feature_root=None, device="auto"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    validate_split(manifest, data_root, feature_root)
    settings = config["training"]
    seed = int(settings.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = device_for(device)
    k = int(config.get("num_neighbors", 6))
    selections = manifest.get("spot_indices", {})
    datasets = {
        group: [prepare_slide(data_root, str(name), feature_root,
                              selections.get(group, {}).get(str(name)), k)
                for name in manifest[group]]
        for group in ("train", "val", "test")
    }
    first = datasets["train"][0]
    genes = first["genes"]
    for slides in datasets.values():
        for slide in slides:
            if slide["genes"] != genes:
                raise ValueError("Gene order differs across sections")
            if slide["normalization"] != first["normalization"]:
                raise ValueError("Expression normalization differs across sections")
            if slide["features"].shape[1] != first["features"].shape[1] or slide["global_feature"].shape != first["global_feature"].shape:
                raise ValueError("Feature dimensions differ across sections")
    embeddings = gene_embeddings(embedding_path, genes).to(device)
    kwargs = dict(config.get("model", {}), spot_dim=first["features"].shape[1],
                  gene_dim=embeddings.shape[1], num_genes=len(genes),
                  global_dim=first["global_feature"].numel())
    model = ProFuseST(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                 weight_decay=float(settings["weight_decay"]))
    epochs, patience = int(settings["epochs"]), int(settings["patience"])
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    alignment_weight = float(config.get("alignment_weight", 0.25))
    if alignment_weight < 0:
        raise ValueError("alignment_weight must be non-negative")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "split.json", manifest)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for index in np.random.permutation(len(datasets["train"])):
            slide = datasets["train"][index]
            optimizer.zero_grad(set_to_none=True)
            prediction = forward(model, slide, embeddings, device)
            target = slide["expression"].to(device)
            loss = expression_loss(prediction, target, alignment_weight)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip", 5.0)), error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            count += len(target)
        score = validation_mse(model, datasets["val"], embeddings, device)
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite validation MSE")
        row = {"epoch": epoch, "train_loss": total / count, "val_mse": score}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best, stale = score, 0
            torch.save({"model_state_dict": model.state_dict(), "model_kwargs": kwargs,
                        "gene_embeddings": embeddings.cpu(), "genes": genes,
                        "normalization": first["normalization"], "num_neighbors": k,
                        "epoch": epoch, "val_mse": score, "config": config}, output / "best.pt")
        else:
            stale += 1
        write_json(output / "history.json", history)
        if stale >= patience:
            break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    metrics = {}
    with torch.inference_mode():
        for slide in datasets["test"]:
            prediction = forward(model, slide, embeddings, device).cpu().numpy()
            metrics[slide["slide_id"]] = expression_metrics(prediction, slide["expression"].numpy())
            save_prediction(output / f'{slide["slide_id"]}.npz', slide, prediction, genes)
    # Per-section averaging keeps unequal section sizes from changing PCC weights.
    names = ("mse", "mae", "pcc", "hvg_pcc_50", "hvg_pcc_200", "heg_pcc_50", "heg_pcc_200")
    average = {name: float(np.mean([m[name] for m in metrics.values() if m[name] is not None]))
               if any(m[name] is not None for m in metrics.values()) else None for name in names}
    write_json(output / "metrics.json", {"sections": metrics, "section_mean": average,
                                        "best_epoch": checkpoint["epoch"], "val_mse": best})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--feature-root", help="Defaults to --data-root")
    p.add_argument("--gene-embeddings", required=True)
    p.add_argument("--split", required=True, help="JSON with train, val, test slide lists")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    manifest = json.loads(Path(args.split).read_text())
    train(config, args.data_root, args.gene_embeddings, manifest, args.output, args.feature_root, args.device)


if __name__ == "__main__":
    main()
