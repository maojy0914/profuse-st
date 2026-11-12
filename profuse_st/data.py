"""Load existing slide features and expression, aligned explicitly by identifiers."""

from pathlib import Path
import pickle

import h5py
import numpy as np
import torch


def strings(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(values).reshape(-1)]


def unique(values, name):
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {name}")


def finite(array, name):
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")


def load_slide(root, slide_id, feature_root=None, require_expression=True):
    """Only load pickle files produced by a trusted preprocessing pipeline."""
    root = Path(root) / slide_id
    feature_root = Path(feature_root or root.parent) / slide_id
    with h5py.File(feature_root / "feature_uni.h5", "r") as f:
        features = np.asarray(f["embeddings"], dtype=np.float32)
        coords = np.asarray(f["coords"], dtype=np.float32)
        barcodes = strings(f["barcodes"][:])
    unique(barcodes, f"feature barcodes in {slide_id}")
    if features.ndim != 2 or coords.shape != (len(features), 2) or len(barcodes) != len(features):
        raise ValueError(f"Invalid feature/coordinate shapes in {slide_id}")
    with h5py.File(feature_root / "slide_feature_gigapath.h5", "r") as f:
        context = np.asarray(f["features"], dtype=np.float32).squeeze()
    if context.ndim != 1 or not context.size:
        raise ValueError(f"Expected one whole-section embedding in {slide_id}")
    payload = None
    if require_expression:
        with open(root / "gene_expression_ground_truth.pkl", "rb") as f:
            payload = pickle.load(f)
        expression_barcodes = strings(payload["barcodes"])
        unique(expression_barcodes, f"expression barcodes in {slide_id}")
        rows = {b: i for i, b in enumerate(barcodes)}
        try:
            order = [rows[b] for b in expression_barcodes]
        except KeyError as e:
            raise ValueError(f"Missing image feature for barcode {e} in {slide_id}") from e
        features, coords, barcodes = features[order], coords[order], expression_barcodes
    result = {"slide_id": slide_id, "barcodes": barcodes,
              "features": torch.from_numpy(np.ascontiguousarray(features)),
              "coords": torch.from_numpy(np.ascontiguousarray(coords)),
              "global_feature": torch.from_numpy(context)}
    if payload is not None:
        genes = strings(payload["genes"])
        unique(genes, f"genes in {slide_id}")
        labels = np.asarray(payload["expression"], dtype=np.float32)
        if labels.shape != (len(barcodes), len(genes)) or not len(genes):
            raise ValueError(f"Invalid expression shape in {slide_id}")
        finite(labels, "Expression")
        result.update(expression=torch.from_numpy(labels), genes=genes,
                      normalization=str(payload.get("normalization", "unspecified")))
    for key in ("features", "coords", "global_feature"):
        finite(result[key].numpy(), key)
    return result


def gene_embeddings(path, genes):
    """Read the pretrained text vectors; accept files with or without a word2vec header."""
    wanted = set(genes)
    found = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            parts = line.split()
            if not parts:
                continue
            if line_no == 0 and len(parts) == 2 and all(s.isdigit() for s in parts):
                continue
            if parts[0] in wanted:
                if parts[0] in found:
                    raise ValueError(f"Duplicate gene embedding: {parts[0]}")
                found[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Missing Gene2vec embeddings for {len(missing)} genes: {sorted(missing)[:5]}")
    array = np.stack([found[g] for g in genes])
    if array.shape[1] == 0:
        raise ValueError("Empty gene embeddings")
    finite(array, "Gene embeddings")
    return torch.from_numpy(array)


def build_graph(features, coords, k=6, chunk_size=256):
    """Exact spatial Euclidean and semantic cosine kNN, excluding self-neighbors."""
    n = len(features)
    if n < 2 or k < 1:
        raise ValueError("Graphs require at least two spots and k >= 1")
    k = min(k, n - 1)
    norm = torch.nn.functional.normalize(features.float(), dim=-1)
    graph = {"spatial_neighbors": [], "semantic_neighbors": []}
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        spatial = torch.cdist(coords[start:end].float(), coords.float())
        semantic = norm[start:end] @ norm.T
        row, col = torch.arange(end - start), torch.arange(start, end)
        spatial[row, col] = float("inf")
        semantic[row, col] = -float("inf")
        graph["spatial_neighbors"].append(spatial.topk(k, largest=False).indices)
        graph["semantic_neighbors"].append(semantic.topk(k).indices)
    return {key: torch.cat(rows) for key, rows in graph.items()}


def select_rows(slide, selection):
    if selection is None:
        return torch.arange(len(slide["barcodes"]))
    if not selection:
        raise ValueError("Empty spot selection")
    if isinstance(selection[0], str):
        rows = {b: i for i, b in enumerate(slide["barcodes"])}
        ids = [rows[b] for b in selection]
    else:
        ids = selection
    indices = torch.tensor(ids, dtype=torch.long)
    if indices.min() < 0 or indices.max() >= len(slide["barcodes"]):
        raise ValueError("Spot index outside slide")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("Duplicate spot indices")
    return indices


def prepare_slide(root, slide_id, feature_root=None, selection=None, k=6,
                  require_expression=True):
    slide = load_slide(root, slide_id, feature_root, require_expression)
    indices = select_rows(slide, selection)
    cache = Path(feature_root or root) / slide_id / "graph_faiss.pt"
    graph = None
    if cache.exists():
        stored = torch.load(cache, map_location="cpu", weights_only=True)
        if "barcodes" in stored and strings(stored["barcodes"]) != slide["barcodes"]:
            cached_barcodes = strings(stored["barcodes"])
            unique(cached_barcodes, f"cached graph barcodes in {slide_id}")
            if set(cached_barcodes) == set(slide["barcodes"]):
                old_rows = {b: i for i, b in enumerate(cached_barcodes)}
                order = torch.tensor([old_rows[b] for b in slide["barcodes"]])
                inverse = torch.empty_like(order)
                inverse[order] = torch.arange(len(order))
                stored = {key: inverse[torch.as_tensor(stored[key], dtype=torch.long)[order]]
                          for key in ("spatial_neighbors", "semantic_neighbors")}
            elif not require_expression:
                # New inference spots need a graph covering exactly those spots.
                stored = None
            else:
                raise ValueError(f"Cached graph spot set differs in {slide_id}; rebuild the cache")
        if stored is not None:
            graph = restrict_graph(stored, indices, len(slide["barcodes"]), k, slide_id)
    for key in ("features", "coords", "expression"):
        if key in slide:
            slide[key] = slide[key][indices]
    slide["barcodes"] = [slide["barcodes"][i] for i in indices.tolist()]
    slide["graph"] = graph if graph is not None else build_graph(slide["features"], slide["coords"], k)
    return slide


def restrict_graph(stored, indices, n, k, slide_id):
    """Restrict and remap the historical cached graph convention."""
    if k < 1 or n < 2:
        raise ValueError("Graphs require at least two spots and k >= 1")
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[indices] = torch.arange(len(indices))
    graph = {}
    for key in ("spatial_neighbors", "semantic_neighbors"):
        neighbors = torch.as_tensor(stored[key], dtype=torch.long)
        if neighbors.ndim != 2 or neighbors.shape[0] != n or neighbors.numel() == 0:
            raise ValueError(f"Invalid cached {key} in {slide_id}")
        if neighbors.min() < 0 or neighbors.max() >= n:
            raise ValueError(f"Cached neighbor index outside {slide_id}")
        if neighbors.shape[1] < min(k, n - 1):
            raise ValueError("Cached graph has fewer neighbors than requested")
        mapped = remap[neighbors[indices, :min(k, n - 1)]]
        graph[key] = torch.stack([
            torch.cat((row[row >= 0], torch.full((int((row < 0).sum()),), i)))
            for i, row in enumerate(mapped)
        ])
    return graph
