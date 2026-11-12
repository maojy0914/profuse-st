"""One explicit fold for within-slide, cross-slide, or cross-patient evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_slide

PATIENT_GROUPS = {
    "DLPFC": {
        "Human1": ("151507", "151508", "151509", "151510"),
        "Human2": ("151669", "151670", "151671", "151672"),
        "Human3": ("151673", "151674", "151675", "151676"),
    },
    "cSCC": {
        "P2": ("GSM4284316", "GSM4284317", "GSM4284318"),
        "P5": ("GSM4284319", "GSM4284320", "GSM4284321"),
        "P9": ("GSM4284322", "GSM4284323", "GSM4284324"),
        "P10": ("GSM4284325", "GSM4284326", "GSM4284327"),
    },
    "HER2ST": {
        "A": ("A2", "A3", "A4", "A5", "A6"),
        "B": ("B1", "B2", "B3", "B4", "B5", "B6"),
        "C": ("C1", "C2", "C3", "C4", "C5", "C6"),
        "D": ("D1", "D2", "D3", "D4", "D5", "D6"),
        "E": ("E1", "E2", "E3"),
        "F": ("F1", "F2", "F3"),
        "G": ("G1", "G2", "G3"),
    },
}

def spatial_labels(coords, seed=42, grid_size=4):
    """Assign whole occupied blocks toward 70/15/15 spot-count targets."""
    coords = np.asarray(coords, dtype=np.float64)
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    span = np.maximum(np.ptp(coords, axis=0), 1.0)
    bins = np.clip(np.floor((coords - coords.min(0)) / span * grid_size), 0, grid_size - 1).astype(int)
    _, inverse, counts = np.unique(bins[:, 0] * grid_size + bins[:, 1], return_inverse=True, return_counts=True)
    if len(counts) < 3:
        raise ValueError("Need at least three occupied spatial blocks; use a larger grid")
    order = np.arange(len(counts))
    np.random.default_rng(seed).shuffle(order)
    assigned = np.zeros(3)
    targets = np.array([0.7, 0.15, 0.15]) * len(coords)
    labels = np.zeros(len(counts), dtype=int)
    for block in order:
        group = np.argmax(targets - assigned)
        labels[block] = group
        assigned[group] += counts[block]
    for group in range(3):
        if np.any(labels == group):
            continue
        donors = [g for g in range(3) if np.sum(labels == g) > 1]
        donor = max(donors, key=lambda g: assigned[g])
        block = min(np.flatnonzero(labels == donor), key=lambda b: counts[b])
        labels[block] = group
        assigned[donor] -= counts[block]
        assigned[group] += counts[block]
    return labels[inverse]


def make_split(dataset, protocol, test, val=None, data_root=None, seed=42, grid_size=4):
    groups = PATIENT_GROUPS[dataset]
    slides = [slide for group in groups.values() for slide in group]
    manifest = {"dataset": dataset, "protocol": protocol}
    if protocol == "cross-slide":
        if test not in slides:
            raise ValueError(f"Unknown section {test}")
        val = val or slides[(slides.index(test) + 1) % len(slides)]
        if val not in slides or val == test:
            raise ValueError("Validation must be a different known section")
        manifest.update(train=[s for s in slides if s not in (test, val)], val=[val], test=[test])
    elif protocol == "cross-patient":
        if test not in groups:
            raise ValueError(f"Unknown patient {test}; choose from {list(groups)}")
        remaining = {p: s for p, s in groups.items() if p != test}
        validation = [s[-1] for s in remaining.values()]
        manifest.update(train=[s for ss in remaining.values() for s in ss if s not in validation],
                        val=validation, test=list(groups[test]), test_patient=test)
    elif protocol == "intra-slide":
        if data_root is None:
            raise ValueError("--data-root is required for intra-slide splits")
        slide = load_slide(data_root, test)
        labels = spatial_labels(slide["coords"].numpy(), seed, grid_size)
        manifest.update(train=[test], val=[test], test=[test], seed=seed, grid_size=grid_size,
                        spot_indices={group: {test: [b for i, b in enumerate(slide["barcodes"]) if labels[i] == j]}
                                      for j, group in enumerate(("train", "val", "test"))})
    else:
        raise ValueError(f"Unknown protocol {protocol}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=PATIENT_GROUPS, required=True)
    p.add_argument("--protocol", choices=("cross-slide", "cross-patient", "intra-slide"), required=True)
    p.add_argument("--test", required=True, help="Section ID, or patient ID for cross-patient")
    p.add_argument("--val", help="Optional validation section for cross-slide")
    p.add_argument("--data-root", help="Required for intra-slide splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-size", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = make_split(args.dataset, args.protocol, args.test, args.val,
                        args.data_root, args.seed, args.grid_size)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
