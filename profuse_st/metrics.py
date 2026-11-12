"""Gene-wise Pearson correlation and expression error metrics."""

import numpy as np


def expression_metrics(prediction, target):
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or target.ndim != 2 or not target.size:
        raise ValueError("Prediction and target must have the same nonempty [spots, genes] shape")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite prediction or target")
    a, b = prediction - prediction.mean(0), target - target.mean(0)
    denominator = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    pcc = np.full(target.shape[1], np.nan)
    np.divide((a * b).sum(0), denominator, out=pcc, where=denominator > 0)
    average = lambda values: float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else None
    result = {"mse": float(np.mean((prediction - target) ** 2)),
              "mae": float(np.mean(np.abs(prediction - target))),
              "pcc": average(pcc), "valid_pcc_genes": int(np.isfinite(pcc).sum()),
              "num_spots": target.shape[0], "num_genes": target.shape[1]}
    for group, score in (("hvg", target.var(0)), ("heg", target.mean(0))):
        ranking = np.argsort(-score, kind="stable")
        for k in (50, 200):
            selected = ranking[:k]
            result[f"{group}_pcc_{k}"] = average(pcc[selected])
            result[f"{group}_genes_{k}"] = len(selected)
    return result
