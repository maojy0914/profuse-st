"""Expression regression with per-spot expression distribution alignment."""

import torch.nn.functional as F


def expression_loss(prediction, target, alignment_weight=0.25):
    """MSE + lambda * mean_spots KL(softmax(target) || softmax(prediction))."""
    if alignment_weight < 0:
        raise ValueError("alignment_weight must be non-negative")
    mse = F.mse_loss(prediction, target)
    alignment = F.kl_div(
        F.log_softmax(prediction.float(), dim=-1),
        F.softmax(target.float(), dim=-1),
        reduction="batchmean",
    )
    return {"loss": mse + alignment_weight * alignment, "mse": mse, "alignment": alignment}
