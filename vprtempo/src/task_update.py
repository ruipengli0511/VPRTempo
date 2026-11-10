"""Explicit per-unit anchor-only triplet feedback on VPRTempo feature weights.

For the same detach semantics, this is an exact gradient implementation, not a
new learning principle. The autograd equivalent is checked in the smoke test.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from vprtempo.src.task_feature import TaskFeature

METHOD = "triplet_anchor"


@torch.no_grad()
def step(model: TaskFeature, anchor: torch.Tensor, positive: torch.Tensor,
         negative: torch.Tensor, optimizer: torch.optim.Optimizer,
         margin: float) -> dict:
    lr = float(optimizer.param_groups[0]["lr"])
    da, ha, pre, va = model.encode(anchor)
    dp, _, _, vp = model.encode(positive)
    dn, _, _, vn = model.encode(negative)
    valid = va & vp & vn
    count = int(valid.sum())
    if not count:
        return {"loss": 0.0, "valid_count": 0, "active_count": 0,
                "update_norm": 0.0, "projected_connections": 0}
    gap = (da * dp).sum(dim=1) - (da * dn).sum(dim=1)
    loss = F.relu(margin - gap[valid]).mean()
    active = valid & (gap < margin)
    active_count = int(active.sum())
    if active_count:
        direction = dp - dn
        tangent = direction - (direction * da).sum(dim=1, keepdim=True) * da
        gate = ((pre > 0.0) & (pre < 0.9)).to(ha.dtype)
        feedback = tangent * gate / ha.norm(dim=1, keepdim=True).clamp_min(1e-8)
        feedback *= active[:, None]
        delta = (feedback.T @ anchor) * (lr / count)
        delta *= model.connection_mask
        before = model.layer.w.weight.detach().clone()
        model.layer.w.weight.add_(delta)
        projection = model.project()
        update_norm = float((model.layer.w.weight.detach() - before).norm())
    else:
        projection = {"projected_connections": 0}
        update_norm = 0.0
    return {"loss": float(loss), "valid_count": count,
            "active_count": active_count, "update_norm": update_norm, **projection}
