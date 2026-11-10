"""Three-path triplet gradient on the shared VPRTempo feature layer."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from vprtempo.src.task_feature import TaskFeature

METHOD = "triplet_full"


def step(model: TaskFeature, anchor: torch.Tensor, positive: torch.Tensor,
         negative: torch.Tensor, optimizer: torch.optim.Optimizer,
         margin: float) -> dict:
    optimizer.zero_grad(set_to_none=True)
    da, _, _, va = model.encode(anchor)
    dp, _, _, vp = model.encode(positive)
    dn, _, _, vn = model.encode(negative)
    valid = va & vp & vn
    count = int(valid.sum())
    if not count:
        return {"loss": 0.0, "valid_count": 0, "active_count": 0,
                "update_norm": 0.0, "projected_connections": 0}
    gap = (da * dp).sum(dim=1) - (da * dn).sum(dim=1)
    loss = F.relu(margin - gap[valid]).mean()
    active = int((gap[valid] < margin).sum())
    if active:
        before = model.layer.w.weight.detach().clone()
        loss.backward()
        model.layer.w.weight.grad.mul_(model.connection_mask)
        optimizer.step()
        projection = model.project()
        update_norm = float((model.layer.w.weight.detach() - before).norm())
    else:
        projection = {"projected_connections": 0}
        update_norm = 0.0
    return {"loss": float(loss.detach()), "valid_count": count,
            "active_count": active, "update_norm": update_norm, **projection}
