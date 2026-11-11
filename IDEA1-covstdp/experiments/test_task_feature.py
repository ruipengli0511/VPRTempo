#!/usr/bin/env python3
"""Small numerical acceptance checks for the task-feature experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import SGD

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vprtempo.src import task_update  # noqa: E402
from vprtempo.src.task_feature import TaskFeature, _rank, schedule  # noqa: E402


def check_rank() -> None:
    # Three places, two references per place. Query 0 matches the fall reference.
    db = torch.tensor([[0., 0., 0., 1.], [0., 1., 0., 0.], [0., 0., 1., 0.],
                       [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]])
    query = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    result = _rank(db, query, 3)
    assert result["place_R@1"] == 1.0, result
    assert result["image_R@1"] == 1.0, result
    zero = query.clone()
    zero[1] = 0.0
    invalid = _rank(db, zero, 3)
    assert invalid["invalid_queries"] == 1
    assert invalid["place_R@1"] == 2 / 3


def check_update() -> None:
    device = torch.device("cpu")
    model = TaskFeature(16, 32, seed=4, device=device)
    initial = model.initial_state()
    assert initial["weight"].shape == (32, 16)
    assert (initial["weight"][~model.connection_mask] == 0).all()
    assert (initial["weight"][model.exc_mask] > 0).all()
    assert (initial["weight"][model.inh_mask] < 0).all()
    torch.manual_seed(12)
    anchor = torch.rand(8, 16)
    positive = torch.rand(8, 16)
    negative = torch.rand(8, 16)
    valid = [model.encode(t)[3] for t in (anchor, positive, negative)]
    assert all(bool(v.all()) for v in valid), "synthetic inputs unexpectedly produced zero descriptors"
    reference = TaskFeature(16, 32, seed=4, device=device)
    reference.load_state_dict(model.state_dict())
    opt = SGD([model.layer.w.weight], lr=1e-3)
    before = model.layer.w.weight.detach().clone()
    report = task_update.step(model, anchor, positive, negative, opt, margin=2.0)
    assert report["active_count"] == 8 and np.isfinite(report["loss"])
    assert report["update_norm"] > 0
    assert (model.layer.w.weight[~model.connection_mask] == 0).all()
    assert (model.layer.w.weight[model.exc_mask] > 0).all()
    assert (model.layer.w.weight[model.inh_mask] < 0).all()

    da, _, _, va = reference.encode(anchor)
    dp, _, _, vp = reference.encode(positive)
    dn, _, _, vn = reference.encode(negative)
    assert bool((va & vp & vn).all())
    if task_update.METHOD == "triplet_anchor":
        dp, dn = dp.detach(), dn.detach()
    else:
        dp.retain_grad()
        dn.retain_grad()
    loss = F.relu(2.0 - (da * dp).sum(dim=1) + (da * dn).sum(dim=1)).mean()
    loss.backward()
    if task_update.METHOD == "triplet_full":
        assert dp.grad is not None and dn.grad is not None
        assert float(dp.grad.norm()) > 0 and float(dn.grad.norm()) > 0
    with torch.no_grad():
        expected = before - 1e-3 * reference.layer.w.weight.grad * reference.connection_mask
        reference.layer.w.weight.copy_(expected)
        reference.project()
    max_delta = float((model.layer.w.weight.detach() - reference.layer.w.weight.detach()).abs().max())
    assert max_delta < 2e-6, f"{task_update.METHOD}: update != autograd ({max_delta})"
    print(f"{task_update.METHOD}: update/autograd max_abs_diff={max_delta:.3g}")

    state = model.state_dict()
    restored = TaskFeature(16, 32, seed=99, device=device)
    restored.load_state_dict(state)
    assert torch.equal(restored.layer.w.weight, model.layer.w.weight)
    assert torch.equal(restored.connection_mask, model.connection_mask)


if __name__ == "__main__":
    check_rank()
    check_update()
    initial = {"spring": torch.nn.functional.normalize(torch.rand(100, 12), dim=1),
               "fall": torch.nn.functional.normalize(torch.rand(100, 12), dim=1)}
    triples, sha = schedule(100, 2, 0, initial, negative_min_gap=20)
    assert all(abs(a - n) >= 20 for epoch in triples for _, a, n in epoch)
    assert schedule(100, 2, 0, initial, negative_min_gap=20)[1] == sha
    try:
        schedule(64, 1, 0, initial, negative_min_gap=50)
    except ValueError:
        pass
    else:
        raise AssertionError("impossible exclusion band must fail")
    print("task feature synthetic acceptance passed")
