#!/usr/bin/env python
# ================================================================================
# diag_lr_3300.py — 3300 地 m0long 的 lr 诊断（只用训练集 + winter，不碰 summer）
#
# 背景：m0long_3300 用 500 地选出的 lr=0.001 训练停滞（loss≈ln(1100)）。
# 本脚本在 module 0 数据（1100 地，2200 样本）上扫 lr ∈ {1e-3, 3e-3, 1e-2, 3e-2}，
# 各训 15 epoch 报告 train loss/acc + winter R@1，据此选 3300 规模的 lr。
#
# 用法：pixi run python IDEA1-covstdp/experiments/diag_lr_3300.py
# ================================================================================
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "IDEA1-covstdp" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_exp import load_config  # noqa: E402
from mlp_baseline import MLPBaseline, train_mlp  # noqa: E402
from run_mlp import (build_module_ranges, make_query_dataset,  # noqa: E402
                     make_train_loader, make_transform, query_logits)


def main():
    assert Path.cwd() == REPO_ROOT
    cfg = load_config(str(REPO_ROOT / "IDEA1-covstdp" / "phase1" / "configs"
                          / "mlp_m0long_3300_28.json"), 0)
    transform, dims = make_transform(cfg)
    in_dim = dims[0] * dims[1]
    rng = build_module_ranges(cfg["database_places"], cfg["max_module"],
                              cfg["filter"])[0]
    loader, n = make_train_loader(cfg, transform, rng, cfg["max_module"],
                                  cfg["batch_size"], 0)
    cfg_winter = dict(cfg, query_skip=0)
    winter_ds = make_query_dataset(cfg_winter, transform, "winter",
                                   max_samples=500)  # 500 查询够看趋势

    from vprtempo.src.gt import build_ground_truth
    from vprtempo.src.metrics import recallAtK

    class _LL:
        def __iter__(self):
            for x, y in loader:
                yield x, torch.round(y / cfg["filter"]).long()

        def __len__(self):
            return len(loader)

    device = "cpu"
    for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
        torch.manual_seed(0)
        np.random.seed(0)
        model = MLPBaseline(in_dim, cfg["max_module"])
        t0 = time.time()

        def cb(m):
            S = query_logits([m], winter_ds, device).numpy().T
            GT = build_ground_truth(cfg["max_module"], S.shape[1], skip=0,
                                    filter=cfg["filter"])
            return recallAtK(S, GT, K=1)

        acc, _, best_ep = train_mlp(model, _LL(), 15, lr, 0.0, device,
                                    log_fn=lambda s: None, eval_callback=cb)
        print(f"lr={lr:g}: final_train_acc={acc[-1]:.4f} "
              f"(ep5={acc[4]:.4f} ep10={acc[9]:.4f}) "
              f"time={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
