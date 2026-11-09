#!/usr/bin/env python
# ================================================================================
# verify_f0f3_smoke.py — S3.6 验收 2/3：feature 层冻结与机制生效的一次性验证
#
# 做法（100 地 B5 冒烟档，单 seed）：
#   1. 按 main.initialize_and_run_model 同款流程（三件套种子 → 构造 VPRTempoTrain）
#      拿到训练前的 feature_layer 权重/阈值快照（md5 + L1 基线）；
#   2. 跑完整 train_new_model（feat_stdp/feat_itp 按模式覆盖）；
#   3. 加载训后的 .pth，dump feature_layer 权重/阈值 md5 与 L1 差。
#
# 判据（S3.6 验收）：
#   F0（off,off）：训前/训后权重 md5 相同（逐位冻结）；
#   F1（off,on） ：训后阈值相对初始化发生移动；
#   F2（on,off） ：训后权重相对初始化 L1 变化 > 0。
#
# 用法：pixi run python IDEA1-covstdp/experiments/verify_f0f3_smoke.py --mode f0|f1|f2
# ================================================================================
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_exp import load_config  # noqa: E402

CONFIG = str(REPO_ROOT / "IDEA1-covstdp/phase3/configs/fsmoke_b5_100.json")
FEAT_KEYS = ("feature_layer.w.weight", "feature_layer.thr")


def md5_of(tensor):
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def snapshot(model, tag):
    fl = model.feature_layer
    snap = {
        "w_md5": md5_of(fl.w.weight.data),
        "thr_md5": md5_of(fl.thr.data),
        "w": fl.w.weight.data.detach().cpu().clone(),
        "thr": fl.thr.data.detach().cpu().clone(),
    }
    print(f"[verify] {tag}: w_md5={snap['w_md5']}  thr_md5={snap['thr_md5']}")
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["f0", "f1", "f2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=CONFIG, help="冒烟配置 JSON（临时小档）")
    cli = ap.parse_args()

    assert Path.cwd() == REPO_ROOT, "必须从仓库根目录运行"
    os.environ["PYTHONHASHSEED"] = str(cli.seed)

    flags = {"f0": ("off", "off"), "f1": ("off", "on"), "f2": ("on", "off")}[cli.mode]
    cfg = load_config(cli.config, cli.seed)
    cfg["feat_stdp"], cfg["feat_itp"] = flags
    cfg["exp_id"] = f"fsmoke_b5_{cli.mode}"
    model_name = f"{cfg['exp_id']}__seed{cli.seed}.pth"

    import random as _random
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    _random.seed(cli.seed)

    from vprtempo.VPRTempoTrain import VPRTempoTrain, train_new_model
    from vprtempo.src.loggers import model_logger

    dims = [int(x) for x in cfg["dims"].split(",")]
    logger = model_logger()
    ns = argparse.Namespace(**cfg)
    model = VPRTempoTrain(ns, dims, logger, 1, cfg["database_places"])
    model.to(torch.device('cpu'))

    pre = snapshot(model, "训前(初始化)")

    model_path = REPO_ROOT / "vprtempo" / "models" / model_name
    if model_path.exists():
        model_path.unlink()
    train_new_model([model], model_name)

    # ---- 训后：从保存的 state_dict 重新读取（与 run_inference 同口径）----
    state = torch.load(model_path, weights_only=True)["model_0"]
    post_w = state["feature_layer.w.weight"]
    post_thr = state["feature_layer.thr"]
    print(f"[verify] 训后: w_md5={md5_of(post_w)}  thr_md5={md5_of(post_thr)}")

    w_l1 = float((post_w - pre["w"]).abs().sum())
    thr_l1 = float((post_thr - pre["thr"]).abs().sum())
    print(f"[verify] 权重 L1 变化 = {w_l1:.6f} | 阈值 L1 变化 = {thr_l1:.6f}")

    ok = True
    if cli.mode == "f0":
        ok = (md5_of(post_w) == pre["w_md5"]) and (md5_of(post_thr) == pre["thr_md5"])
        print(f"[verify] F0 逐位冻结: {'PASS' if ok else 'FAIL'}")
    elif cli.mode == "f1":
        ok = thr_l1 > 0 and w_l1 == 0.0
        print(f"[verify] F1 仅 ITP（权重不动、阈值移动）: {'PASS' if ok else 'FAIL'}")
    elif cli.mode == "f2":
        ok = w_l1 > 0 and thr_l1 == 0.0
        print(f"[verify] F2 仅 STDP（权重移动、阈值不动）: {'PASS' if ok else 'FAIL'}")

    summary = {
        "mode": cli.mode, "seed": cli.seed, "flags": flags,
        "pre": {"w_md5": pre["w_md5"], "thr_md5": pre["thr_md5"]},
        "post": {"w_md5": md5_of(post_w), "thr_md5": md5_of(post_thr)},
        "w_l1_delta": w_l1, "thr_l1_delta": thr_l1, "pass": ok,
    }
    out = REPO_ROOT / "IDEA1-covstdp/results/fsmoke_b5" / f"seed_{cli.seed}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"verify_{cli.mode}.json", "w") as f:
        json.dump(summary, f, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
