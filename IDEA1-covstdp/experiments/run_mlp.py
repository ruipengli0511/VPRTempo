#!/usr/bin/env python
# ================================================================================
# run_mlp.py — MLP-BP 对照实验驱动器
#
# 功能：读 JSON 配置 → 复用 CustomImageDataset/ProcessImage 数据管线（与 SNN 逐比特
#       相同的预处理）→ 逐模块训练 MLPBaseline（BP+CE）→ 查询集推理 →
#       build_ground_truth + recallAtK → 落盘 results/<exp_id>/seed_<n>/
#
# 与 SNN 侧的对齐点：
#   - 输入：ProcessImage 输出 [0,1] 脉冲向量（dataset.py:SetImageAsSpikes）
#   - 模块切分：user_input_ranges 复刻 VPRTempoTrain.py:677-685
#   - 标签换算：round((labels - idx_scale)/filter) 复刻 VPRTempoTrain.py:433
#   - 推理：各模块 logits 拼接 → 全局 argmax（复刻 SNN 多模块语义）
#   - 评测：metrics.recallAtK + gt.build_ground_truth（与轨 A 同一实现）
#
# 成本计量：训练墙钟/更新次数/峰值显存、推理延迟/吞吐、参数量/非零数/FLOPs
#
# 用法：
#   pixi run python IDEA1-covstdp/experiments/run_mlp.py <config.json> --seed 0
#   调参：--query_override winter（仅调参阶段使用，最终报告一律 summer）
# ================================================================================
import argparse
import json
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

from run_exp import load_config, RESULTS_DIR  # noqa: E402
from mlp_baseline import MLPBaseline, build_feature_mask, train_mlp  # noqa: E402


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_module_ranges(database_places, max_module, filter_):
    """复刻 VPRTempoTrain.py:677-685 的 user_input_ranges 切分"""
    num_modules = -(-database_places // max_module)  # 向上取整
    ranges, start = [], 0
    for _ in range(num_modules):
        end = start + (max_module - 1) * filter_
        ranges.append([start, end])
        start = end + filter_
    return ranges


def make_transform(cfg):
    from vprtempo.src.dataset import ProcessImage
    dims = [int(x) for x in cfg["dims"].split(",")]
    return transforms.Compose([
        ProcessImage(dims, cfg["patches"],
                     patch_norm=cfg.get("patch_norm", "on") == "on")
    ]), dims


def make_train_loader(cfg, transform, img_range, max_samples, batch_size, seed):
    """单模块训练 DataLoader（test=False 多季节合并，标签=全局图像索引）"""
    from vprtempo.src.dataset import CustomImageDataset
    db_dirs = [d.strip() for d in cfg["database_dirs"].split(",")]
    csvs = [os.path.join('./vprtempo/dataset', f'{cfg["dataset"]}-{d}.csv')
            for d in db_dirs]
    ds = CustomImageDataset(
        annotations_file=csvs, base_dir=cfg["data_dir"], img_dirs=db_dirs,
        transform=transform, filter=cfg["filter"], skip=cfg["skip"],
        test=False, img_range=img_range, max_samples=max_samples)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=4, generator=g), len(ds)


def make_query_dataset(cfg, transform, query_dir, max_samples=None):
    """查询数据集（test=True 单季节）"""
    from vprtempo.src.dataset import CustomImageDataset
    csv = os.path.join('./vprtempo/dataset', f'{cfg["dataset"]}-{query_dir}.csv')
    return CustomImageDataset(
        annotations_file=csv, base_dir=cfg["data_dir"],
        img_dirs=[query_dir], transform=transform,
        filter=cfg["filter"], skip=cfg.get("query_skip", cfg["skip"]),
        test=True, max_samples=max_samples)


@torch.no_grad()
def query_logits(models, query_ds, device, batch_size=64):
    """所有查询图像在所有模块上的 logits，拼接为 [n_query, total_places]"""
    loader = DataLoader(query_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4)
    outs = []
    for x, _ in loader:
        x = x.to(device).float().view(x.size(0), -1)
        if x.max() > 1.5:
            x = x / 255.0
        per_module = [m(x) for m in models]
        outs.append(torch.cat(per_module, dim=1).cpu())
    return torch.cat(outs, dim=0)


def evaluate(models, query_ds, cfg, device):
    """logits → 相似度矩阵 → recallAtK"""
    from vprtempo.src.gt import build_ground_truth
    from vprtempo.src.metrics import recallAtK
    logits = query_logits(models, query_ds, device)  # [n_query, total_places]
    S = logits.numpy().T  # [total_places, n_query]，与轨 A 同向：行=db 列=query
    GT = build_ground_truth(cfg["database_places"], S.shape[1],
                            skip=cfg.get("query_skip", cfg["skip"]),
                            filter=cfg["filter"],
                            tolerance=cfg["GT_tolerance"])
    assert S.shape == GT.shape, f"S {S.shape} != GT {GT.shape}"
    return {K: recallAtK(S, GT, K=K) for K in [1, 5, 10, 15, 20, 25]}


def main():
    ap = argparse.ArgumentParser(description="MLP-BP 对照实验驱动器")
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query_override", type=str, default=None,
                    help="仅调参用：覆盖查询集（如 winter）。正式实验不要用")
    ap.add_argument("--lr", type=float, default=None, help="调参网格用：覆盖 lr")
    ap.add_argument("--wd", type=float, default=None, help="调参网格用：覆盖 weight_decay")
    cli = ap.parse_args()

    assert Path.cwd() == REPO_ROOT, f"必须从仓库根目录运行（当前 {Path.cwd()}）"
    os.environ["PYTHONHASHSEED"] = str(cli.seed)
    cfg = load_config(cli.config, cli.seed)
    if cli.lr is not None:
        cfg["lr"] = cli.lr
    if cli.wd is not None:
        cfg["weight_decay"] = cli.wd
    set_seed(cli.seed)
    device = get_device()
    exp_id = cfg["exp_id"]
    variant = cfg["variant"]
    query_dir = cli.query_override or cfg["query_dir"]
    print(f"[run_mlp] exp={exp_id} variant={variant} seed={cli.seed} "
          f"query={query_dir} device={device}")

    transform, dims = make_transform(cfg)
    in_dim = dims[0] * dims[1]
    filter_ = cfg["filter"]
    max_module = cfg["max_module"]
    module_ranges = build_module_ranges(cfg["database_places"], max_module, filter_)
    n_modules = len(module_ranges)
    print(f"[run_mlp] in_dim={in_dim} modules={n_modules} ranges={module_ranges}")

    # ---- 逐模块训练 ----
    if device == "cuda:0":
        torch.cuda.reset_peak_memory_stats()
    t_train0 = time.time()
    models, total_updates, train_accs, best_epochs = [], 0, [], []
    for mi, rng in enumerate(module_ranges):
        places_m = min(max_module, cfg["database_places"] - mi * max_module)
        # 特征层掩码/初始化（M1/M2 与 SNN 同拓扑同分布；M0 稠密默认初始化）
        mask, init, freeze = None, "default", False
        if variant in ("m1", "m2"):
            mask = build_feature_mask(in_dim, in_dim * 2, seed=cli.seed * 100 + mi)
            init = "masked_default"
            freeze = variant == "m2"
        model = MLPBaseline(in_dim, places_m, feature_mask=mask,
                            freeze_feature=freeze, init=init,
                            seed=cli.seed * 100 + mi)
        loader, n_samples = make_train_loader(
            cfg, transform, rng, places_m, cfg["batch_size"], cli.seed)
        idx_scale = max_module * filter_ * mi  # 复刻 VPRTempoTrain.py:402

        # 标签换算包装：全局索引 → 模内地点索引（复刻 VPRTempoTrain.py:433）
        class _LocalLabelLoader:
            def __iter__(self_inner):
                for x, y in loader:
                    yield x, torch.round((y - idx_scale) / filter_).long()

            def __len__(self_inner):
                return len(loader)

        # M0-long：winter 验证集早停。每 epoch 末用"已训模块 + 当前模块"
        # 在 winter 上算 R@1（GT 只覆盖已训地点；recallAtK 自动丢弃无匹配查询列）。
        # winter 仅作调参信号，与正式 eval 的 query_skip 无关——数据集与 GT 一律 skip=0
        eval_cb = None
        winter_ds = None
        if variant == "m0long":
            if winter_ds is None:
                cfg_winter = dict(cfg, query_skip=0)
                winter_ds = make_query_dataset(cfg_winter, transform, "winter",
                                               max_samples=cfg["query_places"])
            from vprtempo.src.gt import build_ground_truth
            from vprtempo.src.metrics import recallAtK

            def eval_cb(cur_model, _mi=mi, _models=models):
                cand = _models + [cur_model]
                covered = sum(m.n_places for m in cand)
                logits = query_logits(cand, winter_ds, device)
                S = logits.numpy().T
                GT = build_ground_truth(covered, S.shape[1],
                                        skip=0,
                                        filter=filter_,
                                        tolerance=cfg["GT_tolerance"])
                return recallAtK(S, GT, K=1)

        acc_hist, n_upd, best_ep = train_mlp(
            model, _LocalLabelLoader(), cfg["epochs"], cfg["lr"],
            cfg["weight_decay"], device,
            log_fn=lambda s, m=mi: print(f"  [module {m}]{s}"),
            eval_callback=eval_cb)
        best_epochs.append(best_ep)
        models.append(model)
        total_updates += n_upd
        train_accs.append(acc_hist)
        print(f"[run_mlp] module {mi}: places={places_m} samples={n_samples} "
              f"updates={n_upd} final_train_acc={acc_hist[-1]:.4f}")
    train_wall = time.time() - t_train0
    peak_mem = (torch.cuda.max_memory_allocated() / 1e9
                if device == "cuda:0" else None)

    # ---- 推理 + 评测（含延迟计量）----
    for m in models:
        m.eval()
    t0 = time.time()
    recall = evaluate(models, make_query_dataset(cfg, transform, query_dir,
                                                 max_samples=cfg["query_places"]),
                      cfg, device)
    eval_wall = time.time() - t0

    # 单张推理延迟（预热后计时，batch=1 模拟在线查询）
    qds = make_query_dataset(cfg, transform, query_dir, max_samples=110)
    ql = DataLoader(qds, batch_size=1, shuffle=False, num_workers=2)
    lat = []
    with torch.no_grad():
        for i, (x, _) in enumerate(ql):
            x = x.to(device).float().view(1, -1)
            if x.max() > 1.5:
                x = x / 255.0
            if device == "cuda:0":
                torch.cuda.synchronize()
            t1 = time.time()
            for m in models:
                m(x)
            if device == "cuda:0":
                torch.cuda.synchronize()
            if i >= 10:  # 前 10 张预热
                lat.append(time.time() - t1)
    mean_lat_ms = float(np.mean(lat) * 1000) if lat else None

    # ---- 成本统计 ----
    total_params = sum(m.count_params()[0] for m in models)
    nonzero_params = sum(m.count_params()[1] for m in models)
    flops = sum(m.forward_flops() for m in models)

    result = {
        "exp_id": exp_id, "variant": variant, "seed": cli.seed,
        "query_dir": query_dir, "device": device,
        "recallAtK": recall,
        "final_train_acc_per_module": [a[-1] for a in train_accs],
        "train_acc_curve_per_module": train_accs,
        "best_epoch_per_module": best_epochs,
        "cost": {
            "train_wall_s": round(train_wall, 2),
            "n_weight_updates": total_updates,
            "eval_wall_s": round(eval_wall, 2),
            "inference_latency_ms_per_query": (round(mean_lat_ms, 3)
                                               if mean_lat_ms else None),
            "peak_gpu_mem_gb": (round(peak_mem, 3)
                                if peak_mem is not None else None),
            "total_params": total_params,
            "nonzero_params": nonzero_params,
            "forward_flops_per_query": flops,
        },
        "config": cfg,
    }
    out_dir = RESULTS_DIR / exp_id / f"seed_{cli.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"__{query_dir}" if cli.query_override else ""
    out_file = out_dir / f"{exp_id}__seed{cli.seed}{suffix}__mlp.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    torch.save([m.state_dict() for m in models], out_dir / f"{exp_id}__seed{cli.seed}.pth")
    print(f"[run_mlp] Recall@K = {recall}")
    print(f"[run_mlp] cost = {result['cost']}")
    print(f"[run_mlp] 结果已写入 {out_file}")


if __name__ == "__main__":
    main()
