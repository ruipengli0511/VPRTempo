#!/usr/bin/env python
# ================================================================================
# diag_f0f3_mech.py — IDEA1 S3.6 机制诊断：发放率分布（B2）+ 描述子稀疏度（B3）
#
# 假设链（B1 轨 A 反转后待验）：
#   ITP = 逐神经元增益控制（消灭死/饱和通道）→ 描述子几何可用（轨 B F1≈F3）
#   STDP = 稀疏/低干扰码结构 → spike-forcing 单遍监督可读出（轨 A F3≫F1）
# 本脚本对 F0–F3 模型跑数据库前向，统计：
#   B2  每神经元发放率分布（死通道<1% / 饱和>99% 占比）、阈值分布、
#       发放率 vs ITP 目标率的秩相关（ITP 是否在干活）
#   B3  每描述子激活计数分布、全零描述子占比、零元素占比（clamp 零拒绝强度）
# 产出：results/f0f3_mech/<exp>__seed<s>__mech.json + fig_f0f3_mech.png（seed 0）
#
# 用法（仓库根目录）：
#   pixi run python IDEA1-covstdp/experiments/diag_f0f3_mech.py --scale 500 --seeds 0 1 2
# ================================================================================
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_exp import load_config, RESULTS_DIR  # noqa: E402
from eval_retrieval import build_inference_models  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CFG_DIR = "IDEA1-covstdp/phase3/configs"
MECH_DIR = RESULTS_DIR / "f0f3_mech"


def forward_collect(models, dataset):
    """逐样本前向，收集 feature 层 pre-clamp / post-clamp 输出。返回 (Z, D, thr)。"""
    import vprtempo.src.blitnet as bn
    from vprtempo.src import conv_frontend as cf

    use_conv = getattr(models[0], 'frontend', 'none') != 'none' and hasattr(models[0], 'conv_layer')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    Z, D = [], []
    with torch.no_grad():
        for spikes, _ in loader:
            if use_conv:
                spikes = cf.conv_forward(models[0].conv_layer, spikes)
            m = models[0]
            w_dev = m.feature_layer.w.weight.device
            z = m.feature_layer.w(spikes.to(w_dev))          # pre-clamp 净输入
            d = bn.clamp_spikes(z, m.feature_layer)           # post-clamp 描述子
            Z.append(z.cpu())
            D.append(d.cpu())
    thr = models[0].feature_layer.thr.detach().cpu()
    return torch.cat(Z, dim=0), torch.cat(D, dim=0), thr   # [N,H], [N,H], [1,H]


def rank_corr(a, b):
    """Spearman 秩相关（torch 实现，无 scipy 依赖）。"""
    def rank(x):
        order = x.argsort()
        r = torch.empty_like(order, dtype=torch.float32)
        r[order] = torch.arange(len(x), dtype=torch.float32)
        return r
    ra, rb = rank(a), rank(b)
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def per_variant_stats(Z, D, thr):
    N, H = D.shape
    fire = (D > 0).float().mean(dim=0)          # 每神经元发放率 [H]
    thr_v = thr.reshape(-1).float()
    active = (D > 0).sum(dim=1).float()          # 每描述子激活计数 [N]
    return {
        "n_images": int(N), "n_neurons": int(H),
        "neuron_fire_rate": fire.tolist(),
        "thr": thr_v.tolist(),
        "fire_dead_frac(<1%)": float((fire < 0.01).float().mean()),
        "fire_saturated_frac(>99%)": float((fire > 0.99).float().mean()),
        "fire_rate_mean": float(fire.mean()),
        # ITP 目标率沿神经元序号单调线性分配（blitnet.py:164-167）→
        # 秩相关直接用序号作目标秩，免依赖推理模型不保存的 fire_rate
        "fire_rate_vs_target_spearman": rank_corr(fire, torch.arange(H, dtype=torch.float32)),
        "thr_mean": float(thr_v.mean()), "thr_std": float(thr_v.std()),
        "desc_active_mean": float(active.mean()), "desc_active_std": float(active.std()),
        "desc_active_p50": float(active.median()),
        "zero_descriptor_frac": float((active == 0).float().mean()),
        "zero_entry_frac": float((D == 0).float().mean()),
        "desc_amplitude_mean": float(D.mean()),
        "pre_z_std_mean": float(Z.std(dim=0).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="S3.6 机制诊断：发放率分布 + 稀疏度")
    ap.add_argument("--scale", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", type=str, nargs="+", default=["f0", "f1", "f2", "f3"])
    cli = ap.parse_args()

    assert Path.cwd() == REPO_ROOT, "必须从仓库根目录运行"
    MECH_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = {}   # (variant, seed) -> stats
    for v in cli.variants:
        cfg = load_config(f"{CFG_DIR}/{v}_b5_{cli.scale}.json", cli.seeds[0])
        from vprtempo.src.dataset import CustomImageDataset, ProcessImage
        patch_norm = cfg.get("patch_norm", "on") == "on"
        transform = ProcessImage([int(x) for x in cfg["dims"].split(",")], cfg["patches"],
                                 patch_norm=patch_norm)
        db_csvs = [os.path.join('./vprtempo/dataset', f'{cfg["dataset"]}-{d}.csv')
                   for d in cfg["database_dirs"].split(',')]
        db_dataset = CustomImageDataset(
            annotations_file=db_csvs, base_dir=cfg["data_dir"],
            img_dirs=cfg["database_dirs"].split(','), transform=transform,
            filter=cfg["filter"], skip=cfg.get("skip_db", cfg["skip"]), test=False,
            max_samples=cfg["database_places"])

        for seed in cli.seeds:
            cfg["seed"] = seed
            models = build_inference_models(cfg)
            Z, D, thr = forward_collect(models, db_dataset)
            stt = per_variant_stats(Z, D, thr)
            stt["exp_id"] = f"{v}_b5_{cli.scale}"
            stt["seed"] = seed
            all_stats[(v, seed)] = stt
            out = MECH_DIR / f"{v}_b5_{cli.scale}__seed{seed}__mech.json"
            out.write_text(json.dumps(stt, ensure_ascii=False, indent=1))
            print(f"[diag] {v} seed{seed}: zero_desc={stt['zero_descriptor_frac']:.3f} "
                  f"zero_entry={stt['zero_entry_frac']:.3f} "
                  f"dead={stt['fire_dead_frac(<1%)']:.3f} sat={stt['fire_saturated_frac(>99%)']:.3f} "
                  f"ρ(fire,target)={stt['fire_rate_vs_target_spearman']:.3f}")

    # ---------------- 汇总图（seed 0）----------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = dict(f0="#888888", f1="#2196F3", f2="#FF9800", f3="#E91E63")
    labels = dict(f0="F0 frozen", f1="F1 ITP-only", f2="F2 STDP-only", f3="F3 full")

    ax = axes[0, 0]  # 发放率经验 CDF
    for v in cli.variants:
        fr = np.sort(np.array(all_stats[(v, 0)]["neuron_fire_rate"]))
        ax.plot(fr, np.linspace(0, 1, len(fr)), color=colors[v], label=labels[v], lw=2)
    ax.set_xlabel("per-neuron firing rate"); ax.set_ylabel("CDF")
    ax.set_title("B2a 发放率分布（经验 CDF）"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]  # 阈值分布
    for v in cli.variants:
        ax.hist(all_stats[(v, 0)]["thr"], bins=60, histtype="step", color=colors[v],
                label=labels[v], lw=2, density=True)
    ax.set_xlabel("threshold θ"); ax.set_ylabel("density")
    ax.set_title("B2b 阈值分布"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]  # 每描述子激活计数（箱线）
    data = [[all_stats[(v, s)]["desc_active_mean"] for s in cli.seeds] for v in cli.variants]
    bp = ax.boxplot(data, tick_labels=[labels[v] for v in cli.variants], patch_artist=True)
    for patch, v in zip(bp["boxes"], cli.variants):
        patch.set_facecolor(colors[v]); patch.set_alpha(0.5)
    ax.set_ylabel("active entries per descriptor (mean)")
    ax.set_title("B3a 描述子激活计数（3 seeds）"); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]  # 零描述子占比 + 死/饱和通道占比（分组柱状）
    x = np.arange(len(cli.variants)); w = 0.25
    zd = [np.mean([all_stats[(v, s)]["zero_descriptor_frac"] for s in cli.seeds]) for v in cli.variants]
    dead = [np.mean([all_stats[(v, s)]["fire_dead_frac(<1%)"] for s in cli.seeds]) for v in cli.variants]
    sat = [np.mean([all_stats[(v, s)]["fire_saturated_frac(>99%)"] for s in cli.seeds]) for v in cli.variants]
    ax.bar(x - w, zd, w, label="zero descriptor", color="#333333")
    ax.bar(x, dead, w, label="dead neuron (<1%)", color="#2196F3")
    ax.bar(x + w, sat, w, label="saturated neuron (>99%)", color="#E91E63")
    ax.set_xticks(x); ax.set_xticklabels([labels[v] for v in cli.variants])
    ax.set_title("B3b 零拒绝 vs 工作点异常占比"); ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"S3.6 mechanism diagnostics ({cli.scale} places, B2 firing-rate / B3 sparsity)")
    fig.tight_layout()
    fig.savefig(MECH_DIR / "fig_f0f3_mech.png", dpi=150)
    print(f"[diag] 图已写入 {MECH_DIR/'fig_f0f3_mech.png'}")


if __name__ == "__main__":
    main()
