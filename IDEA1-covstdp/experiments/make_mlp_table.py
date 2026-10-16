#!/usr/bin/env python
# ================================================================================
# make_mlp_table.py — MLP 对照实验汇总表
#
# 扫描 results/mlp_*/seed_*（MLP 臂）与 results/t1_b0_500 / t1_b0_3300 /
# results/b0_500_56（SNN 基线），生成 results/mlp_comparison.md：
#   每个规模×口径一张表，行=实验臂，列=Recall@K mean±std + 训练墙钟 +
#   推理延迟 + 参数量/非零数 + FLOPs
#
# 用法：pixi run python IDEA1-covstdp/experiments/make_mlp_table.py
# ================================================================================
import glob
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "IDEA1-covstdp" / "results"

# SNN 基线来源（exp_id → 规模×口径标签）
SNN_BASELINES = {
    "t1_b0_500": "A·500地·28×28",
    "t1_b0_3300": "B·3300地·28×28",
    "b0_500_56": "C·500地·56×56",
}
# MLP 臂 → 规模×口径标签
MLP_GROUPS = {
    "A·500地·28×28": ["mlp_m0_500_28", "mlp_m0long_500_28",
                    "mlp_m1_500_28", "mlp_m2_500_28"],
    "B·3300地·28×28": ["mlp_m0_3300_28", "mlp_m0long_3300_28",
                      "mlp_m1_3300_28", "mlp_m2_3300_28"],
    "C·500地·56×56": ["mlp_m0_500_56", "mlp_m0long_500_56",
                     "mlp_m1_500_56", "mlp_m2_500_56"],
}
ARM_LABELS = {"m0": "M0 稠密MLP", "m0long": "M0-long 充分训练",
              "m1": "M1 稀疏MLP(同拓扑)", "m2": "M2 冻特征+读出",
              "b0": "B0 SNN (VPRTempo)"}
KS = [1, 5, 10, 15, 20, 25]


def load_seed_results(exp_id, suffix="__mlp"):
    """读 results/<exp_id>/seed_*/ 下的结果 JSON，按 seed 排序"""
    rows = []
    for f in sorted(glob.glob(str(RESULTS_DIR / exp_id / "seed_*" / "*.json"))):
        name = os.path.basename(f)
        if suffix == "__mlp" and "__mlp" not in name:
            continue
        if suffix == "__eval" and "__eval" not in name:
            continue
        if "__winter__" in name:  # 调参产物不进表
            continue
        with open(f) as fp:
            rows.append(json.load(fp))
    return rows


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def arm_row(label, rows, is_snn):
    """生成一行的 Markdown 单元格"""
    if not rows:
        return f"| {label} | 缺失 |" + " — |" * (len(KS) + 4)
    rec = {K: [r["recallAtK"].get(str(K), r["recallAtK"].get(K))
               for r in rows if "recallAtK" in r] for K in KS}
    cells = [mean_std(rec[K]) for K in KS]
    if is_snn:
        cost = ["—", "—", "—", "—", "—"]
    else:
        train_s = [r["cost"]["train_wall_s"] for r in rows if "cost" in r]
        lat = [r["cost"]["inference_latency_ms_per_query"] for r in rows if "cost" in r]
        params = rows[0]["cost"]["total_params"] if rows and "cost" in rows[0] else None
        nz = rows[0]["cost"]["nonzero_params"] if rows and "cost" in rows[0] else None
        flops = rows[0]["cost"]["forward_flops_per_query"] if rows and "cost" in rows[0] else None
        cost = [mean_std([s / 60 for s in train_s]) + " min" if train_s else "—",
                mean_std(lat) + " ms" if lat else "—",
                f"{params/1e6:.2f}M" if params else "—",
                f"{nz/1e6:.2f}M" if nz else "—",
                f"{flops/1e6:.2f}M" if flops else "—"]
    return f"| {label} | n={len(rows)} | " + " | ".join(cells + cost) + " |"


def main():
    lines = ["# MLP-BP vs VPRTempo (SNN) 对照实验汇总",
             "",
             "> 由 `experiments/make_mlp_table.py` 自动生成。",
             "> MLP 臂：训练=spring+fall，查询=summer；指标为 3 seeds mean±std。",
             "> 成本列：训练墙钟(min) / 单张查询延迟(ms) / 总参数 / 非零参数 / 单次前向FLOPs。",
             "> SNN 基线的成本列留空（其训练/推理口径见 run_exp.py 归档日志）。",
             ""]
    header = ("| 实验臂 | n | R@1 | R@5 | R@10 | R@15 | R@20 | R@25 |"
              " 训练时间 | 延迟 | 参数 | 非零 | FLOPs |")
    sep = "|" + "---|" * 12

    for group, mlp_exps in MLP_GROUPS.items():
        lines += [f"\n## {group}", "", header, sep]
        snn_id = [k for k, v in SNN_BASELINES.items() if v == group]
        if snn_id:
            rows = load_seed_results(snn_id[0], suffix="__eval")
            lines.append(arm_row(ARM_LABELS["b0"], rows, is_snn=True))
        for exp in mlp_exps:
            variant = exp.split("_")[1]
            rows = load_seed_results(exp, suffix="__mlp")
            lines.append(arm_row(ARM_LABELS[variant], rows, is_snn=False))

    out = RESULTS_DIR / "mlp_comparison.md"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已写入 {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
