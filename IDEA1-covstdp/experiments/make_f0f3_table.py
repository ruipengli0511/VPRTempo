#!/usr/bin/env python
# ================================================================================
# make_f0f3_table.py — IDEA1 S3.6 F0–F3 汇总表生成
#
# 读取 results/ 下的轨 B 结果 JSON，出两张表：
#   表 1：F0–F3（feature_layer 描述子）+ Gabor 直通（conv 锚点）+ HOG 直通
#         × {500, 3300} 地，指标 R@1 / R@100%P / P@100%R（3 seeds mean±std）
#   表 2：预注册对比的 R@1 差值（F1−F0、F2−F0、F3−F1、F3−Gabor直通）
# 落盘 results/f0f3_feature_ablate.md；缺失格显示 —。
#
# 用法（仓库根目录）：pixi run python IDEA1-covstdp/experiments/make_f0f3_table.py
# ================================================================================
import json
import statistics as st
from pathlib import Path

RESULTS = Path("IDEA1-covstdp/results")
OUT = RESULTS / "f0f3_feature_ablate.md"
SEEDS = (0, 1, 2)


def load(exp_id, seed, suffix):
    p = RESULTS / exp_id / f"seed_{seed}" / f"{exp_id}__seed{seed}__{suffix}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def stats(exp_id, suffix, seeds=SEEDS):
    """返回 (dict(r1,r100,p100 显示串, r1_mean 原始均值), n)。缺任何 seed 返回 (None, 0)。"""
    r1, r100, p100 = [], [], []
    for s in seeds:
        d = load(exp_id, s, suffix)
        if d is None:
            return None, 0
        r1.append(d["recallAtK"]["1"])
        r100.append(d["recallAt100precision"])
        p100.append(d["precisionAt100recall"])
    fmt = lambda xs: f"{st.mean(xs):.3f}±{st.stdev(xs):.3f}" if len(xs) > 1 else f"{st.mean(xs):.3f}"
    return dict(r1=fmt(r1), r100=fmt(r100), p100=fmt(p100), r1_mean=st.mean(r1)), len(r1)


ROWS = [  # (显示名, exp_id 模板, suffix)
    ("F0  无权重+无阈值", "f{v}_b5_{scale}", "trackB_feature_layer"),
    ("F1  无权重+ITP",   "f{v}_b5_{scale}", "trackB_feature_layer"),
    ("F2  STDP+无阈值",  "f{v}_b5_{scale}", "trackB_feature_layer"),
    ("F3  STDP+ITP",     "f{v}_b5_{scale}", "trackB_feature_layer"),
]
VARIANTS = ["f0", "f1", "f2", "f3"]


def main():
    lines = ["# S3.6 feature 层可塑性分解（F0–F3）+ 手工描述子直通基线", ""]
    means = {}  # (scale, name) -> r1_mean

    for scale in (500, 3300):
        lines += [f"## {scale} 地（轨 B，feature_layer 描述子）", ""]
        lines += ["| 组 | R@1 | R@100%P | P@100%R |", "|---|---|---|---|"]
        cells = []
        for v in VARIANTS:
            name = dict(f0="F0 无权重+无阈值", f1="F1 无权重+ITP",
                        f2="F2 STDP+无阈值", f3="F3 STDP+ITP")[v]
            exp_id = f"{v}_b5_{scale}"
            d, n = stats(exp_id, "trackB_feature_layer")
            if d:
                lines.append(f"| {name} | {d['r1']} | {d['r100']} | {d['p100']} |")
                means[(scale, v)] = d["r1_mean"]
            else:
                lines.append(f"| {name} | — | — | — |")
            cells.append((exp_id, d is not None))
        # 直通基线
        for name, exp_id, suffix, seeds in [
            ("Gabor 前端直通（锚）", f"t1_b5_{scale}", "trackB_conv", SEEDS),
            ("HOG 直通（324 维）", f"hog_{scale}", "trackB_hog", (0,)),
        ]:
            d, n = stats(exp_id, suffix, seeds)
            key = "gabor_direct" if "t1_" in exp_id else "hog"
            if d:
                lines.append(f"| {name} | {d['r1']} | {d['r100']} | {d['p100']} |")
                means[(scale, key)] = d["r1_mean"]
            else:
                lines.append(f"| {name} | — | — | — |")
        lines.append("")

    lines += ["## 预注册对比（R@1 差值，mean 口径）", ""]
    lines += ["| 对比 | 500 地 | 3300 地 |", "|---|---|---|"]
    for scale in (500, 3300):
        pass
    pairs = [
        ("F1−F0（阈值适应的作用）", "f1", "f0"),
        ("F2−F0（权重学习的作用）", "f2", "f0"),
        ("F3−F1（有阈值时 STDP 增量）", "f3", "f1"),
        ("F3−Gabor直通（加 feature 层值不值）", "f3", "gabor_direct"),
    ]
    for label, a, b in pairs:
        cols = []
        for scale in (500, 3300):
            if (scale, a) in means and (scale, b) in means:
                cols.append(f"{means[(scale, a)] - means[(scale, b)]:+.3f}")
            else:
                cols.append("—")
        lines.append(f"| {label} | {cols[0]} | {cols[1]} |")
    lines.append("")
    lines += ["> HOG vs Gabor 直通对比见表 1（外部基线定位）；"
              "预注册效应判据：|ΔR@1|>2pt 且 3 seeds 同号。", ""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[make_f0f3_table] 已写入 {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
