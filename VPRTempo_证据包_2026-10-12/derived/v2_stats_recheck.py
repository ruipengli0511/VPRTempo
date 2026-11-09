#!/usr/bin/env python
# v2 修订用统计复核脚本：从原始结果 JSON 全精度重算关键统计量
# 纯读取与汇总计算，不改动任何文件。输出机器可读清单（供证据包）。
import json
import statistics as st
from pathlib import Path

R = Path("IDEA1-covstdp/results")


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cell(exp_id, seed, suffix):
    return load(R / exp_id / f"seed_{seed}" / f"{exp_id}__seed{seed}__{suffix}.json")


def r1(d):
    if d is None:
        return None
    ra = d.get("recallAtK")
    if isinstance(ra, dict):
        return ra.get("1")
    return None


def r100(d):
    return None if d is None else d.get("recallAt100precision")


def p100(d):
    return None if d is None else d.get("precisionAt100recall")


def fmt(x, n=6):
    return "None" if x is None else f"{x:.{n}f}"


print("=" * 90)
print("A. B0 500 地轨B 逐 seed 全精度（回答 0.950/0.942/0.936 vs 汇总 0.942）")
print("=" * 90)
for seed in (0, 1, 2):
    d = cell("b0_pn_on", seed, "trackB")
    print(f"  seed{seed}: R@1={fmt(r1(d))} R@100%P={fmt(r100(d))} P@100%R={fmt(p100(d))}")
vals = [r1(cell("b0_pn_on", s, "trackB")) for s in (0, 1, 2)]
print(f"  全精度均值 = {st.mean(vals):.6f}（四舍五入 4 位 = {round(st.mean(vals),4)}；文档记 0.942）")
print(f"  样本标准差 = {st.stdev(vals):.6f}（文档记 0.006）")
d = cell("b0ref_trackb", 0, "trackB_feature_layer") or cell("b0ref_trackb", 0, "trackB")
if d:
    print(f"  b0ref_trackb（种子化参照）: R@1={fmt(r1(d))}（路径后缀见文件）")

print()
print("=" * 90)
print("B. block2 时代 B2 500 地轨B 0.894(CPU) vs 0.878(GPU) 溯源")
print("=" * 90)
for exp in ["b2_500_block2", "b2_500"]:
    for seed in (0,):
        for suf in ["trackB_conv", "trackB", "trackB_feature_layer"]:
            d = cell(exp, seed, suf)
            if d:
                print(f"  {exp}/seed{seed}/{suf}: R@1={fmt(r1(d))} device={d.get('device')} "
                      f"wall={d.get('wall_time_s')} 配置conv_epoch={d['config'].get('conv_epoch')} "
                      f"stdp={d['config'].get('conv_stdp_rate')}")
# s25 变体扫描时代的 b2_500_block2 CPU 数字只在汇总文档；检查 results 里是否有多个 JSON
import glob as _g
print("  b2_500_block2 目录内容:", [Path(x).name for x in _g.glob(str(R/"b2_500_block2"/"*/*"))])

print()
print("=" * 90)
print("C. F 组配对差值（同 seed 配对：F 组共享同 seed 初始化，见 verify_f*.json）")
print("=" * 90)
for scale in (500, 3300):
    print(f"--- {scale} 地 ---")
    series = {}
    for v in ("f0", "f1", "f2", "f3"):
        tb = [r1(cell(f"{v}_b5_{scale}", s, "trackB_feature_layer")) for s in (0, 1, 2)]
        ta = [r1(cell(f"{v}_b5_{scale}", s, "eval")) for s in (0, 1, 2)]
        r1p = [r100(cell(f"{v}_b5_{scale}", s, "trackB_feature_layer")) for s in (0, 1, 2)]
        series[v] = dict(tb=tb, ta=ta, r100=r1p)
        print(f"  {v}: 轨B R@1 {[fmt(x,4) for x in tb]} | 轨B R@100%P {[fmt(x,4) for x in r1p]} | 轨A {[fmt(x,4) for x in ta]}")
    for a, b in [("f1", "f0"), ("f2", "f0"), ("f3", "f1")]:
        diffs = [x - y for x, y in zip(series[a]["tb"], series[b]["tb"])]
        md = st.mean(diffs)
        sd = st.stdev(diffs)
        print(f"  {a}−{b}（轨B R@1，逐 seed 配对差值）: {[fmt(x,4) for x in diffs]} "
              f"→ 差值均值 {md:+.4f} ± 差值std {sd:.4f}（对照：非配对联合std口径见 v1）")
    # F1 vs F3 R@100%P 配对（3300 的关键观察）
    diffs = [x - y for x, y in zip(series["f1"]["r100"], series["f3"]["r100"])]
    print(f"  F1−F3（轨B R@100%P 逐 seed 差值）: {[fmt(x,4) for x in diffs]} → "
          f"均值 {st.mean(diffs):+.4f} ± {st.stdev(diffs):.4f}")

print()
print("=" * 90)
print("D. 主表关键对比的逐 seed 配对差值（conv 变体同 seed 共享 feature 层初始化）")
print("=" * 90)
def t1(v, scale, s, track="B"):
    if track == "B":
        suf = "trackB_feature_layer" if v == "b0" else "trackB_conv"
        d = cell(f"t1_{v}_{scale}", s, suf)
        return r1(d) if d else None
    d = cell(f"t1_{v}_{scale}", s, "eval")
    return r1(d) if d else None

for scale in (500, 3300):
    print(f"--- {scale} 地 ---")
    for a, b, note in [("b2", "b1", "Gate1"), ("b5", "b2", ""), ("b6a", "b5", "Gate1.5"),
                       ("b5bcm", "b5", ""), ("dct", "b5", ""), ("loggabor", "b5", "")]:
        pa = [t1(a, scale, s) for s in (0, 1, 2)]
        pb = [t1(b, scale, s) for s in (0, 1, 2)]
        if None in pa or None in pb:
            print(f"  {a}−{b}: 缺格 {pa} vs {pb}")
            continue
        diffs = [x - y for x, y in zip(pa, pb)]
        md, sd = st.mean(diffs), st.stdev(diffs)
        # 对照：v1/主表的非配对联合 std
        ja = st.stdev(pa)
        jb = st.stdev(pb)
        joint = (ja**2 + jb**2) ** 0.5
        print(f"  {a}−{b} {note}（轨B R@1）: 逐seed差 {[fmt(x,4) for x in diffs]} → "
              f"差值均值 {md:+.4f} ± 差值std {sd:.4f} | 对照联合std={joint:.4f}")
print()
print("注：B0 无 conv 前端、RNG 序列与其余变体不同——B0 参与的对比不适用同 seed 配对。")

print()
print("=" * 90)
print("E. 机器可读结果清单（写 /tmp/v2_inventory.json，供证据包 manifest 派生）")
print("=" * 90)
inv = []
for p in sorted(R.glob("**/seed_*/*.json")):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(d, dict) or "exp_id" not in d:
        continue
    inv.append({
        "path": str(p),
        "exp_id": d.get("exp_id"), "seed": d.get("seed"), "phase": d.get("phase"),
        "feature_point": d.get("feature_point"), "descriptor": d.get("descriptor"),
        "recallAt1": (d.get("recallAtK") or {}).get("1") if isinstance(d.get("recallAtK"), dict) else None,
        "recallAt100precision": d.get("recallAt100precision"),
        "precisionAt100recall": d.get("precisionAt100recall"),
        "wall_time_s": d.get("wall_time_s"), "device": d.get("device"),
        "config_frontend": (d.get("config") or {}).get("frontend"),
        "config_database_places": (d.get("config") or {}).get("database_places"),
        "config_skip": (d.get("config") or {}).get("skip"),
        "config_model_name": (d.get("config") or {}).get("model_name"),
    })
Path("/tmp/v2_inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1))
print(f"  清单条目数 = {len(inv)}")
