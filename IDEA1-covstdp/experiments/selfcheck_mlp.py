#!/usr/bin/env python
# ================================================================================
# selfcheck_mlp.py — mlp_baseline.py 单元自检
#
# 检查项：
#   1. build_feature_mask 密度 ≈ 1-(1-0.1)(1-0.5) = 0.55（偏差 <2%）
#   2. blitnet_init_ 非零率与掩码一致、列 L1 归一化生效
#   3. MLPBaseline 前向形状正确
#   4. feature_mask 在 optimizer.step 后仍保持稀疏拓扑（apply_mask 生效）
#   5. freeze_feature 时特征层梯度为 None
#   6. FLOPs/参数统计合理
#
# 用法：pixi run python IDEA1-covstdp/experiments/selfcheck_mlp.py
# ================================================================================
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "IDEA1-covstdp" / "src"))

from mlp_baseline import (MLPBaseline, blitnet_init_, build_feature_mask,  # noqa: E402
                          train_mlp)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        sys.exit(1)


def main():
    in_dim, n_places = 784, 500
    feat_dim = in_dim * 2

    # 1. 掩码密度
    mask = build_feature_mask(in_dim, feat_dim, seed=0)
    density = float(mask.mean())
    expected = 1 - (1 - 0.1) * (1 - 0.5)  # 0.55
    check("掩码密度", abs(density - expected) < 0.02,
          f"density={density:.4f} expected≈{expected:.4f}")

    # 2. blitnet_init_ 非零率 + L1 列归一化（仅供消融参考；BP+ReLU 下为死亡初始化）
    w = torch.empty(feat_dim, in_dim)
    blitnet_init_(w, seed=0)
    nz = float((w != 0).float().mean())
    check("blitnet_init 非零率", abs(nz - expected) < 0.02, f"nonzero={nz:.4f}")
    col_l1 = w.abs().sum(dim=0)
    connected = col_l1[col_l1 > 0]
    check("L1 列归一化", torch.allclose(connected, torch.ones_like(connected), atol=1e-5),
          f"max_dev={float((connected - 1).abs().max()):.2e}")
    # 符号结构：正值≈兴奋比例，负值≈抑制比例
    pos_frac = float((w > 0).float().mean())
    check("符号结构", 0.05 < pos_frac < 0.15, f"pos_frac={pos_frac:.4f} (期望≈0.1)")

    # 2b. masked_default 初始化：拓扑与掩码一致且 ReLU 后不死亡
    m_md = MLPBaseline(in_dim, 10, feature_mask=mask, init="masked_default")
    nz_md = float((m_md.feature.weight != 0).float().mean())
    check("masked_default 非零率", abs(nz_md - density) < 1e-6, f"nonzero={nz_md:.4f}")
    with torch.no_grad():
        h_md = torch.relu(m_md.feature(torch.rand(8, in_dim)))
    check("masked_default 前层不死亡", float((h_md > 0).float().mean()) > 0.1,
          f"frac_pos={float((h_md > 0).float().mean()):.3f}")

    # 3. 前向形状
    model = MLPBaseline(in_dim, n_places)
    x = torch.rand(4, in_dim)
    out = model(x)
    check("前向形状", out.shape == (4, n_places), f"out={tuple(out.shape)}")

    # 4. step 后掩码保持
    model_m1 = MLPBaseline(in_dim, n_places, feature_mask=mask)
    opt = torch.optim.Adam(model_m1.parameters(), lr=1e-2)
    loss = model_m1(torch.rand(8, in_dim)).sum()
    loss.backward()
    opt.step()
    model_m1.apply_mask()
    viol = float((model_m1.feature.weight * (1 - mask)).abs().max())
    check("step后掩码保持", viol == 0.0, f"violation={viol}")
    nz_m1 = float((model_m1.feature.weight != 0).float().mean())
    check("M1非零率=掩码密度", abs(nz_m1 - density) < 1e-6, f"nonzero={nz_m1:.4f}")

    # 5. freeze_feature
    model_m2 = MLPBaseline(in_dim, n_places, freeze_feature=True)
    loss = model_m2(torch.rand(8, in_dim)).sum()
    loss.backward()
    check("freeze特征层无梯度", model_m2.feature.weight.grad is None)
    check("读出层有梯度", model_m2.readout.weight.grad is not None)

    # 6. 成本统计
    total, nonzero = MLPBaseline(in_dim, n_places, feature_mask=mask).count_params()
    flops = MLPBaseline(in_dim, n_places).forward_flops()
    expected_total = in_dim * feat_dim + feat_dim * n_places
    check("参数总量", total == expected_total, f"total={total}")
    check("FLOPs合理", flops == 2 * in_dim * feat_dim + 2 * feat_dim * n_places,
          f"flops={flops/1e6:.2f}M")

    # 7. train_mlp 烟测（合成数据可学习）
    torch.manual_seed(0)
    syn_x = torch.rand(200, in_dim)
    syn_y = (syn_x[:, 0] > 0.5).long()
    loader = [(syn_x[i:i + 32], syn_y[i:i + 32]) for i in range(0, 200, 32)]
    model_syn = MLPBaseline(in_dim, 2)
    acc_hist, n_upd, best_ep = train_mlp(model_syn, loader, epochs=10, lr=3e-3,
                                         weight_decay=0.0, device="cpu", log_fn=lambda s: None)
    check("合成数据可学习", acc_hist[-1] > 0.9, f"final_acc={acc_hist[-1]:.3f}")
    check("更新次数正确", n_upd == 10 * 7, f"n_updates={n_upd}")
    check("无早停时best_epoch=epochs", best_ep == 10, f"best_epoch={best_ep}")

    print("\n全部自检通过")


if __name__ == "__main__":
    main()
