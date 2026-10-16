#!/bin/bash
# ================================================================================
# run_mlp_all.sh — MLP 对照实验正式批跑（组合 A+B+C，4 臂 + m0b1 补充臂，3 seeds）
#
# 调参结论（winter，单 seed，组合 A）：
#   lr=0.001, wd=0 为 6 组网格最优（4 epoch 下所有组合都接近随机，
#   这是"等呈现量"本身的发现；m0long 用 winter 早停选 epoch）
# m0b1（batch=1）为计划外补充臂：与 SNN 等更新次数（4000 次）的对照，
#   回应"M0 batch=32 只有 128 次更新"的口径差距。
#
# 用法：bash IDEA1-covstdp/experiments/run_mlp_all.sh [combo]
#   combo ∈ {A, B, C, all}（默认 all）
# ================================================================================
set -e
cd /home/ubuntu/projects/VPRTempo
COMBO=${1:-all}
LR=0.001
WD=0
CFG=IDEA1-covstdp/phase1/configs

run_arm() { # $1=config $2=seed
  echo "=== $(basename $1 .json) seed $2 ==="
  pixi run python IDEA1-covstdp/experiments/run_mlp.py "$1" --seed "$2" --lr $LR --wd $WD
}

run_combo() { # $1=places $2=dims
  for v in m0 m0long m1 m2; do
    for seed in 0 1 2; do
      run_arm $CFG/mlp_${v}_$1_$2.json $seed
    done
  done
}

if [ "$COMBO" = "A" ] || [ "$COMBO" = "all" ]; then
  run_combo 500 28
  # m0b1 补充臂（batch=1，等更新次数）仅 28×28·500 地
  for seed in 0 1 2; do
    run_arm $CFG/mlp_m0b1_500_28.json $seed
  done
fi
if [ "$COMBO" = "B" ] || [ "$COMBO" = "all" ]; then
  run_combo 3300 28
fi
if [ "$COMBO" = "C" ] || [ "$COMBO" = "all" ]; then
  run_combo 500 56
fi
echo "ALL MLP RUNS DONE"
