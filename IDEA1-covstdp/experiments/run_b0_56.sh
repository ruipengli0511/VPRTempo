#!/bin/bash
# B0·500地·56×56 SNN 基线 3 seeds（MLP 对照实验组合 C 的基线）
set -e
cd /home/ubuntu/projects/VPRTempo
for seed in 0 1 2; do
  echo "=== b0_500_56 seed $seed train ==="
  pixi run python IDEA1-covstdp/experiments/run_exp.py IDEA1-covstdp/phase1/configs/b0_500_56.json --train --seed $seed
  echo "=== b0_500_56 seed $seed eval ==="
  pixi run python IDEA1-covstdp/experiments/run_exp.py IDEA1-covstdp/phase1/configs/b0_500_56.json --eval --seed $seed
done
echo "ALL DONE"
