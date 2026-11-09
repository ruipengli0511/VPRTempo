# GPT → 工位机 Kimi · 第1批 22 项精简清单（2026-11-09）

> 处置含义：**历史已归档，可从 `work/task-feature` 活动分支的当前文件树删除**。
> 普通 Git 删除提交，不是清除历史 / 删存档 tag / 销毁原件。
> 历史保留位置：`archive/feat-mlp-baseline-20261109`（→ d7a4ff34547821588c1023cedc180ee65375e6bc）。
> GPT 在线核对结论：22 个目标文件在归档提交 f50b00a 中的 blob 与审阅基点 d7a4ff 一致。

## 一、6 个 MLP 专属代码/脚本

| # | 路径 |
|---:|---|
| 1 | `IDEA1-covstdp/src/mlp_baseline.py` |
| 2 | `IDEA1-covstdp/experiments/run_mlp.py` |
| 3 | `IDEA1-covstdp/experiments/run_mlp_all.sh` |
| 4 | `IDEA1-covstdp/experiments/make_mlp_table.py` |
| 5 | `IDEA1-covstdp/experiments/selfcheck_mlp.py` |
| 6 | `IDEA1-covstdp/experiments/diag_lr_3300.py` |

## 二、16 个 MLP 专属配置（phase1/configs/mlp_*.json）

| # | 路径 |
|---:|---|
| 7 | `IDEA1-covstdp/phase1/configs/mlp_m0_3300_28.json` |
| 8 | `IDEA1-covstdp/phase1/configs/mlp_m0_500_28.json` |
| 9 | `IDEA1-covstdp/phase1/configs/mlp_m0_500_56.json` |
| 10 | `IDEA1-covstdp/phase1/configs/mlp_m0b1_500_28.json` |
| 11 | `IDEA1-covstdp/phase1/configs/mlp_m0long_3300_28.json` |
| 12 | `IDEA1-covstdp/phase1/configs/mlp_m0long_500_28.json` |
| 13 | `IDEA1-covstdp/phase1/configs/mlp_m0long_500_56.json` |
| 14 | `IDEA1-covstdp/phase1/configs/mlp_m1_3300_28.json` |
| 15 | `IDEA1-covstdp/phase1/configs/mlp_m1_500_28.json` |
| 16 | `IDEA1-covstdp/phase1/configs/mlp_m1_500_56.json` |
| 17 | `IDEA1-covstdp/phase1/configs/mlp_m2_3300_28.json` |
| 18 | `IDEA1-covstdp/phase1/configs/mlp_m2_500_28.json` |
| 19 | `IDEA1-covstdp/phase1/configs/mlp_m2_500_56.json` |
| 20 | `IDEA1-covstdp/phase1/configs/mlp_smoke100_m0.json` |
| 21 | `IDEA1-covstdp/phase1/configs/mlp_smoke100_m1.json` |
| 22 | `IDEA1-covstdp/phase1/configs/mlp_smoke100_m2.json` |

**只处理这 22 条精确路径**，不扩大为删除整个 phase1 / experiments / src 目录。

## 三、本批明确保留

- `results/mlp_comparison.md` 及 MLP 原始结果 JSON；
- 所有 `archive/*` tag；
- P0 材料、conv-only 工件、生成脚本及工件索引；
- Gabor、conv 前向、预处理、GT 和指标代码；
- `run_exp.py`、`eval_retrieval.py` 及 B5 配置；
- `main.py`、VPRTempo 原训练/推理及其兼容依赖；
- 原图、checkpoint、缓存和工作站 stash。

## 四、执行与验收（按序）

1. 前置检查：存档可取回（tag → d7a4ff）；逐项核对相对 d7a4ff 无变化、无新增依赖；不满足的项单独暂停。
2. 独立清理提交：本清单 + 删除 22 项，不夹带核心代码修正。
3. 静态与入口检查：剩余活动 Python/shell 无新增失效导入或调用；CURRENT 不指向被删执行入口（指向处改固定存档链接）。
4. 轻量回归：B5 配置加载、P0 固定样本短前向（`p0_rebuild_check.py`，24 样本）、状态检查；`frontend=none` 入口做导入/帮助/配置构造级检查（不训练）。
5. CURRENT 记录归档 SHA、精简 SHA、处理数量、暂停项、验收结果；推送后按 §八格式返回。

本批不删远端分支、不强推、不改写历史、不启动新正式训练。
