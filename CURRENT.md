本文档是当前唯一入口说明，最后更新 2026-11-09

# CURRENT.md — VPRTempo / IDEA1-covstdp 当前状态

## 1. 当前研究目标

- 固定 **Gabor（B5 手工 Gabor 前端）为工作基准**，研究 VPR 地点关系如何进入 feature 层学习。
- 分支职责（日常口径）：`main` = 经核验稳定入口；`work/task-feature` = 唯一活动开发分支。旧分支不再使用，已全部转为 `archive/*` 存档 tag（见文末 tag 列表，共 11 个）。

## 2. 机器角色

- **本机 = 工位机**：代码 `/home/ubuntu/projects/VPRTempo`，CPU 环境（pixi 默认环境），数据 `/mnt/d/Data/datasets/vpr/Nordland`。
- **GPU 工作站**：`/root/VPRTempo`（容器内路径），双 RTX 4090，SSH 别名 `vpr-workspace`（端口 8024）/ `vpr-host`（端口 22），pixi cuda 环境。
- 两台机器的路径与环境各自独立，引用时不混称。

## 3. 当前代码版本

- 当前 HEAD：`d7a4ff34547821588c1023cedc180ee65375e6bc`（d7a4ff3）。
- 口径纪律：所有新结果须区分**训练使用 SHA** 与**发布结果 SHA**；历史记录中缺失的 SHA 记"未知"，不补造。

## 4. 当前可用入口

- **P0 验收材料**：`IDEA1-covstdp/handoff/P0_2026-11-08/`（`p0_forward_acceptance.py`、`样本与前向验收.json`、`P0_样本三联图.png`、`工作站状态与复用差异.md`、`P1实施建议.md`）。
- **B5 前端检索入口**：`IDEA1-covstdp/experiments/eval_retrieval.py` + `IDEA1-covstdp/phase2/configs/b5_500.json` + `feature_point=conv`，模型 `vprtempo/models/b5_500__seed0.pth`。
- **conv-only 工件**：`IDEA1-covstdp/handoff/conv_only_state_b5_500__seed0.json`。
- **工件索引**：`IDEA1-covstdp/handoff/artifact_index.json`。

## 5. 关键历史结果及适用范围

1. **B5 手工 Gabor 前端**：3300 地轨A R@1 0.717±0.006 vs B0 0.597±0.006（+12.0pt；3 seeds；Nordland 28×28）——适用范围：会议规模、轨A 完整系统口径；500 地迭代档同向（0.980 vs 0.930）。
2. **B2 从零 Conv-STDP 无增益**：3300 地轨A B2−B0 仅 +0.003±0.006，轨B 在 3300 地为负——从零学习未带来稳定增量。
3. **广播/密集层权重可塑性无益或有害**：F 组轨B 相对锚点无增量；feat-BCM 在 feature 层系统性崩塌（3300 地轨A 全部 <0.14）。
4. **ITP 实为单调阈值去抑制**：冻结诊断中阈值均值 0.27→0.006、发放率 1.6%→45.9%；`layer.x` 语义已核查——非发放率自适应机制，引用按此口径。
5. **MLP-BP 对照（M2）**：等呈现量下严重欠训；充分训练后接近但不及 SNN；随机特征 + BP 读出与 SNN 接近。
6. **HOG 直通**：显著弱于 Gabor，且随规模退化。
7. **规模反转三例**：池化粒度、freesign、BCM——500 地结论不可直接外推 3300 地。
8. **指标注意**：P@100%R 结构上等价 R@1；R@100%P 为条件召回，不与 R@K 混用。
9. **F 组口径**：按执行方报告记录，GPT 未独立验证，不写严格等价。

结果索引：`IDEA1-covstdp/results/table1_main.md`、`IDEA1-covstdp/results/f0f3_feature_ablate.md`、`IDEA1-covstdp/results/mlp_comparison.md`、`IDEA1-covstdp/results/fig2_kernels.png`；档案 `VPRTempo_历史探索实验档案_2026-10-12_v3.md`；证据包 `VPRTempo_证据包_2026-10-12/`。

## 6. 当前任务状态

- **本轮已完成**：旧分支 `archive/*` tag 存档（11 个，见文末）、A 组材料入库、`handoff/artifact_index.json` 工件索引、本 CURRENT.md。
- **剩余事项**：P1a（Gabor 描述子缓存 + 地点关系学习 + 限定短训练）；源码精简清单待 GPT 审阅；ORC/winter 臂未跑；工作站 s32a stash 保全待连通；证据包 ZIP 与目录差异待裁决。

## 附：archive/* 存档 tag 列表（11 个）

- `archive/feat-convstdp-base-20261109`
- `archive/feat-convstdp-bcm-feature-20261109`
- `archive/feat-convstdp-f0f3-feature-ablate-20261109`
- `archive/feat-convstdp-s210-b6-20261109`
- `archive/feat-convstdp-s2122-forward-wta-20261109`
- `archive/feat-convstdp-s2324-learning-20261109`
- `archive/feat-convstdp-s25-integration-20261109`
- `archive/feat-convstdp-s29-gabor-20261109`
- `archive/feat-convstdp-s31-kernel-viz-20261109`
- `archive/feat-mlp-baseline-20261109`
- `archive/main-baseline-20260902`

## 整理记录（2026-11-09）

- **归档提交**：`f50b00ae25e1d97f9c17a9b7f6e65e2074c86872` —— A 组历史材料入库（P0 五件、档案 v1/v2/v3 以 v3 为入口、证据包整包）、conv-only 工件（block=4 口径）+ 工件索引（3 工件）+ P0 重建对拍（24 样本 max|Δ|=0）；11 个 `archive/*` tag 已全部推送并读回核验。
- **第 1 批精简提交**：`29358a3` —— 按 `handoff/cleanup_batch1_2026-11-09.md` 移除 22 项 MLP 执行代码/配置（22/22，无暂停项；逐项 blob 与基点 d7a4ff 一致、无反向依赖）；历史封存于 `archive/feat-mlp-baseline-20261109`（MLP 原始结果 JSON 与 `results/mlp_comparison.md` 保留在活动树）。
- **回归结果**：前置检查 22/22 通过；删后无残留引用、关键模块 py_compile 通过；`main.py --help` 入口正常（frontend 全选项含 none/gabor）；B5 配置加载 OK（frontend=gabor, wta_block=4）；frontend=none 配置构造 OK；VPRTempo/VPRTempoTrain/eval_retrieval/eval_hog 导入 OK；P0 固定样本短前向回归 24/24 all_match、max|Δ|=0.0（前端/阈值/评价语义未变，未训练）。
- **未决**：工作站 s32a stash 保全（SSH 不可达，未触碰）；证据包 ZIP 删除待远端核验后执行；老分支删除待 GPT 第 2 批清单。
