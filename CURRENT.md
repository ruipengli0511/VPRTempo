# CURRENT.md — VPRTempo 研究入口

最后更新：2026-11-13

## B10 原版地点输出冬季评价（2026-11-13）

`eval/original-winter` 从 B09 固定版本分出，仅增加原版地点输出的冬季评价。复用B04三个seed的原版权重、B09冬季缓存，先对拍全部2700夏季原版top5及分数，再评价3300地点／2700冬季查询。无新训练、无归一化描述子检索或冬季调参。入口 `IDEA1-covstdp/experiments/eval_original_winter.py`，配置 `IDEA1-covstdp/phase4/configs/original_winter.json`。

## B09 冬季冻结评价（2026-11-13）

`eval/winter-frozen` 从 B08 `46ab86eef28b312505a276c1bf7122802086c98a` 分出；仅新增冬季评价适配器与配置，不改训练或权重。沿用 B08 夏季锁定的 Full／Anchor 第2轮及冻结原feature，三seed共9组，3300参考地点／2700冬季查询，GT容差0。冬季不选轮次、不调参；空间分区仅用于诊断，不称为冬季开发集。

入口：`IDEA1-covstdp/experiments/eval_frozen_winter.py`；配置：`IDEA1-covstdp/phase4/configs/frozen_winter.json`。批次请求、提交状态和结果按开题项目的 B09 记录查阅。

## 研究范围

当前只维护两条输入路线，并使用统一的检索评价接口：

- **灰度像素**：原 VPRTempo 预处理后的 28×28 输入，784 维，`frontend=none`；
- **固定 Gabor**：当前 B5 滤波器组，`frontend=gabor`；P0/B5 checkpoint 对应 local WTA block=4、1152 维 conv 描述子。

研究对象是 feature 描述子检索及后续 feature 学习。Gabor conv 直通作为绕过 feature 层的参照，原版地点输出路径作为整体系统基线。不同取点必须在配置中用 `feature_point` 明确区分，不能混用历史数字。

## 分支职责

- `main`：原始 VPRTempo 稳定基线，固定于 `2e4adfc91eb188281a6613e5fffa245b0bed756a`，不接收研究改动。
- `research/base`：公共代码、灰度像素基线、统一描述子检索和评价入口。
- `base/gabor`：固定 Gabor/B5 工作基线。这里的“最佳”仅指已有 Nordland 配置与协议，不代表所有数据集上的绝对最优。
- `exp/<backend>`：后续短期实验分支；Gabor 实验从 `base/gabor` 开始，像素实验从 `research/base` 开始。

公共修改先进入 `research/base`，再通过正常 merge 合入 `base/gabor`。不在两个 base 中分别修改公共评价语义，不使用强推或历史重写。

## 机器与本地文件

- 工位机：`/home/ubuntu/projects/VPRTempo`，pixi 默认 CPU 环境；Nordland 数据位于 `/mnt/d/Data/datasets/vpr/Nordland`。
- GPU 工作站：`/root/VPRTempo`，双 RTX 4090；其环境和路径与工位机分开记录。
- `vprtempo/models/b5_500__seed0.pth` 是本地 checkpoint，不被 Git 跟踪；SHA256 为 `2f6d8736a5762608f4d0fd6428dd9a35aa57223d3fbcbf716f6349a501c3b6be`。
- `local_override.json`、原图、NPY、日志和新增 checkpoint 保持本地，不入库。工作站 s32a stash 仍需连通后单独保全。

## 活动入口

- 配置驱动训练/原版输出评价：`IDEA1-covstdp/experiments/run_exp.py`。
- 描述子检索：`IDEA1-covstdp/experiments/eval_retrieval.py`。
- 固定 Gabor 轻量自检：`IDEA1-covstdp/experiments/test_gabor_frozen.py`。
- 像素 B0 与 Gabor B5 主配置：`IDEA1-covstdp/phase3/configs/table1/`。
- F0–F3 feature STDP/ITP 对照：`IDEA1-covstdp/phase3/configs/f*_b5_*.json`。
- P0 block4 配置：`IDEA1-covstdp/phase2/configs/b5_500.json`。
- conv-only 状态、P0 对拍和工件索引：`IDEA1-covstdp/handoff/`。

`run_exp.py` 不再自动删除同名模型；训练目标已存在时会拒绝运行，需使用新的 `exp_id`、seed 或模型名。

## 关键历史结果及边界

- 固定 Gabor/B5 在 3300 地轨 A 为 R@1 `0.717±0.006`，B0 为 `0.597±0.006`；该整体差值包含历史容量等差异，不能全部归因于结构编码。
- F 组结果支持：阈值适应是当前稳定收益的重要来源；已测试的 feature STDP 权重更新没有稳定增量。结论限于已有结构、初始化和训练预算。
- Gabor conv 直通是独立参照，不能解释为大 feature 层的收益；B0 的 feature 读出与 B5 的 conv 直读取点不同。
- HOG、MLP、BCM、B6 和其他前端探索已退出活动树，不作为当前研究路线。

当前结果摘要：

- `IDEA1-covstdp/results/table1_main.md`
- `IDEA1-covstdp/results/table1_main.json`
- `IDEA1-covstdp/results/f0f3_feature_ablate.md`
- `IDEA1-covstdp/results/s29_b5_gabor.md`

## 本次收束记录

- 收束前完整树：`archive/work-task-feature-pre-convergence-20261109` → `c7d9152ec2910bfcc2094a13b3951b5e15558a4a`。
- 更早的 11 个 `archive/*` tag 继续保留；旧代码、原始结果、日志、相似度矩阵和证据包可从归档读取，不复制回活动树。
- 保留代码已删除废弃前端分发、Conv-STDP/BCM/R1 死代码和旧 CLI；feature 层原有 STDP/ITP 对照不变。
- 代码与活动树精简提交：`99bd21806fe3eeb832e2fdcab5a2c463b35be164`；1058 个路径发生收束，历史均可由归档 tag 取回。
- 最终回归：19 份实验配置加载通过；保留 Python 编译与 CLI 检查通过；B5 核/冻结权重/ITP/1152 维前向通过；P0 24/24 样本一致，`max|Δ|=0`；模型覆盖保护通过且 checkpoint 哈希不变。
- 真实 500 地评价：像素 feature R@1=0.950、原版输出 0.920；B5 conv 直通 0.906、feature 0.894、原版输出 0.880。临时结果未入库，未启动训练。

新结果必须区分实际训练代码 SHA 与结果发布 SHA；历史运行 SHA 缺失时记“未知”，不补造。
