# idea_explore — 论文卖点增强探索工作区

> 2026-09-13 建立｜母会话：论文卖点增强方向探索
> 地位：PLAN.md 阶段 3 之上的**探索层**（非正式实验档），不改变 PLAN §6 已锁定的 Gate 判定与主表阵容。
> 纪律：沿用 PLAN §8 全部规范（三档协议、预注册判据、B0 回归、配置即代码、默认行为不变）。

## 探索方向索引

| # | 文档 | 一句话 | 状态 |
|---|---|---|---|
| 1 | [01_readout_adaptation.md](01_readout_adaptation.md) | 读出适配：把轨B→轨A 的 8.3pt 差距从软肋变成"读出瓶颈被定位并修复"的贡献 | 设计中 |
| 2 | [02_online_adaptation.md](02_online_adaptation.md) | 无标签在线适应旗舰实验：局部可塑性对 BP 方法的差异化必杀（winter 部署场景） | 设计中 |
| 3 | [03_winter_main_table.md](03_winter_main_table.md) | winter 跨季节主表：比 summer 难的硬骨头场景，回应"单季节方向"质疑 | 设计中 |

## 三方向的关系

```
③ winter 主表 ──共享 winter 数据管线── ② 在线适应
        │                                    │
        └──── ② 的适应增益行叠加在 ③ 的 A0 对照上 ───┘

① 读出适配 独立线（诊断对象是所有变体共用的 spike-forcing 读出，
  修复后直接惠及 ③ 的系统级数字 → 建议最优先执行）
```

依赖顺序建议：**① →（winter CSV + ② 单 arm 试点）→ ③ 确认档**。

## 环境事实（2026-09-13 盘点）

- 本地数据：`/mnt/e/Datasets/Nordland/`（spring/fall/summer/winter 各 35768 张，**winter 图像已就位，缺 winter CSV**）。
- 本地 GPU：GTX 1650 4GB（可用但慢）；工作站双 RTX 4090 可用（规程见 PLAN §8 第 6 条）。
- 实验模型 .pth 不在本地（在工作站，gitignore）；t1_* 正式档无 npy（已排除），早期目录有部分 npy。
- `local_override.json`（gitignored）需填 `{"data_dir": "/mnt/e/Datasets/Nordland"}`。

## 结果落盘约定

- 探索结果先落 `idea_explore/results/<exp_id>/seed_<n>/`，**与主 `results/` 隔离**，避免污染主表；
- 判定"可进论文"后按 S3.2 程序升正式档（重跑、迁回 `results/`、exp_id 加前缀隔离）；
- 每个探索 run 同样落盘配置全文 + 诊断 JSON（PLAN §8 锁定清单第 6 条）。
