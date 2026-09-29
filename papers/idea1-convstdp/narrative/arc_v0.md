# 叙事弧 v0 —— IDEA1: Conv 前端 for VPRTempo

> 版本：v0（2026-09-02）| 状态：**无任何实验数字，全部条件语气**
> 纪律：每条修辞性卖点必须能指到真实机制差异（见 §4 映射表）；预期结果一律"若…则…"。
> 叙事主线（已拍板）：**方法论优先**——双轨评测协议为骨，"无 BP 局部可塑性"定位为皮，效率为防御性佐证。
> 选择理由：对 Gate 结果最稳健——即使 Gate 2 失败（轨A 无增益），双轨协议恰好成为解释失配的分析工具，故事完整幸存；"空位优先"需要 Gate 1+2 双过，风险高。

---

## 1. 三句话故事线

### 中文版

1. **问题**：面向视觉场景识别（VPR）的脉冲神经网络（VPRTempo 系）把图像展平为全连接向量，丢弃了空间局部结构；而已有的卷积 SNN-VPR 方案（LoCS-Net）依赖端到端反向传播与多步脉冲仿真——"**无 BP 的局部可塑性能否为 VPR 学到空间特征**"这一问题尚无答案。
2. **主张**：我们提出一个**单步幅度域卷积前端**，以竞争性 Hebbian 可塑性（BLiTNet/VPRTempo STDP 规则在单步极限下的退化形式）无监督学习卷积核，插入 VPRTempo 的脉冲编码与特征层之间——无 BP、无多步仿真、与既有逐层训练框架完全兼容。
3. **证据（条件语气）**：通过**双轨评测协议**（轨A：经 spike-forcing 读出层的端到端 Recall@K；轨B：绕开读出层的原始特征检索）分离编码器与读出的贡献——若轨B 上学习核一致超越同初始化的随机核与同结构的手工 Gabor 组，则表明无 BP 可塑性确实学到了任务相关的空间结构；核的 Gabor 拟合优度与方向选择性分析将为此提供机制层面的定性证据。

### English draft（供摘要迭代用）

1. *Spiking neural networks for visual place recognition (VPR) flatten images into fully-connected vectors, discarding spatial structure; the only convolutional SNN for VPR instead relies on end-to-end backpropagation with multi-step simulation — whether BP-free local plasticity can learn spatial features for VPR remains open.*
2. *We insert a single-step, amplitude-domain convolutional frontend between spike encoding and the feature layer of VPRTempo, whose kernels are learned unsupervised by a competitive Hebbian rule — the single-step limit of the BLiTNet/VPRTempo STDP formulation — requiring neither backpropagation nor multi-step simulation.*
3. *A dual-track evaluation protocol — end-to-end recall through the supervised readout (Track A) versus readout-free feature retrieval (Track B) — isolates the encoder's contribution: should learned kernels consistently outperform matched random and hand-crafted Gabor frontends on Track B, BP-free plasticity demonstrably acquires task-relevant spatial structure.*

---

## 2. 审稿人第一反应预演（模拟 90 秒速读）

> 预演目的：模拟审稿人只读标题+摘要+图 1 时的本能反应，提前暴露叙事漏洞。

- **机器人/ VPR 背景审稿人**："又一个给 VPRTempo 加模块的工作？……哦，双轨评测有点意思，他们敢把 encoder 单独拆出来测。等等，28×28 这么小？——得看实验节解释。"（风险：输入尺寸质疑；应对：方法节一句话注明沿用 VPRTempo/VPRSNN 系口径，效率动机。）
- **神经形态背景审稿人**："STDP？单步幅度域没有 timing，这名字不严谨。"（**最高危反应**，见攻击点 1；标题/摘要若出现 "STDP" 字样需立刻跟上限定语。）
- **深度学习背景审稿人**："56 像素图 + 玩具级网络，为什么不直接端对端 BP？"（应对：摘要第二句就把"无 BP 可行边界研究"的定位亮出来，B4 参照系存在的事实要在 intro 末尾点名。）

**v0 暴露的叙事漏洞（诚实记录）**：
1. 摘要里 "competitive Hebbian" 对 VPR 读者偏陌生，需要一个括注式定义（"winner-take-all 竞争下的局部幅度规则"）；
2. 三句话都没提效率数字，ICRA 版摘要在有数字后需要加一句训练/推理开销对比（条件位预留）；
3. "the only convolutional SNN for VPR"（指 LoCS-Net）这个断言在投稿前需再次文献核查，防止新并发工作。

---

## 3. 审稿人最可能攻击的三个点（Top-3）

### 攻击 1：「这不是 STDP——timing 在哪里？」
- **可能性**：高（神经形态审稿人必中）｜**杀伤力**：中（措辞问题，不伤结果）
- **一句话回应**：*本规则是 BLiTNet/VPRTempo STDP 公式在单步幅度极限下的退化形式，正文全程使用 "single-step competitive Hebbian plasticity" 并显式声明该继承关系；与多步 trace-based 规则（SpikingJelly/bindsnet 系）的对比在 related work 与附录参照行给出。*
- **叙事防线**：命名诚实化（已在 PLAN S3.5 写作决策中锁定）；绝不单独使用裸 "STDP" 字样。

### 攻击 2：「学到的不就是 Gabor 吗？/ 随机卷积可能一样好」
- **可能性**：高（任何审稿人都可能问）｜**杀伤力**：高（直击 Gate 1/1.5）
- **一句话回应**：*主表内含同初始化的随机核前端（B1）与同结构手工 Gabor 组（B5）两个对照行；学习 vs 随机 vs 手工的三方对比正是本文核心证据，核形态学分析（Gabor 拟合 R²、方向选择性分布）进一步刻画了学到的结构与手工设计的差异。*
- **叙事防线**：B1/B5 不是被动防御，而是被叙事主动"邀请"进主表——v0 起就把这两行写成故事的转折点（"读者此刻一定在想……")。

### 攻击 3：「LoCS-Net 已经做了卷积 SNN 的 VPR，且性能更高 / 你只赢了 VPRTempo 一点点」
- **可能性**：中高（VPR 审稿人）｜**杀伤力**：高（定位问题，处理不好会被判 increment性不足）
- **一句话回应**：*LoCS-Net 与本文处于正交位置：它以端到端 BP + 多步仿真换性能上界，本文刻画无 BP 局部可塑性的可行边界与机制——双轨协议将"编码器质量"与"读出适配"解耦，这一分析方法本身即贡献；效率表（Table 4）给出两条路线的训练/推理开销对比。*
- **叙事防线**：intro 第三段必须主动引用并划界 LoCS-Net（不能等审稿人发现）；claim 措辞永远是 "BP-free boundary"，永不暗示性能超越 BP 路线。

---

## 4. 修辞 → 机制映射表（纪律检查）

| 叙事中的"噱头"措辞 | 指向的真实机制差异 | 证据位置（未来） |
|---|---|---|
| "无 BP" | 局部 winner 相关更新（conv2d_weight 互相关）vs LoCS-Net 端到端反传 | 方法节 + Table 1 B4 行 |
| "无多步仿真" | 单步幅度域规则 vs SpikingJelly T 步 trace | 附录参照行 + Table 4 训练时间列 |
| "竞争即池化" | local WTA 块 winner 图 = 4×4 max-pool（零额外计算） | 方法节 + ADR-1 维度账 |
| "encoder 与读出解耦" | 双轨协议（轨B 绕开 spike-forcing） | 图 1（协议示意图）+ Table 1 双栏 |
| "核学到了结构" | Gabor 拟合 R²↑、方向选择性、稀疏度、DC/AC 比↓ | Figure 2 + 附录统计表 |
| "与 VPRTempo 框架兼容" | layer_dict 逐层训练机制原样复用，B0 路径零改动 | 方法节一段 + 代码开源 |

---

## 5. 体裁适配备忘（两手准备）

| 元素 | ICRA 6 页版 | RA-L/期刊版 |
|---|---|---|
| 三句话故事线 | 全部保留，摘要逐字打磨 | 保留，允许展开动机段 |
| 双轨协议 | 图 1 + 方法节 1 栏 | 可独立小节 + 协议形式化 |
| B1/B5 对照 | 主表两行，文字点到 | 加核形态统计附录 |
| PatchNorm 交互（Table 3/3b） | 只报结论一句 + 指向附录 | 完整分析小节（最独特的分析） |
| 效率表 | 必须有（VPRTempo 读者期待） | 必须有 + 能效讨论 |
| ORC 复跑 | 附录一句话 | 正文小节 |

---

## 6. 下一步所需输入

1. **S1.5 的 B0 基线数字**（到位后即可起草 related work 与实验设置节的非结果段落）；
2. 你对 §2 预演中"叙事漏洞 1"（competitive Hebbian 括注定义）的措辞偏好；
3. 是否启动 `submission/` 的投稿决策备忘录（ICRA 2027 vs RA-L 时间线细算）。
