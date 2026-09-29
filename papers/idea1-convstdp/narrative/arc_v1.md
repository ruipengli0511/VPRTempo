# 叙事弧 v1 —— 卖点审计后的重构版

> 版本：v1（2026-09-02）｜ 状态：**零数据，全部条件语气**
> 演变：candidates_v0（三弧并立）→ 本版。触发：卖点审计——能耗与性能两条路线被证伪关闭，核心卖点重定位。
> 纪律：禁止声称未做的实验；技术细节以 PLAN.md 为准；对外命名 "single-step competitive Hebbian plasticity"；
> **全文禁用 energy/能耗表述**（无神经形态硬件部署，计算成本 ≠ 能耗）；"first" 不进摘要第一句。

---

## 1. 卖点审计（哪些路在设计上就已关闭）

### 1.1 能耗——直接排除
我们在 GPU/CPU 上做 SNN 仿真。领域内已有工作指出 SNN 在 GPU 仿真下能效并不优于 ANN（仿真窗口开销），能效兑现必须上 Loihi/BrainScaleS 类硬件；LoCS-Net 真部署了 Kapoho Bay，我们没有。该领域审稿人对此格外敏感，写能耗就是送拒稿理由。**Table 4 的 FLOPs、参数量、墙钟时间只能称"计算成本（computational cost）"。**

### 1.2 性能对外对标——关闭
LoCS-Net 在 Nordland 上 P@100%R 78.6%（BP + 卷积路线）；B4（CNN+BP）在 S2.8 中被自设为上界参照且声明不超越；500 地单模块配置与已发表数字不可比。"我们更准"在设计上已关闭。
**保留的唯一性能声明**：Gate 2 若通过，"同 500 地配置、同 seed、同协议下对 VPRTempo 家族的内部改进"是诚实的支撑证据——但它不进标题、不进摘要第一句，只在结果节出现。

### 1.3 方法论——只做支撑
双轨协议、容量对齐、决策门提前承诺、预注册 k 选择程序都是真贡献，但"实验设计严谨"不是接收理由，只是不被拒的理由。永不当主卖点。

## 2. 真正的卖点：三层结构（按可信度排序）

### 第一层（核心）：一个没人定量回答过的科学问题
> **当预处理已经用手工设计的局部归一化（PatchNorm）完成了早期视觉，无监督局部可塑性在前端还剩多少可学？天花板在哪里？**

承载物：§0.6 阶梯表（R0→R4 边际增益）+ Table 3（PatchNorm × 前端 2×4）+ Table 3b（直流塌缩的机制证据格）。关键是 Table 3b：它把"直流塌缩"从猜想变成带机制解释的实证——**有机制的现象报告与纯现象报告是两个档次**。本文货币是解释，不是排行榜名次；RA-L 接受这种货币。

### 第二层：成本论证的比值表述
真正可测准的量化资产是两条比值：①卷积 STDP 每样本只更新 |W|·k² 个元素 vs 全连接 STDP tile 整个 [in,out]；②单步 vs 多步的 1/T 仿真开销比。
主张措辞定型：*「以 1/T 的前端代价、且不使用梯度，取得了卷积前端收益的 X%」*——**单位成本的能力比**，不是能耗，普通硬件即可测准。
配套动作（实验需求建议，见 §5）：阶梯表加 ΔFLOPs 列，使每阶读作"这个机制值多少、花多少"，阶梯表由此从消融升级为论文中心表，同时装进第一层和第二层。

### 第三层（被低估的实用卖点）：Gate 1.5 的部署论证
若 B2 ≈ B5：*「一条零设计成本的学习规则达到了手工调参滤波器组的水平，且能随部署环境自适应」*。手工 PatchNorm 窗口与 Gabor 参数要按环境调，学习规则不用。**此论证在 Gate 2 不过的退路剧本中依然成立**——是最抗跌的一格。

## 3. 修正后的三句话故事线（v1）

### 中文版
1. **问题**：VPR 用脉冲网络依赖手工设计的预处理（局部块归一化）与全连接读出层——当早期视觉已由手工完成时，无监督局部可塑性在前端还剩多少可学、天花板在哪里，尚无定量答案。
2. **主张**：以 VPRTempo 为测试床，我们在脉冲编码与特征层之间插入由 single-step competitive Hebbian plasticity 学习的卷积前端，并用双轨评测（经读出层 / 绕开读出层）与同初始化的随机核、手工 Gabor 组对照，把"结构收益、预处理重叠、读出适配"三个因素定量分离。
3. **证据（条件语气）**：若阶梯表显示各机制边际增益的分布、Table 3b 证实直流塌缩的机制解释，则本文以 1/T 的前端代价、无梯度的规则，给出无 BP 局部可塑性在 VPR 中可行边界的定量刻画——包括增益成立时的来源定位，与（若 Gate 2 不过）失配发生的位置。

### English draft
1. *Spiking networks for visual place recognition rely on hand-crafted preprocessing (local patch normalization) followed by fully-connected readout — once early vision is done by hand, how much is left for unsupervised local plasticity to learn, and where is its ceiling? This question has no quantitative answer.*
2. *Using VPRTempo as a testbed, we insert a convolutional frontend learned by single-step competitive Hebbian plasticity between spike encoding and the feature layer, and quantitatively separate three factors — structural gain, preprocessing overlap, and readout fit — via a dual-track protocol (with/without the supervised readout) against matched random and hand-crafted Gabor frontends.*
3. *Should the contribution ladder and the DC-collapse evidence hold as predicted, the paper delivers a quantitative map of where BP-free local plasticity helps VPR — at 1/T frontend cost and without gradients — locating both the source of any gain and, should end-to-end gains not materialize, the locus of the mismatch.*

## 4. 与 candidates_v0 的对应关系（追溯）

| v0 候选弧 | v1 中的去向 |
|---|---|
| 弧 A（双轨方法论） | 降为**工具层**——双轨是回答核心问题的仪器，不再是卖点本身（修正了 v0 的错位：把仪器当成了发现） |
| 弧 B（机制涌现） | 拆解吸收——核形态证据服务于第一层问题的"还剩多少可学"；Gabor 对比服务于第三层部署论证 |
| 弧 C（效率） | 重写为第二层比值表述；**"能耗"措辞全灭**；Arc C 标题级主张废弃 |

**v1 相对于 v0 的本质变化**：论文中心从"我们提出了一个前端"移到"我们定量回答了一个问题"。前端是仪器，阶梯表 + Table 3/3b 是发现。

## 5. 实验需求建议（由你决定是否回灌 PLAN.md，我无权改设计）

1. **阶梯表加 ΔFLOPs 列**（R0→R4 每阶报 ΔRecall + ΔFLOPs）：需 S1.1 的 run_exp.py 在每个 run 落盘 FLOPs 计数，S1 阶段不做事后补不齐 → 建议回灌 S1.1/S3.4。
2. **写作纪律条款**：Table 4 命名"计算成本"，全文禁"能耗"→ 建议回灌 PLAN S3.4 与 S3.5 写作决策。
3. Gate 1.5 的部署论证需要 B5 行的双轨数字足够稳（3 seeds 全跑，不裁剪）→ 已在 PLAN S2.9 验收内，无需改动。

## 6. 审稿人第一反应预演（v1 版摘要句）

- **神经形态审稿人**："没有能耗声称，没有乱叫 STDP——至少不冒犯。"（攻击面收窄到规则细节，可应对）
- **VPR 审稿人**："不拼排行榜？那他凭什么……'还剩多少可学'——这个问题倒是真没人算过。"（第一层卖点生效）
- **怀疑型审稿人**："阶梯表的 ΔFLOPs 列……他们把消融做成了成本-收益分析，这个我读得下去。"（第二层生效）

**v1 暴露的叙事漏洞（诚实记录）**：
1. 第一层问题的成立依赖"PatchNorm = 手工早期视觉"这一等价表述被接受——intro 需要一段论证局部 Z-score 归一化与经典前馈滤波的功能等价性，否则问题框架被质疑；
2. "X%"（收益份额）的措辞要等 Gate 1/2 数字出来后才能填，摘要该位置目前是占位符；
3. 若阶梯表所有边际增益都 ≈ 0（全机制无效），第一层答案变成"什么都不剩"——这是负面结果叙事，v1 框架下仍可写（天花板答案），但投稿目标需重新评估。

---

**待你决策**：①确认 v1 的三层卖点结构（或指出异议）；②§5 的实验需求建议是否回灌 PLAN.md；③确认后我展开章节级论证骨架（outline_v1.md）。
