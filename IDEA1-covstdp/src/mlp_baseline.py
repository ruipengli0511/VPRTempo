# ================================================================================
# mlp_baseline.py — MLP-BP 对照模型（MLP 对照实验：BP vs STDP 学习机制比较）
#
# 目的：在与 VPRTempo (SNN) 完全对齐的输入/维度/稀疏拓扑/数据呈现量下，
#       用端到端 BP + CrossEntropy 训练同构 MLP，隔离"学习规则"这一唯一变量。
#
# 实验臂：
#   M0      稠密 MLP-BP（结构不控制，粗对比）
#   M0-long M0 延长训练 + winter 早停（BP 充分训练天花板）
#   M1      稀疏掩码 MLP-BP（掩码复刻 blitnet.addWeights 拓扑，核心对比）
#   M2      冻结随机特征层 + 仅训读出层（隔离 STDP 特征学习贡献）
#
# 对应 SNN 侧锚点：
#   - 网络维度  input→2·input→n_places   vprtempo/VPRTempoTrain.py:178-183
#   - 稀疏拓扑  p_exc=0.1, p_inh=0.5      vprtempo/src/blitnet.py:216-234
#   - 初始化    正态→符号裁剪→按p置零→L1列归一化  vprtempo/src/blitnet.py:250-327
# ================================================================================
import numpy as np
import torch
import torch.nn as nn


# ================================================================================
# build_feature_mask — 复刻 blitnet.addWeights 的稀疏连接拓扑
# ================================================================================
# SNN 特征层的连接生成过程（blitnet.py:250-327）：
#   兴奋矩阵：N(0.5, 1/6) 采样 → 负值置零 → 按 p_exc=0.1 保留
#   抑制矩阵：N(-0.5, 1/6) 采样 → 正值置零 → 按 p_inh=0.5 保留
#   合并 W = W_exc + W_inh，非零位置即连接
# 期望密度 ≈ 1 - (1-p_exc)(1-p_inh) ≈ 0.55（符号裁剪几乎不改变密度，
# 因为 N(±0.5, 1/6) 的异号概率 < 0.2%）
# ================================================================================
def build_feature_mask(in_dim, out_dim, p_exc=0.1, p_inh=0.5, seed=0):
    """生成与 SNN 特征层同分布的 {0,1} 连接掩码。

    Args:
        in_dim: 输入维度（像素数，如 784）
        out_dim: 特征层维度（如 1568）
        p_exc: 兴奋连接概率（SNN 表 I：0.1）
        p_inh: 抑制连接概率（SNN 表 I：0.5）
        seed: numpy 随机种子（与 SNN 一样走 np.random）

    Returns:
        torch.FloatTensor [out_dim, in_dim]，1=有连接，0=无连接
    """
    rng = np.random.RandomState(seed)
    # 兴奋侧：正态采样→负值裁剪→按 p_exc 稀疏化（与 addWeights 同流程）
    exc = rng.normal(loc=0.5, scale=1.0 / 6.0, size=(out_dim, in_dim))
    exc[exc < 0] = 0.0
    exc[rng.rand(out_dim, in_dim) > p_exc] = 0.0
    # 抑制侧：正态采样→正值裁剪→按 p_inh 稀疏化
    inh = rng.normal(loc=-0.5, scale=1.0 / 6.0, size=(out_dim, in_dim))
    inh[inh > 0] = 0.0
    inh[rng.rand(out_dim, in_dim) > p_inh] = 0.0
    mask = ((exc != 0) | (inh != 0)).astype(np.float32)
    return torch.from_numpy(mask)


# ================================================================================
# blitnet_init_ — 按 addWeights 方式初始化权重（正态→符号裁剪→按p置零→L1列归一化）
# ================================================================================
# 【使用注意】该初始化是为 STDP 动力学设计的：抑制连接占主导（p_inh=0.5 vs
# p_exc=0.1）且 L1 归一化后权重极小（~1e-3），净输入几乎恒负。SNN 靠 ITP
# 阈值适应与 STDP 增 Exc 权重存活；而 BP+ReLU 下前层输出恒 0（死亡网络，
# 已在 selfcheck 中验证）。因此 M1/M2 实际使用 masked_default 初始化
# （Kaiming × 掩码）——结构对齐的公平性由掩码保证，初始化分布不应让 BP
# 侧带病起跑。本函数保留用于参考/消融。
# ================================================================================
def blitnet_init_(weight, p_exc=0.1, p_inh=0.5, seed=0):
    """以 blitnet.addWeights 的同分布方式就地初始化 weight [out_dim, in_dim]。

    用于 M1/M2 的特征层：使初始权重分布与 SNN 一致（含符号结构与列归一化），
    保证"唯一变量是学习规则"。M0 使用 PyTorch 默认初始化（BP 社区标准做法）。
    """
    rng = np.random.RandomState(seed)
    out_dim, in_dim = weight.shape
    exc = rng.normal(loc=0.5, scale=1.0 / 6.0, size=(out_dim, in_dim))
    exc[exc < 0] = 0.0
    exc[rng.rand(out_dim, in_dim) > p_exc] = 0.0
    inh = rng.normal(loc=-0.5, scale=1.0 / 6.0, size=(out_dim, in_dim))
    inh[inh > 0] = 0.0
    inh[rng.rand(out_dim, in_dim) > p_inh] = 0.0
    W = exc + inh
    # L1 列归一化（blitnet.py:319-325：对每个输入神经元的出边求 L1 范数）
    nrm = np.linalg.norm(W, ord=1, axis=0)
    nrm[nrm == 0.0] = 1.0
    W = W / nrm
    with torch.no_grad():
        weight.copy_(torch.from_numpy(W.astype(np.float32)))
    return weight


# ================================================================================
# MLPBaseline — 同构 MLP：Linear(in, 2·in) → LeakyReLU → Linear(2·in, n_places)
# ================================================================================
# 【激活函数设计决定（2026-10-16 诊断确认）】
# 输出类数较大（≥500）时，标准 ReLU 在 BP 下发生大规模神经元死亡：
# 1100 类时 ep1 死神经元比例即达 99.2%，loss 恒为 ln(1100)，训练完全停滞
# （500 类时处于死亡边缘——部分 seed 碰巧存活、部分死亡，方差是假象）。
# 因此所有臂统一使用 LeakyReLU(0.01)——给 BP 侧标准、稳定的训练条件，
# 使对比聚焦于学习规则而非激活函数的病理。对照实验（diag）证实 LeakyReLU
# 后 1100 类 15 epoch train_acc 即达 0.80。
# ================================================================================
class MLPBaseline(nn.Module):
    """与 VPRTempo 单模块同维度的两层 MLP。

    Args:
        in_dim: 输入维度（H*W）
        n_places: 输出地点数（模块内）
        feat_mult: 特征层倍数（SNN 默认 2）
        feature_mask: 可选 [feat_dim, in_dim] {0,1} 掩码（M1/M2），
                      前向时 weight*mask，保持稀疏拓扑
        freeze_feature: True 时特征层不参与训练（M2：随机特征+BP读出）
        init: 'default'（PyTorch Kaiming，M0 用）、'blitnet'（与 SNN 同分布，
              仅供消融参考——对 BP+ReLU 是死亡初始化）、'masked_default'
              （Kaiming × 掩码，M1/M2 用：拓扑与 SNN 对齐，幅度对 BP 友好）
        seed: 初始化随机种子
    """

    def __init__(self, in_dim, n_places, feat_mult=2, feature_mask=None,
                 freeze_feature=False, init="default", seed=0):
        super().__init__()
        feat_dim = int(in_dim * feat_mult)
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.n_places = n_places

        self.feature = nn.Linear(in_dim, feat_dim, bias=False)
        self.readout = nn.Linear(feat_dim, n_places, bias=False)

        if feature_mask is not None:
            self.register_buffer("feature_mask", feature_mask)
        else:
            self.feature_mask = None

        if init == "blitnet":
            blitnet_init_(self.feature.weight, seed=seed)
        elif init == "masked_default":
            assert self.feature_mask is not None, "masked_default 需要 feature_mask"
            with torch.no_grad():
                # Kaiming（按有效 fan_in 缩放）× 掩码，保留稀疏拓扑
                nn.init.kaiming_uniform_(self.feature.weight, a=5 ** 0.5)
                self.feature.weight *= self.feature_mask
        elif init != "default":
            raise ValueError(f"未知 init: {init}")

        if freeze_feature:
            for p in self.feature.parameters():
                p.requires_grad_(False)

        self.apply_mask()

    def apply_mask(self):
        """前向前确保特征层权重满足稀疏拓扑（训练每 step 后也要调用）"""
        if self.feature_mask is not None:
            with torch.no_grad():
                self.feature.weight *= self.feature_mask

    def forward(self, x):
        self.apply_mask()
        h = torch.nn.functional.leaky_relu(self.feature(x), negative_slope=0.01)
        return self.readout(h)

    # ---- 成本计量辅助 ----
    def count_params(self):
        """返回 (总参数量, 非零参数量)"""
        total = sum(p.numel() for p in self.parameters())
        nonzero = sum(int((p != 0).sum().item()) for p in self.parameters())
        return total, nonzero

    def forward_flops(self):
        """单次前向 FLOPs（乘+加各计 1）：两个矩阵乘，2*fan_in*fan_out"""
        return 2 * self.in_dim * self.feat_dim + 2 * self.feat_dim * self.n_places


# ================================================================================
# train_mlp — BP + CrossEntropy 训练循环
# ================================================================================
def train_mlp(model, loader, epochs, lr, weight_decay, device, log_fn=print,
              eval_callback=None):
    """标准 BP 训练。返回 (train_acc_history, n_weight_updates, best_epoch)。

    eval_callback: 可选，每个 epoch 末调用 score = eval_callback(model)，
    追踪最优分数并在训练结束后恢复最优权重（早停语义，用于 M0-long 的
    winter 验证集选 epoch）。None 时不追踪，best_epoch = epochs。
    """
    import copy
    model.to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    acc_history = []
    n_updates = 0
    best_score, best_state, best_epoch = -1.0, None, epochs
    for ep in range(epochs):
        correct, total = 0, 0
        for x, y in loader:
            x = x.to(device).float().view(x.size(0), -1)
            # ProcessImage 输出 uint8 幅度（0-255），与 SNN 脉冲输入同分布需归一到 [0,1]
            if x.max() > 1.5:
                x = x / 255.0
            y = y.to(device).long()
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.apply_mask()  # 梯度更新会破坏稀疏拓扑，每 step 重新掩码
            n_updates += 1
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
        acc = correct / max(total, 1)
        acc_history.append(acc)
        log_fn(f"[train_mlp] epoch {ep+1}/{epochs} loss={loss.item():.4f} train_acc={acc:.4f}")
        if eval_callback is not None:
            score = eval_callback(model)
            model.train()
            log_fn(f"[train_mlp]   val_score={score:.4f} (best={max(best_score, score):.4f})")
            if score > best_score:
                best_score = score
                best_epoch = ep + 1
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
        log_fn(f"[train_mlp] 早停：恢复 epoch {best_epoch} 的最优权重 (score={best_score:.4f})")
    return acc_history, n_updates, best_epoch
