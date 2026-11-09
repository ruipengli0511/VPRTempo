# MIT License
#
# Copyright (c) 2023 Adam Hines, Peter Stratton, Michael Milford, Tobias Fischer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
================================================================================
Part 1: 模块说明 (Imports & Overview)
================================================================================
本文件实现 IDEA1 S3.6 的 HOG 手工描述子直通基线（外部参照，不进 SNN）。

设计要点（S3.6 卡片）：
- 使用环境已有的 cv2.HOGDescriptor（opencv>=4.12，无新增依赖）。
- 28×28 灰度输入：win=28、block=14、stride=7、cell=7、nbins=9。
  实际维度 = ((28-14)/7+1)^2 块 × (14/7)^2 cell × 9 bins = 9×4×9 = 324 维
  （S3.6 卡片所写 972 维为 3×324 的笔误，以实测维度为准）。
- 输入约定：与 SNN 输入前三步一致（灰度 → gamma → resize 28×28，
  ProcessImage patch_norm='off' 的 [0,1] 输出，即 uint8/255）。
- 输出：逐样本 L2 归一化描述子（在 cv2 内部块归一化之上再做整描述子
  L2 归一化，与轨 B 的 cosine 检索协议对齐）。
================================================================================
"""
import cv2
import numpy as np
import torch


# ================================================================================
# 函数：hog_descriptor —— 批量计算 HOG 描述子
# ================================================================================
def hog_descriptor(images,
                   win_size=(28, 28),
                   block_size=(14, 14),
                   block_stride=(7, 7),
                   cell_size=(7, 7),
                   nbins=9):
    """
    批量计算 HOG 描述子（cv2.HOGDescriptor 封装）。

    参数:
        images      : torch.Tensor 或 np.ndarray，形状 [N, 1, H, W] 或 [N, H*W]，
                      0~1 浮点灰度图（H=W=win_size 的边长）。
        win_size    : HOG 窗口尺寸（默认 28×28，与 SNN 输入同分辨率）。
        block_size  : 块尺寸（默认 14×14）。
        block_stride: 块步长（默认 7×7）。
        cell_size   : cell 尺寸（默认 7×7）。
        nbins       : 梯度方向 bin 数（默认 9）。

    返回:
        np.ndarray [N, D]：逐样本 L2 归一化的 HOG 描述子。
                      D = nblocks × ncells_per_block × nbins
                        = ((W-bw)/bs+1)×((H-bh)/bs+1) × (bw/cw)×(bh/ch) × nbins
                      默认参数下 D = 9 × 4 × 9 = 324。
    """
    # ---- 行级：统一转 numpy ----
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    # ---- 行级：展平输入 [N, H*W] → [N, 1, H, W]；[N,1,H,W] 原样通过 ----
    images = np.asarray(images)
    if images.ndim == 2:
        n = images.shape[0]
        side = int(round(images.shape[1] ** 0.5))
        assert side * side == images.shape[1], "展平输入的维度必须是完全平方数（H*W）"
        images = images.reshape(n, 1, side, side)
    assert images.ndim == 4 and images.shape[1] == 1, \
        f"期望 [N,1,H,W] 输入，得到 {images.shape}"

    # ---- 行级：实例化 cv2 HOG 描述子（默认参数对齐 S3.6 卡片）----
    hog = cv2.HOGDescriptor(_winSize=win_size, _blockSize=block_size,
                            _blockStride=block_stride, _cellSize=cell_size,
                            _nbins=nbins)

    # ---- 行级：cv2 要求图像尺寸与 winSize 一致，且输入为 uint8 ----
    h, w = images.shape[2], images.shape[3]
    assert (h, w) == tuple(win_size), f"图像尺寸 {(h, w)} 与 winSize {win_size} 不一致"

    # ---- 行级：逐样本计算：0~1 浮点 → ×255 转 uint8 → compute → reshape ----
    feats = np.zeros((images.shape[0], hog.getDescriptorSize()), dtype=np.float32)
    for i in range(images.shape[0]):
        img_u8 = (images[i, 0] * 255.0).clip(0, 255).astype(np.uint8)
        d = hog.compute(img_u8)          # 返回值形状 [D, 1]
        feats[i] = d.ravel()

    # ---- 行级：整描述子 L2 归一化（与轨 B cosine 检索协议对齐）----
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, 1e-12)

    return feats
