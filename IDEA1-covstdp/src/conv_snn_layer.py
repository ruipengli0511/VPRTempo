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

"""Convolutional spiking frontend used by the fixed Gabor representation.

Forward: conv2d -> ON/OFF sign handling -> threshold -> clamp -> WTA -> pooling.
The initialization order is retained because its random-number consumption
also determines downstream feature/output initialization under a fixed seed.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ConvLayerOutput:
    pre_wta: torch.Tensor
    post_wta: torch.Tensor
    winner_mask: torch.Tensor
    pooled_flat: torch.Tensor
    winner_values: torch.Tensor


class ConvSNNLayer(nn.Module):
    def __init__(self,
                 input_dims=[28, 28],
                 in_channels=1,
                 out_channels=32,
                 kernel_size=5,
                 padding=0,
                 thr_range=[0, 0.5],
                 fire_rate=[0.2, 0.9],
                 ip_rate=0.15,
                 p_exc=0.5,
                 thr_min=0.0,
                 wta_mode='local',
                 wta_block=4,
                 device=None,
                 inference=False,
                 frozen=False):
        super(ConvSNNLayer, self).__init__()
        self.device = device
        self.inference = inference
        self.frozen = frozen
        self.wta_mode = wta_mode
        self.wta_block = wta_block
        self.thr_min = float(thr_min)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.input_dims = list(input_dims)

        H, W = self.input_dims
        self.H_out = (H + 2 * padding - kernel_size) // 1 + 1
        self.W_out = (W + 2 * padding - kernel_size) // 1 + 1
        b = wta_block
        self.H_pool = self.H_out // b
        self.W_pool = self.W_out // b
        if wta_mode in ('local', 'none'):
            self.flat_dim = out_channels * self.H_pool * self.W_pool
        elif wta_mode == 'global':
            self.flat_dim = out_channels
        else:
            raise ValueError(f"Unknown wta_mode: {wta_mode}")

        self.w = nn.Conv2d(in_channels, out_channels, kernel_size,
                           padding=padding, bias=False)
        self.w.to(device)
        self.thr = nn.Parameter(torch.zeros([1, out_channels, 1, 1],
                                            device=device).uniform_(thr_range[0],
                                                                    thr_range[1]))
        n_exc = int(round(out_channels * p_exc))
        havconnExc = torch.zeros(out_channels, dtype=torch.bool, device=device)
        havconnExc[torch.randperm(out_channels, device=device)[:n_exc]] = True
        self.register_buffer('havconnExc', havconnExc)
        if inference:
            return

        if np.isscalar(thr_range):
            thr_range = [thr_range, thr_range]
        if np.isscalar(fire_rate):
            fire_rate = [fire_rate, fire_rate]

        self.eta_ip = torch.tensor(ip_rate, device=device)
        self.fire_rate = torch.zeros([1, out_channels, 1, 1], device=device)
        fstep = (fire_rate[1] - fire_rate[0]) / out_channels
        for i in range(out_channels):
            self.fire_rate[:, i] = fire_rate[0] + fstep * (i + 1)

        self.w.weight = self._add_conv_weights()

    def _add_conv_weights(self):
        C = self.out_channels
        k = self.kernel_size
        W = torch.empty(C, self.in_channels, k, k, device=self.device)

        exc = self.havconnExc
        W[exc] = torch.empty(exc.sum(), self.in_channels, k, k,
                             device=self.device).normal_(mean=0.5, std=1.0 / 6.0)
        W[exc] = W[exc].clamp(min=0.0)
        inh = ~exc
        if inh.any():
            W[inh] = torch.empty(inh.sum(), self.in_channels, k, k,
                                 device=self.device).normal_(mean=-0.5, std=1.0 / 6.0)
            W[inh] = W[inh].clamp(max=0.0)

        nrm = torch.linalg.norm(W.flatten(1), ord=1, dim=1).view(C, 1, 1, 1)
        nrm[nrm == 0.0] = 1.0
        return nn.Parameter(W / nrm)

    def reshape_input(self, x_flat):
        H, W = self.input_dims
        return x_flat.view(1, self.in_channels, H, W)

    def forward(self, x):
        z = self.w(x)
        exc = self.havconnExc.view(1, -1, 1, 1)
        z = torch.where(exc, z, -z)
        pre_wta = torch.clamp(z - self.thr, min=0.0, max=0.9)

        post_wta, winner_mask = self._apply_wta(pre_wta)
        pooled_flat = self._pool_flat(pre_wta, post_wta)
        winner_values = pre_wta[winner_mask] if winner_mask.any() else \
            torch.zeros(0, device=x.device)

        return ConvLayerOutput(pre_wta=pre_wta,
                               post_wta=post_wta,
                               winner_mask=winner_mask,
                               pooled_flat=pooled_flat,
                               winner_values=winner_values)

    def _apply_wta(self, pre_wta):
        mode = self.wta_mode
        if mode == 'none':
            return pre_wta, torch.ones_like(pre_wta, dtype=torch.bool)

        mask = torch.zeros_like(pre_wta, dtype=torch.bool)
        if mode == 'global':
            C, H, W = pre_wta.shape[1], pre_wta.shape[2], pre_wta.shape[3]
            flat_idx = pre_wta.view(1, C, -1).argmax(dim=-1)
            ys, xs = flat_idx // W, flat_idx % W
            mask[0, torch.arange(C, device=pre_wta.device), ys[0], xs[0]] = True
        elif mode == 'local':
            b = self.wta_block
            C = pre_wta.shape[1]
            Hc, Wc = self.H_pool * b, self.W_pool * b
            x = pre_wta[:, :, :Hc, :Wc]
            blocks = x.view(1, C, self.H_pool, b, self.W_pool, b) \
                      .permute(0, 1, 2, 4, 3, 5).reshape(1, C, self.H_pool, self.W_pool, b * b)
            idx = blocks.argmax(dim=-1, keepdim=True)
            blk_mask = torch.zeros_like(blocks).scatter_(-1, idx, True)
            blk_mask = blk_mask.view(1, C, self.H_pool, self.W_pool, b, b) \
                               .permute(0, 1, 2, 4, 3, 5).reshape(1, C, Hc, Wc)
            mask[:, :, :Hc, :Wc] = blk_mask

        return pre_wta * mask, mask

    def _pool_flat(self, pre_wta, post_wta):
        b = self.wta_block
        if self.wta_mode == 'global':
            pooled = post_wta.amax(dim=(2, 3))
        elif self.wta_mode == 'local':
            Hc, Wc = self.H_pool * b, self.W_pool * b
            pooled = torch.nn.functional.max_pool2d(
                post_wta[:, :, :Hc, :Wc], kernel_size=b)
        else:
            Hc, Wc = self.H_pool * b, self.W_pool * b
            pooled = torch.nn.functional.max_pool2d(
                pre_wta[:, :, :Hc, :Wc], kernel_size=b)
        return pooled.reshape(1, -1)


class ConvFrontendModule(nn.Module):
    def __init__(self, conv_layer):
        super().__init__()
        self.conv_layer = conv_layer

    def forward(self, x_flat):
        out = self.conv_layer(self.conv_layer.reshape_input(x_flat))
        return out.pooled_flat
