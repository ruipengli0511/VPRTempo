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

"""Deterministic fixed Gabor filter bank used by the B5 frontend.

The bank has 4 orientations × 2 wavelengths × 2 phases × 2 sigmas = 32
channels. Each kernel is mean-centred and L1-normalized. Loading freezes the
weights while retaining threshold adaptation; no weight clamp or
renormalization is applied to the signed kernels.
"""
import math

import torch


def gabor_kernel_bank(kernel_size=5, device=None):
    """Return the [32, 1, k, k] mean-centred, unit-L1 Gabor bank."""
    k = kernel_size
    half = k // 2
    ys, xs = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij")

    orientations = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    wavelengths = [2.0, 3.0]
    phases = [0.0, math.pi / 2]
    sigmas = [1.0, 2.0]
    gamma = 0.5

    kernels = []
    for theta in orientations:
        ct, st = math.cos(theta), math.sin(theta)
        xr = xs * ct + ys * st
        yr = -xs * st + ys * ct
        for lam in wavelengths:
            for psi in phases:
                carrier = torch.cos(2 * math.pi * xr / lam + psi)
                for sigma in sigmas:
                    envelope = torch.exp(-(xr ** 2 + gamma ** 2 * yr ** 2)
                                         / (2 * sigma ** 2))
                    g = envelope * carrier
                    g = g - g.mean()
                    nrm = g.abs().sum()
                    if nrm > 0:
                        g = g / nrm
                    kernels.append(g)

    bank = torch.stack(kernels).unsqueeze(1)
    return bank.to(device)


def load_gabor_weights(layer):
    """Load the fixed bank, freeze weights, and retain frontend ITP."""
    bank = gabor_kernel_bank(kernel_size=layer.kernel_size,
                             device=layer.w.weight.device)
    assert bank.shape == layer.w.weight.shape, (
        f"Gabor bank shape {tuple(bank.shape)} does not match layer weights "
        f"{tuple(layer.w.weight.shape)}; B5 requires 32 channels")
    with torch.no_grad():
        layer.w.weight.copy_(bank)
    layer.frozen = True
    layer.itp_on_frozen = True
    return layer
