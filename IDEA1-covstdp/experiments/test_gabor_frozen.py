#!/usr/bin/env python
"""Lightweight regression for the fixed B5 Gabor frontend."""
import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
assert Path.cwd() == REPO_ROOT, "Run from the repository root"
sys.path.insert(0, str(REPO_ROOT))


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv_mod = load_mod("conv_snn_layer", "IDEA1-covstdp/src/conv_snn_layer.py")
learn_mod = load_mod("conv_learning", "IDEA1-covstdp/src/conv_learning.py")
gabor_mod = load_mod("gabor_frontend", "IDEA1-covstdp/src/gabor_frontend.py")

torch.manual_seed(0)
bank = gabor_mod.gabor_kernel_bank(kernel_size=5)
assert bank.shape == (32, 1, 5, 5)
assert torch.allclose(bank.abs().flatten(1).sum(dim=1), torch.ones(32), atol=1e-5)
assert bank.flatten(1).mean(dim=1).abs().max() < 1e-6
assert (bank < 0).any()

layer = conv_mod.ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                              wta_mode='local', wta_block=4, device='cpu', frozen=True)
gabor_mod.load_gabor_weights(layer)
assert layer.frozen is True and layer.itp_on_frozen is True
assert torch.equal(layer.w.weight.data, bank)

x = torch.rand(1, 1, 28, 28)
out = layer(x)
assert out.pooled_flat.shape == (1, 1152)
assert (out.pre_wta >= 0).all() and (out.pre_wta <= 0.9).all()

w_before = layer.w.weight.data.clone()
thr_before = layer.thr.data.clone()
learn_mod.apply_itp_conv(out, layer)
assert torch.equal(layer.w.weight.data, w_before)
assert not torch.equal(layer.thr.data, thr_before)
assert (layer.w.weight.data < 0).any()

print("B5 Gabor regression passed: bank, frozen weights, ITP, and 1152-D forward")
