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

"""Bridge between VPRTempo and the fixed Gabor convolutional frontend.

The implementation lives in ``IDEA1-covstdp`` and is loaded lazily so the
original ``frontend='none'`` path does not import or initialize it.
"""
import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOADED = {}


def _load(mod_name, rel_path):
    if mod_name not in _LOADED:
        spec = importlib.util.spec_from_file_location(mod_name, _REPO_ROOT / rel_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LOADED[mod_name] = mod
    return _LOADED[mod_name]


def _layer_mod():
    return _load("idea1_conv_snn_layer", "IDEA1-covstdp/src/conv_snn_layer.py")


def _learn_mod():
    return _load("idea1_conv_learning", "IDEA1-covstdp/src/conv_learning.py")


def _gabor_mod():
    return _load("idea1_gabor_frontend", "IDEA1-covstdp/src/gabor_frontend.py")


def __getattr__(name):
    if name in ("ConvSNNLayer", "ConvFrontendModule"):
        return getattr(_layer_mod(), name)
    if name in ("train_conv_layer", "apply_itp_conv"):
        return getattr(_learn_mod(), name)
    raise AttributeError(name)


def is_conv_layer(obj):
    """Return whether *obj* is a frontend layer without forcing a module load."""
    if obj.__class__.__name__ != "ConvSNNLayer":
        return False
    return isinstance(obj, _layer_mod().ConvSNNLayer)


def _parse_pair(value):
    if isinstance(value, str):
        return [float(x) for x in value.split(",")]
    return list(value)


def build_conv_layer(model, dims, device, inference):
    """Construct the fixed Gabor frontend while preserving model initialization order."""
    frontend = getattr(model, "frontend", "none")
    if frontend != "gabor":
        raise ValueError(f"Unsupported convolutional frontend: {frontend!r}; use 'none' or 'gabor'")

    layer = _layer_mod().ConvSNNLayer(
        input_dims=dims,
        in_channels=1,
        out_channels=int(getattr(model, "conv_channels", 32)),
        kernel_size=int(getattr(model, "conv_kernel", 5)),
        thr_range=_parse_pair(getattr(model, "conv_thr_range", "0,0.5")),
        thr_min=float(getattr(model, "conv_thr_min", 0.0)),
        fire_rate=_parse_pair(getattr(model, "conv_fire_rate", "0.2,0.9")),
        ip_rate=float(getattr(model, "conv_ip_rate", 0.15)),
        wta_mode=getattr(model, "wta_mode", "local"),
        wta_block=int(getattr(model, "wta_block", 4)),
        device=device,
        inference=inference,
        frozen=True,
    )
    _gabor_mod().load_gabor_weights(layer)
    return layer


def conv_forward(layer, spikes_flat):
    """Run the frontend and return its pooled descriptor input."""
    import torch
    spikes_flat = spikes_flat.to(layer.w.weight.device)
    with torch.no_grad():
        out = layer(layer.reshape_input(spikes_flat))
    return out.pooled_flat
