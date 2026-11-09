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

"""Threshold adaptation for the frozen Gabor frontend.

The Gabor weights never enter a weight-update, clamp, or renormalization path.
Only the feature-layer STDP/ITP controls affect downstream feature learning.
"""
import torch


def apply_itp_conv(out, layer):
    """Update thresholds from pre-WTA firing rates without changing weights."""
    with torch.no_grad():
        observed = (out.pre_wta > 0).float().mean(dim=(2, 3), keepdim=True)
        layer.thr.data += layer.eta_ip * (observed - layer.fire_rate)
        layer.thr.data.clamp_(min=layer.thr_min)


def train_conv_layer(train_loader, layer, model, model_num=0):
    """Run the B5 frontend's ITP schedule with its independent epoch budget."""
    from tqdm import tqdm

    if not layer.frozen or not getattr(layer, "itp_on_frozen", False):
        raise ValueError("The retained convolutional training path is fixed Gabor + ITP only")

    n_imgs = len(train_loader.dataset)
    conv_epoch = int(getattr(model, "conv_epoch", 2))
    T = int(n_imgs * conv_epoch)
    pbar = tqdm(total=T, desc=f"Module {model_num + 1} conv_layer", position=0)

    init_itp = layer.eta_ip.detach()
    mod = 0
    for _ in range(conv_epoch):
        for spikes, _ in train_loader:
            spikes = spikes.to(model.device)
            out = layer(layer.reshape_input(spikes))
            apply_itp_conv(out, layer)

            if mod % 100 == 0:
                pt = pow(float(T - mod) / T, 2)
                layer.eta_ip = torch.mul(init_itp, pt)
            mod += 1
            pbar.update(1)
    pbar.close()
