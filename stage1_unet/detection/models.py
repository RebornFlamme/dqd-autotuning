"""Segmentation model: a CNN U-Net, optionally behind a parameter-free physical preprocessing.

* ``unet``          -- plain CNN U-Net (1.9 M parameters), the reference.
* ``unet_robust``   -- ``PhysInput`` + the same U-Net (the U-Net LOFO recipe).
* ``unet16_robust`` -- same, base 16 (0.5 M parameters), capacity axis.

Inputs are padded (replicate) to the model's stride and the output is cropped back.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- utils
def conv_bn_act(cin: int, cout: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.GELU()
    )


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(conv_bn_act(cin, cout), conv_bn_act(cout, cout))


class Padded(nn.Module):
    """Pads (replicate) to a multiple of ``stride``, crops logits back."""

    def __init__(self, net: nn.Module, stride: int) -> None:
        super().__init__()
        self.net, self.stride = net, stride

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.stride, (-w) % self.stride
        if ph or pw:
            x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode="replicate")
        y = self.net(x)
        return y[..., ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]


# --------------------------------------------------------------------------- U-Net
class UNet(nn.Module):
    def __init__(self, base: int = 32, depth: int = 3) -> None:
        super().__init__()
        ch = [base * 2**i for i in range(depth + 1)]
        self.inc = DoubleConv(1, ch[0])
        self.downs = nn.ModuleList(DoubleConv(ch[i], ch[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in reversed(range(depth))
        )
        self.dec = nn.ModuleList(DoubleConv(2 * ch[i], ch[i]) for i in reversed(range(depth)))
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        y = skips.pop()
        for up, dec in zip(self.ups, self.dec):
            y = dec(torch.cat([up(y), skips.pop()], 1))
        return self.out(y)


# --------------------------------------------------------------------------- factory
ARCHS = {
    "unet": (lambda: UNet(base=32, depth=3), 8),
}


class PhysInput(nn.Module):
    """Physical input conditioning, part of the model (so it is applied at inference too).

    On top of the global median/MAD normalisation done by the caller:
    * row-median subtraction: removes per-line offsets (stripes, drift, charge jumps), the
      same prior as the classical detectors (sticks are sparse, so the median is background);
    * re-scaling by the MAD, so the input stays in units of noise sigma;
    * de-spiking: a pixel above 6 sigma whose 8 neighbours all stay below min(30 % of it,
      3.5 sigma) is an isolated glitch, not a stick (a stick is >= 4 px long, its neighbours
      are bright too): it is replaced by its 3x3 median. Weak sticks (< 6 sigma) are never
      touched. Measured: removes 90 % of +-10 spikes, alters 0.4 % of stick pixels (those
      are single bright pixels, indistinguishable from a glitch).
    No learned parameter; the sign is left to the network (polarity is augmented).
    """

    def __init__(self, spike_z: float = 6.0, spike_ratio: float = 0.3, spike_cap: float = 3.5) -> None:
        super().__init__()
        self.spike_z, self.spike_ratio, self.spike_cap = spike_z, spike_ratio, spike_cap

    def forward(self, x):
        b, _, h, w = x.shape
        x = x - x.median(dim=-1, keepdim=True).values
        mad = x.flatten(1).abs().median(dim=1).values.view(b, 1, 1, 1)
        x = x / (1.4826 * mad + 1e-6)
        p = F.unfold(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3).view(b, 9, h, w)
        neigh = torch.cat([p[:, :4], p[:, 5:]], 1).abs().amax(1, keepdim=True)
        med = p.median(dim=1, keepdim=True).values
        spike = (x.abs() > self.spike_z) & (neigh < (self.spike_ratio * x.abs()).clamp_max(self.spike_cap))
        return torch.where(spike, med, x)


def build_model(name: str) -> nn.Module:
    if name == "unet_robust":  # U-Net behind the physical input conditioning
        return nn.Sequential(PhysInput(), Padded(UNet(), ARCHS["unet"][1]))
    if name == "unet16_robust":  # same, 4x fewer parameters (base 16: ~0.5 M), capacity axis
        return nn.Sequential(PhysInput(), Padded(UNet(base=16), ARCHS["unet"][1]))
    make, stride = ARCHS[name]
    return Padded(make(), stride)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
