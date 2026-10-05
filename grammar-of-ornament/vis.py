#!/usr/bin/env python3
"""
Grammar of Ornament — feature visualization core.

Activation maximization in the style of Lucid/lucent (Olah et al. 2017):
an image is parameterized in Fourier space with decorrelated colours, pushed
through random jitter/scale/rotation each step, and optimized with Adam so that
one channel (or one neuron at a spatial location) of a chosen layer fires as
hard as possible. Everything here is plain torch/torchvision/timm; no
visualization library is needed.

CLI smoke test:
  python3 vis.py --model convnext_base --layer features.1.0 --channels 0-15 --preset thumb
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    want = os.environ.get("GOO_DEVICE")
    if want:
        return torch.device(want)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = pick_device()

# Does the inverse FFT run on this device? If not, do it on the CPU and move the
# result over (slower, but the rest of the pipeline still runs on the GPU).
FFT_ON_DEVICE = True
try:
    _probe = torch.randn(1, 3, 8, 5, 2, device=DEVICE)
    torch.fft.irfftn(torch.view_as_complex(_probe), s=(8, 8), norm="ortho")
except Exception:  # noqa: BLE001
    FFT_ON_DEVICE = False
finally:
    _probe = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ─────────────────────────────────────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    name: str
    model: nn.Module
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    input_size: int = 224
    layers: list["LayerInfo"] = field(default_factory=list)
    modules: dict[str, nn.Module] = field(default_factory=dict)


@dataclass
class LayerInfo:
    name: str
    channels: int
    height: int
    width: int
    kind: str  # "conv" or the block's class name
    depth: int  # 0-based position in forward order


def _tv(name: str):
    import torchvision.models as tvm

    if name == "convnext_base":
        return tvm.convnext_base(weights=tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    if name == "convnext_tiny":
        return tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    if name == "resnet50":
        return tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    if name == "vgg16":
        return tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
    raise KeyError(name)


# Built-in choices shown in the UI, in menu order. Any other name is tried as a
# timm model ("timm:<name>" or just "<name>").
BUILTIN_MODELS = ["convnext_base", "convnext_tiny", "resnet50", "vgg16"]

_LOADED: dict[str, LoadedModel] = {}


def load_model(name: str) -> LoadedModel:
    """Load (and cache) a pretrained model plus its layer catalog."""
    if name in _LOADED:
        return _LOADED[name]
    t0 = time.time()
    if name in BUILTIN_MODELS:
        model = _tv(name)
        mean, std, size = IMAGENET_MEAN, IMAGENET_STD, 224
    else:
        import timm
        from timm.data import resolve_data_config

        tname = name[5:] if name.startswith("timm:") else name
        model = timm.create_model(tname, pretrained=True)
        cfg = resolve_data_config({}, model=model)
        mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
        size = int(cfg["input_size"][-1])
    model = model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    lm = LoadedModel(name=name, model=model, mean=mean, std=std, input_size=size)
    lm.layers, lm.modules = catalog_layers(lm)
    if not lm.layers:
        raise ValueError(f"{name} has no 4-D (conv-like) layers to visualize; pick a convnet.")
    print(f"  loaded {name} on {DEVICE} in {time.time() - t0:.1f}s: {len(lm.layers)} layers", flush=True)
    _LOADED[name] = lm
    return lm


def catalog_layers(lm: LoadedModel) -> tuple[list[LayerInfo], dict[str, nn.Module]]:
    """One dry forward pass; keep every module whose output is an N×C×H×W map.

    Leaf convolutions and non-leaf blocks (residual blocks, ConvNeXt blocks…)
    are kept. Plain containers (Sequential stages) that just return their last
    child's output are dropped as duplicates. Order is forward order, shallow
    first.
    """
    records: list[tuple[str, nn.Module, torch.Tensor]] = []
    handles = []

    def make_hook(name: str, module: nn.Module):
        def hook(_m, _inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() != 4 or out.shape[0] != 1:
                return
            is_leaf = len(list(module.children())) == 0
            if is_leaf and not isinstance(module, nn.Conv2d):
                return
            if not is_leaf and isinstance(module, (nn.Sequential, nn.ModuleList)):
                return  # plain containers: their children are catalogued instead
            if records and records[-1][1] is not module and torch.equal(records[-1][2], out):
                return
            records.append((name, module, out.detach()))
        return hook

    for name, module in lm.model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_hook(make_hook(name, module)))
    try:
        with torch.no_grad():
            x = torch.zeros(1, 3, lm.input_size, lm.input_size, device=DEVICE)
            lm.model(x)
    finally:
        for h in handles:
            h.remove()

    layers, modules = [], {}
    for i, (name, module, out) in enumerate(records):
        kind = "conv" if isinstance(module, nn.Conv2d) else type(module).__name__
        layers.append(LayerInfo(name, int(out.shape[1]), int(out.shape[2]), int(out.shape[3]), kind, i))
        modules[name] = module
    return layers, modules


# ─────────────────────────────────────────────────────────────────────────────
#  Image parameterization (FFT + decorrelated colour), after lucent
# ─────────────────────────────────────────────────────────────────────────────

_COLOR_CORR = np.asarray([[0.26, 0.09, 0.02],
                          [0.27, 0.00, -0.05],
                          [0.27, -0.09, 0.03]], dtype=np.float32)
_COLOR_CORR = _COLOR_CORR / np.max(np.linalg.norm(_COLOR_CORR, axis=0))
COLOR_CORR_T = torch.tensor(_COLOR_CORR.T)


def _rfft2d_freqs(h: int, w: int) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[: w // 2 + 1 + (w % 2)]
    return np.sqrt(fx * fx + fy * fy)


def fft_param(batch: int, h: int, w: int, sd: float = 0.01, decay_power: float = 1.0,
              generator: torch.Generator | None = None):
    """Returns (trainable params, fn -> raw image batch in roughly [-1, 1])."""
    freqs = _rfft2d_freqs(h, w)
    init = torch.randn((batch, 3) + freqs.shape + (2,), generator=generator) * sd
    fft_dev = DEVICE if FFT_ON_DEVICE else torch.device("cpu")
    spectrum = init.to(fft_dev).requires_grad_(True)
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale_t = torch.tensor(scale, dtype=torch.float32)[None, None, ..., None].to(fft_dev)
    color = COLOR_CORR_T.to(DEVICE)

    def image() -> torch.Tensor:
        s = torch.view_as_complex((scale_t * spectrum).contiguous())
        img = torch.fft.irfftn(s, s=(h, w), norm="ortho")[:, :, :h, :w] / 4.0
        img = img.to(DEVICE)
        img = torch.matmul(img.permute(0, 2, 3, 1), color).permute(0, 3, 1, 2)
        return torch.sigmoid(img)

    return [spectrum], image


# ─────────────────────────────────────────────────────────────────────────────
#  Transformation robustness
# ─────────────────────────────────────────────────────────────────────────────

SCALES = [1 + (i - 5) / 50.0 for i in range(11)]          # 0.90 … 1.10
ANGLES = list(range(-10, 11)) + 5 * [0]                   # −10° … 10°, biased to 0


def _jitter(x: torch.Tensor, d: int, rng: random.Random) -> torch.Tensor:
    return torch.roll(x, shifts=(rng.randint(-d, d), rng.randint(-d, d)), dims=(2, 3))


def _scale(x: torch.Tensor, s: float) -> torch.Tensor:
    if s == 1.0:
        return x
    h, w = x.shape[2:]
    y = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
    nh, nw = y.shape[2:]
    if nh >= h:
        t, l = (nh - h) // 2, (nw - w) // 2
        return y[:, :, t:t + h, l:l + w]
    pt, pl = (h - nh) // 2, (w - nw) // 2
    return F.pad(y, (pl, w - nw - pl, pt, h - nh - pt), mode="constant", value=0.5)


def _rotate(x: torch.Tensor, deg: float) -> torch.Tensor:
    if deg == 0:
        return x
    a = math.radians(deg)
    theta = torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                          [math.sin(a), math.cos(a), 0.0]], device=x.device).expand(x.shape[0], 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def standard_transforms(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    x = F.pad(x, (12, 12, 12, 12), mode="constant", value=0.5)
    x = _jitter(x, 8, rng)
    x = _scale(x, rng.choice(SCALES))
    x = _rotate(x, rng.choice(ANGLES))
    x = _jitter(x, 4, rng)
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  Objectives and rendering
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Target:
    """One thing to maximize: a channel, or one neuron of it at (y, x).

    y and x are given in the layer's catalog coordinates (the H×W reported by
    LayerInfo). During optimization the feature map is a different size (the
    image is padded and jittered), so the location is mapped proportionally.
    """
    channel: int
    y: int | None = None
    x: int | None = None
    negate: bool = False

    @property
    def is_neuron(self) -> bool:
        return self.y is not None and self.x is not None

    def label(self) -> str:
        s = f"c{self.channel}"
        if self.is_neuron:
            s += f"_y{self.y}x{self.x}"
        if self.negate:
            s += "_neg"
        return s


PRESETS = {
    "thumb": {"size": 128, "steps": 128, "decay": 1.25, "tv": 0.0},
    "full": {"size": 224, "steps": 512, "decay": 1.25, "tv": 0.0},
}


class _Stop(Exception):
    pass


def render(lm: LoadedModel, layer: str, targets: list[Target], size: int = 224, steps: int = 512,
           seed: int = 0, lr: float = 0.05, decay_power: float = 1.0, tv_weight: float = 0.0,
           progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Optimize one image per target. Returns uint8 array (B, size, size, 3).

    decay_power > 1 biases the parameterization toward low frequencies;
    tv_weight > 0 adds a total-variation penalty (1.0 ≈ a few percent of the
    objective). Both reduce high-frequency striping at the cost of detail.
    """
    if layer not in lm.modules:
        raise KeyError(f"unknown layer {layer!r} for {lm.name}")
    info = next(l for l in lm.layers if l.name == layer)
    for t in targets:
        if not 0 <= t.channel < info.channels:
            raise ValueError(f"channel {t.channel} out of range for {layer} ({info.channels} channels)")
    module = lm.modules[layer]
    B = len(targets)

    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    params, image_fn = fft_param(B, size, size, decay_power=decay_power, generator=gen)
    opt = torch.optim.Adam(params, lr=lr)
    mean = torch.tensor(lm.mean, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(lm.std, device=DEVICE).view(1, 3, 1, 1)

    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        captured["acts"] = out
        raise _Stop()

    handle = module.register_forward_hook(hook)
    try:
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            raw = image_fn()
            img = standard_transforms(raw, rng)
            x = (img - mean) / std
            try:
                lm.model(x)
            except _Stop:
                pass
            acts = captured.pop("acts")
            H, W = acts.shape[2], acts.shape[3]
            loss = torch.zeros((), device=DEVICE)
            for b, t in enumerate(targets):
                if t.is_neuron:
                    yy = min(H - 1, round(t.y / max(info.height - 1, 1) * (H - 1)))
                    xx = min(W - 1, round(t.x / max(info.width - 1, 1) * (W - 1)))
                    val = acts[b, t.channel, yy, xx]
                else:
                    val = acts[b, t.channel].mean()
                loss = loss + (val if t.negate else -val)
            if tv_weight > 0:
                # total variation, scaled relative to the objective so the
                # weight means the same thing at every layer
                tv = (raw[:, :, 1:, :] - raw[:, :, :-1, :]).abs().mean() + \
                     (raw[:, :, :, 1:] - raw[:, :, :, :-1]).abs().mean()
                loss = loss + tv_weight * tv * loss.detach().abs()
            loss.backward()
            opt.step()
            if progress and (step % 8 == 0 or step == steps - 1):
                progress(step + 1, steps)
    finally:
        handle.remove()

    with torch.no_grad():
        out = image_fn().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return (out * 255).round().astype(np.uint8)


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, optimize=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _parse_channels(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="convnext_base")
    ap.add_argument("--layer", help="layer name; omit to list layers")
    ap.add_argument("--channels", default="0-15")
    ap.add_argument("--neuron", help="y,x location (catalog coords) for a neuron objective")
    ap.add_argument("--preset", choices=list(PRESETS), default="thumb")
    ap.add_argument("--size", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decay", type=float, default=1.25, help="frequency decay power (1.0 = lucent default)")
    ap.add_argument("--tv", type=float, default=0.0, help="total-variation penalty weight")
    ap.add_argument("--out", default=str(HERE / "cache" / "_cli"))
    a = ap.parse_args()

    print(f"  device: {DEVICE}   fft on device: {FFT_ON_DEVICE}")
    lm = load_model(a.model)
    if not a.layer:
        for l in lm.layers:
            print(f"  {l.depth:3d}  {l.name:40s} {l.kind:14s} {l.channels:5d} ch  {l.height}×{l.width}")
        return
    size = a.size or PRESETS[a.preset]["size"]
    steps = a.steps or PRESETS[a.preset]["steps"]
    yx = tuple(int(v) for v in a.neuron.split(",")) if a.neuron else (None, None)
    targets = [Target(c, yx[0], yx[1]) for c in _parse_channels(a.channels)]
    t0 = time.time()
    imgs = render(lm, a.layer, targets, size=size, steps=steps, seed=a.seed,
                  decay_power=a.decay, tv_weight=a.tv,
                  progress=lambda i, n: print(f"\r  step {i}/{n}", end="", flush=True))
    print(f"\n  rendered {len(targets)} images at {size}px/{steps} steps in {time.time() - t0:.1f}s")
    out = Path(a.out)
    for t, im in zip(targets, imgs):
        p = out / f"{a.model}__{a.layer}__{t.label()}__{size}px_d{a.decay}_tv{a.tv}_s{steps}.png"
        save_png(im, p)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
