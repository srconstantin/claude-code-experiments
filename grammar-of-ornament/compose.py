#!/usr/bin/env python3
"""
Grammar of Ornament — compose: one picture painted by the network to a layout.

Ordinary feature visualization fills a square with one channel. Here a single
poster-shaped image is optimized against a layout of masks:

  figure   a silhouette (a cleaned motif's body, scaled up and placed off
           centre) where a deep channel, the architectural or creature-like
           kind, is driven up
  ground   everything else, where a shallow channel, the Turing-pattern or
           cellular kind, is driven up
  quiet    a title block that is held at a flat paper tone in the picture the
           network sees, so it must arrange its forms around the space

The masks go through the same random jitter, scale and rotation as the image,
so they stay registered with it, and are shrunk to each layer's map size. Each
term is divided by its own current size, so neither layer drowns the other.

Output: posters/composed/<name>.png (the raw render) and <name>.json (the recipe).
Run:  python3 compose.py --preset vgg      # or resnet, convnext, all
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

import vis

HERE = Path(__file__).resolve().parent
OUT = HERE / "posters" / "composed"
CLEAN = HERE / "vgg16motifs-clean"

PRESETS = {
    "vgg": {"model": "vgg16", "figure": ("features.24", 119), "ground": ("features.10", 1), "shape": "features.24__c119"},
    "vgg2": {"model": "vgg16", "figure": ("features.24", 141), "ground": ("features.7", 58), "shape": "features.21__c78"},
    "resnet": {"model": "resnet50", "figure": ("layer3.3", 381), "ground": ("layer1.1", 201), "shape": "features.24__c27"},
    "convnext": {"model": "convnext_base", "figure": ("features.5.6", 245), "ground": ("features.3.1", 228), "shape": "features.24__c66"},
}


def shape_mask(key: str) -> np.ndarray:
    f = next(iter(sorted(CLEAN.glob(f"*{key}__*/*/mask.png"))))
    return np.asarray(Image.open(f).convert("L")) > 127


def layout(h: int, w: int, key: str, seed: int) -> dict:
    """figure, ground and quiet masks (float, 0–1) at the poster size."""
    rng = np.random.default_rng(seed)
    m = shape_mask(key).astype(np.float32)
    s = 0.78 * h / m.shape[0]
    m = ndi.zoom(m, s, order=1)
    if m.shape[1] > 0.95 * w:
        m = ndi.zoom(m, 0.95 * w / m.shape[1], order=1)
    fig = np.zeros((h, w), np.float32)
    y0 = int(0.04 * h)
    x0 = int(w - 0.82 * m.shape[1]) if rng.random() < 0.5 else int(-0.18 * m.shape[1])   # cropped off one side
    ys, xs = slice(max(0, y0), min(h, y0 + m.shape[0])), slice(max(0, x0), min(w, x0 + m.shape[1]))
    fig[ys, xs] = m[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    fig = ndi.gaussian_filter((fig > 0.5).astype(np.float32), 3)
    quiet = np.zeros((h, w), np.float32)
    quiet[int(0.80 * h):int(0.975 * h), int(0.05 * w):int(0.95 * w)] = 1
    quiet = ndi.gaussian_filter(quiet, 4)
    fig = fig * (1 - quiet)
    ground = np.clip(1 - fig - quiet, 0, 1)
    return {"figure": fig, "ground": ground, "quiet": quiet}


def compose(p: dict, h: int, w: int, steps: int, seed: int, paper: float = 0.92) -> np.ndarray:
    lm = vis.load_model(p["model"])
    (lf, cf), (lg, cg) = p["figure"], p["ground"]
    masks = layout(h, w, p["shape"], seed)
    M = torch.tensor(np.stack([masks["figure"], masks["ground"], masks["quiet"]])[None], device=vis.DEVICE)
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    params, image_fn = vis.fft_param(1, h, w, decay_power=1.25, generator=gen)
    opt = torch.optim.Adam(params, lr=0.05)
    mean = torch.tensor(lm.mean, device=vis.DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(lm.std, device=vis.DEVICE).view(1, 3, 1, 1)
    acts = {}

    class Stop(Exception):
        pass

    def hook(name, last):
        def f(_m, _i, out):
            acts[name] = out
            if last:
                raise Stop()
        return f

    order = [l.name for l in lm.layers]
    deep_last = order.index(lf) > order.index(lg)
    hs = [lm.modules[lf].register_forward_hook(hook("fig", deep_last)),
          lm.modules[lg].register_forward_hook(hook("gr", not deep_last))]

    def composite():
        raw = image_fn()
        q = M[:, 2:3]
        return raw * (1 - q) + paper * q

    t0 = time.time()
    try:
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            x = torch.cat([composite(), M], 1)          # masks ride along through the transforms
            x = vis.standard_transforms(x, rng)
            img, mk = x[:, :3], x[:, 3:].clamp(0, 1)
            try:
                lm.model((img - mean) / std)
            except Stop:
                pass

            def term(a, c, k):
                mm = F.interpolate(mk[:, k:k + 1], size=a.shape[2:], mode="bilinear", antialias=False, align_corners=False)
                return (a[:, c:c + 1] * mm).sum() / (mm.sum() + 1e-6)

            af, ag = acts["fig"], acts["gr"]
            A = term(af, cf, 0)                          # deep channel in the figure
            B = term(ag, cg, 1)                          # shallow channel in the ground
            C = term(af, cf, 1)                          # deep channel leaking into the ground
            loss = -A / (A.detach().abs() + 1e-3) - B / (B.detach().abs() + 1e-3) + 0.3 * C / (A.detach().abs() + 1e-3)
            loss.backward()
            opt.step()
            if step % 64 == 0 or step == steps - 1:
                print(f"    step {step + 1}/{steps}  figure {A.item():.2f}  ground {B.item():.2f}  leak {C.item():.2f}  {time.time() - t0:.0f}s", flush=True)
    finally:
        for hd in hs:
            hd.remove()
    with torch.no_grad():
        out = composite()[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (out * 255).round().astype(np.uint8), masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="all")
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--w", type=int, default=448)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = list(PRESETS) if a.preset == "all" else a.preset.split(",")
    for n in names:
        p = PRESETS[n]
        print(f"composing {n}: {p['model']} figure {p['figure']} ground {p['ground']} in the shape of {p['shape']}", flush=True)
        img, masks = compose(p, a.h, a.w, a.steps, a.seed)
        Image.fromarray(img).save(OUT / f"{n}.png")
        Image.fromarray((np.stack([masks["figure"], masks["ground"], masks["quiet"]], -1) * 255).astype(np.uint8)).save(OUT / f"{n}.masks.png")
        (OUT / f"{n}.json").write_text(json.dumps({**p, "h": a.h, "w": a.w, "steps": a.steps, "seed": a.seed}, indent=1))
        print(f"  wrote {OUT / (n + '.png')}", flush=True)


if __name__ == "__main__":
    main()
