#!/usr/bin/env python3
"""Seamless upscales of the ConvNeXt features (the colorway features), for page backgrounds that tile.

The upscaler draws artifacts along an image's edges, so each pass pads the image with a border wrapped round
from its opposite sides, upscales, and cuts the border away (as posters.py does for its 4× tiles); the last
step cross-fades the edges with a half-shifted copy, so whatever seam is left disappears.

  pass 1   224 → 896 px    → cache/sr_tile/<name>.png    (kept if present)
  pass 2   896 → 3584 px   → cache/sr16_tile/<name>.jpg  and cache/sr8_tile/<name>.jpg (1792 px)

  .venv/bin/python library/tile_all.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "library"))
import motifs  # noqa: E402
from upscale_all import run, DEV, log  # noqa: E402

SRC = ROOT / "convnext_basefeatures"
T4, T16, T8 = ROOT / "cache" / "sr_tile", ROOT / "cache" / "sr16_tile", ROOT / "cache" / "sr8_tile"


def seamless(a, band=0.18):
    """posters.seamless: near the edges, cross-fade to a copy shifted by half a tile."""
    h, w = a.shape[:2]
    ramp1 = lambda n: np.clip(np.minimum(np.arange(n), n - 1 - np.arange(n)) / (band * n), 0, 1)
    wgt = np.minimum.outer(ramp1(h), ramp1(w))
    wgt = (wgt * wgt * (3 - 2 * wgt))[..., None]
    return a * wgt + np.roll(a, (h // 2, w // 2), (0, 1)) * (1 - wgt)


def wrapped(net, rgb, b, tile=None):
    """rgb upscaled 4× with a wrapped border of b px that is cut away again; tiled with context when tile is set."""
    h, w = rgb.shape[:2]
    p = np.pad(rgb, ((b, b), (b, b), (0, 0)), mode="wrap")
    if tile is None:
        return run(net, p)[4 * b:4 * (b + h), 4 * b:4 * (b + w)]
    out = np.zeros((4 * h, 4 * w, 3), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            t = p[y0:y0 + tile + 2 * b, x0:x0 + tile + 2 * b]
            out[4 * y0:4 * (y0 + tile), 4 * x0:4 * (x0 + tile)] = run(net, t)[4 * b:4 * (b + tile), 4 * b:4 * (b + tile)]
    return out


def main():
    net = motifs._net().to(DEV)
    for d in (T4, T16, T8):
        d.mkdir(parents=True, exist_ok=True)
    feats = sorted(SRC.glob("*.png"))
    log(f"{len(feats)} ConvNeXt features, seamless, on {DEV}")
    t0 = time.time()
    for n, f in enumerate(feats, 1):
        name = f.stem
        t4 = T4 / f"{name}.png"
        if not t4.exists():
            rgb = np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255
            Image.fromarray((wrapped(net, rgb, 24) * 255).round().astype(np.uint8)).save(t4)
        o16, o8 = T16 / f"{name}.jpg", T8 / f"{name}.jpg"
        if not o16.exists() or not o8.exists():
            t = time.time()
            rgb = np.asarray(Image.open(t4).convert("RGB")).astype(np.float32) / 255
            big = seamless(wrapped(net, rgb, 32, tile=448))
            im = Image.fromarray((big * 255).round().astype(np.uint8))
            im.save(o16, quality=92)
            im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(o8, quality=90)
            log(f"[{n}/{len(feats)}] {name[:44]} {time.time() - t:.0f}s")
    log(f"done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
