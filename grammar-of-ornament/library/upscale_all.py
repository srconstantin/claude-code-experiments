#!/usr/bin/env python3
"""Upscale every feature of every model, twice, on the GPU (MPS) when there is one.

  pass 1   224 → 896 px   (4×, Real-ESRGAN as motifs.py runs it)      → cache/sr/<name>.png     (kept if present)
  pass 2   896 → 3584 px  (4× again, in four tiles with context)      → cache/sr16/<name>.jpg   (quality 92)
                                                                      → cache/sr8/<name>.jpg    (1792 px, quality 90)

Resumable: anything already made is skipped. Prints a line per feature.

  .venv/bin/python library/upscale_all.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import motifs  # noqa: E402

MODELS = ["convnext_basefeatures", "vgg16features", "resnet50features"]
SR, SR16, SR8 = ROOT / "cache" / "sr", ROOT / "cache" / "sr16", ROOT / "cache" / "sr8"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PAD = 8          # reflect pad before the net, as motifs.py does
CTX = 32         # context round each tile in pass 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(net, rgb):
    """rgb float HxWx3 in 0–1 → 4× upscale, same convention as motifs.upscaled."""
    with torch.no_grad():
        x = F.pad(torch.from_numpy(rgb.transpose(2, 0, 1)[None]).to(DEV), (PAD, PAD, PAD, PAD), mode="reflect")
        y = net(x)[0, :, 4 * PAD:-4 * PAD, 4 * PAD:-4 * PAD].clamp(0, 1)
        return y.cpu().numpy().transpose(1, 2, 0)


def pass1(net, src: Path, out: Path):
    rgb = np.asarray(Image.open(src).convert("RGB")).astype(np.float32) / 255
    y = run(net, rgb)
    Image.fromarray((y * 255).round().astype(np.uint8)).save(out)


def pass2(net, src: Path, out16: Path, out8: Path, tile=448):
    """The 896 px upscale in a 2×2 of tiles, each run with CTX px of context, stitched without seams."""
    im = np.asarray(Image.open(src).convert("RGB")).astype(np.float32) / 255
    n = im.shape[0]
    k = n // tile
    padded = np.pad(im, ((CTX, CTX), (CTX, CTX), (0, 0)), mode="reflect")
    out = np.zeros((4 * n, 4 * n, 3), np.float32)
    for i in range(k):
        for j in range(k):
            y0, x0 = i * tile, j * tile
            t = padded[y0:y0 + tile + 2 * CTX, x0:x0 + tile + 2 * CTX]
            y = run(net, t)
            out[4 * y0:4 * (y0 + tile), 4 * x0:4 * (x0 + tile)] = y[4 * CTX:4 * CTX + 4 * tile, 4 * CTX:4 * CTX + 4 * tile]
    big = Image.fromarray((out * 255).round().astype(np.uint8))
    big.save(out16, quality=92)
    big.resize((2 * n, 2 * n), Image.LANCZOS).save(out8, quality=90)


def main():
    net = motifs._net().to(DEV)
    SR16.mkdir(parents=True, exist_ok=True)
    SR8.mkdir(parents=True, exist_ok=True)
    feats = [(m, f) for m in MODELS for f in sorted((ROOT / m).glob("*.png"))]
    log(f"{len(feats)} features on {DEV}")
    t0 = time.time()
    for n, (m, f) in enumerate(feats, 1):
        name = f.stem
        sr = SR / f"{name}.png"
        if not sr.exists():
            t = time.time()
            pass1(net, f, sr)
            log(f"[{n}/{len(feats)}] pass 1 {name[:44]} {time.time() - t:.0f}s")
        o16, o8 = SR16 / f"{name}.jpg", SR8 / f"{name}.jpg"
        if not o16.exists() or not o8.exists():
            t = time.time()
            pass2(net, sr, o16, o8)
            log(f"[{n}/{len(feats)}] pass 2 {name[:44]} {time.time() - t:.0f}s")
    log(f"done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
