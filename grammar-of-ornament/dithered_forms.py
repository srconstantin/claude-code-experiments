#!/usr/bin/env python3
"""
Grammar of Ornament — dithered forms.

Turns every image in features/ into a dithered one-bit image in
dithered-forms/. Like forms.py, each image is first projected onto its axis of
greatest colour variation (the first principal component of its pixels in
Lab space), oriented so that brighter in the original is lighter here; but
instead of thresholding into two regions, the projection is kept as a
continuous tone and rendered with black-and-white dithering.

Pipeline (per image):
  1. optional upscale ×SCALE (so the dither pattern is finer relative to the forms)
  2. Gaussian blur (--sigma) to suppress the fine hatching
  3. PCA over the pixels' Lab colours; project onto the first component
  4. stretch the projection between its --clip percentiles, apply --gamma
  5. dither to 1-bit: Floyd–Steinberg error diffusion (default) or an ordered
     8×8 Bayer matrix (--method bayer)

Run:  python3 dithered_forms.py             # all of features/ → dithered-forms/
      python3 dithered_forms.py --help      # the knobs
Only images newer than their output are redone unless --force is given.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color

from forms import principal_axis

HERE = Path(__file__).resolve().parent

BAYER8 = (np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
]) + 0.5) / 64.0


def tone(img: Image.Image, scale: int = 2, sigma: float = 1.0, clip: float = 1.0,
         gamma: float = 1.0, space: str = "lab", invert: bool = False) -> np.ndarray:
    """Continuous-tone version of the image along its principal colour axis, in [0, 1]."""
    w, h = img.size
    big = img.convert("RGB").resize((w * scale, h * scale), Image.BICUBIC) if scale != 1 else img.convert("RGB")
    rgb = np.asarray(big).astype(np.float32) / 255.0
    if sigma > 0:
        rgb = np.stack([ndi.gaussian_filter(rgb[..., i], sigma) for i in range(3)], -1)
    proj = principal_axis(rgb, space)
    # orient the axis so that brighter in the original reads as lighter
    lum = color.rgb2gray(rgb)
    if np.corrcoef(proj.ravel(), lum.ravel())[0, 1] < 0:
        proj = -proj
    if invert:
        proj = -proj
    lo, hi = np.percentile(proj, [clip, 100 - clip])
    t = np.clip((proj - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        t = t ** gamma
    return t


def dither(t: np.ndarray, method: str = "fs") -> Image.Image:
    if method == "bayer":
        h, w = t.shape
        thr = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w]
        bits = t > thr
        return Image.fromarray((bits * 255).astype(np.uint8), "L").convert("1")
    gray = Image.fromarray((t * 255).round().astype(np.uint8), "L")
    return gray.convert("1", dither=Image.Dither.FLOYDSTEINBERG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "features"))
    ap.add_argument("--dst", default=str(HERE / "dithered-forms"))
    ap.add_argument("--only", help="substring; process only files whose name contains it")
    ap.add_argument("--scale", type=int, default=2, help="upscale factor; the dither pattern is at the output size (1 = native, grainy)")
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian blur before projection (0 to disable)")
    ap.add_argument("--clip", type=float, default=1.0, help="percentile clipped at each end of the tone range")
    ap.add_argument("--gamma", type=float, default=1.0, help="tone curve; >1 darkens midtones, <1 lightens")
    ap.add_argument("--method", choices=["fs", "bayer"], default="fs", help="Floyd–Steinberg or ordered Bayer dithering")
    ap.add_argument("--space", choices=["lab", "rgb"], default="lab", help="colour space for the principal axis")
    ap.add_argument("--invert", action="store_true", help="swap light and dark")
    ap.add_argument("--force", action="store_true", help="redo outputs that are already up to date")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.glob("*.png") if not a.only or a.only in p.name)
    if not files:
        print(f"no PNGs in {src}")
        sys.exit(1)
    done = skipped = 0
    t0 = time.time()
    for p in files:
        out = dst / p.name
        if not a.force and out.exists() and out.stat().st_mtime >= p.stat().st_mtime:
            skipped += 1
            continue
        t = tone(Image.open(p), scale=a.scale, sigma=a.sigma, clip=a.clip, gamma=a.gamma,
                 space=a.space, invert=a.invert)
        dither(t, a.method).save(out, optimize=True)
        done += 1
        print(f"  {p.name}", flush=True)
    print(f"  {done} done, {skipped} already up to date, {time.time() - t0:.1f}s → {dst}")


if __name__ == "__main__":
    main()
