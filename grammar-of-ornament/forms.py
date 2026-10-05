#!/usr/bin/env python3
"""
Grammar of Ornament — forms.

Turns every image in features/ into contiguous black-and-white regions in
forms/. Instead of plain grayscale, each image is projected onto the axis of
colour along which it varies most (the first principal component of its
pixels in Lab space), so a feature that is, say, orange-vs-blue at constant
brightness still splits cleanly.

Pipeline (per image):
  1. upscale ×SCALE (bicubic) for smoother region boundaries
  2. Gaussian blur (--sigma) to suppress the fine hatching
  3. PCA over the pixels' Lab colours; project onto the first component
  4. Otsu threshold → two classes; the darker class (by luminance) is black
  5. remove regions and holes smaller than --min-area, then open/close with a
     disk of --smooth to round off the boundaries
  6. downsample to the original size and re-threshold so the output stays
     strictly black and white (--soft keeps anti-aliased edges instead)

Run:  python3 forms.py              # all of features/ → forms/
      python3 forms.py --help       # the knobs
Only images newer than their form are redone unless --force is given.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, morphology

HERE = Path(__file__).resolve().parent


def principal_axis(img: np.ndarray, space: str = "lab") -> np.ndarray:
    """Project an H×W×3 float image onto its first principal colour component."""
    if space == "lab":
        px = color.rgb2lab(img)
    else:
        px = img * 100.0  # comparable scale to Lab
    flat = px.reshape(-1, 3)
    mean = flat.mean(0)
    centered = flat - mean
    # covariance eigenvector with the largest eigenvalue
    cov = centered.T @ centered / len(flat)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argmax(vals)]
    proj = centered @ axis
    return proj.reshape(img.shape[:2])


def form(img: Image.Image, scale: int = 2, sigma: float = 3.0, min_area: float = 0.004,
         smooth: int = 2, space: str = "lab", invert: bool = False, soft: bool = False) -> Image.Image:
    w, h = img.size
    big = img.convert("RGB").resize((w * scale, h * scale), Image.BICUBIC)
    rgb = np.asarray(big).astype(np.float32) / 255.0
    if sigma > 0:
        rgb = np.stack([ndi.gaussian_filter(rgb[..., i], sigma) for i in range(3)], -1)
    proj = principal_axis(rgb, space)
    t = filters.threshold_otsu(proj)
    mask = proj > t
    # polarity: the class that is darker in the original becomes black
    lum = color.rgb2gray(rgb)
    if lum[mask].mean() > lum[~mask].mean():
        mask = ~mask
    if invert:
        mask = ~mask
    area = int(min_area * mask.size)
    if area > 0:
        mask = morphology.remove_small_objects(mask, min_size=area)
        mask = morphology.remove_small_holes(mask, area_threshold=area)
    if smooth > 0:
        disk = morphology.disk(smooth)
        mask = morphology.binary_opening(mask, disk)
        mask = morphology.binary_closing(mask, disk)
    ink = (255 - mask.astype(np.uint8) * 255)
    out = Image.fromarray(ink, "L").resize((w, h), Image.LANCZOS)
    if not soft:
        out = out.point(lambda v: 255 if v >= 128 else 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "features"))
    ap.add_argument("--dst", default=str(HERE / "forms"))
    ap.add_argument("--only", help="substring; process only files whose name contains it")
    ap.add_argument("--scale", type=int, default=2, help="upscale factor before processing")
    ap.add_argument("--sigma", type=float, default=3.0, help="Gaussian blur at the upscaled size (0 to disable)")
    ap.add_argument("--min-area", type=float, default=0.004, help="drop regions/holes smaller than this fraction of the image")
    ap.add_argument("--smooth", type=int, default=2, help="radius of the opening/closing disk (0 to disable)")
    ap.add_argument("--space", choices=["lab", "rgb"], default="lab", help="colour space for the principal axis")
    ap.add_argument("--invert", action="store_true", help="swap black and white")
    ap.add_argument("--soft", action="store_true", help="keep anti-aliased edges instead of strict black/white")
    ap.add_argument("--force", action="store_true", help="redo forms that are already up to date")
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
        im = form(Image.open(p), scale=a.scale, sigma=a.sigma, min_area=a.min_area, smooth=a.smooth,
                  space=a.space, invert=a.invert, soft=a.soft)
        im.save(out, optimize=True)
        done += 1
        print(f"  {p.name}", flush=True)
    print(f"  {done} done, {skipped} already up to date, {time.time() - t0:.1f}s → {dst}")


if __name__ == "__main__":
    main()
