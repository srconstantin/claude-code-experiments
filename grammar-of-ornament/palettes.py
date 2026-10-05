#!/usr/bin/env python3
"""
Grammar of Ornament — palettes.

Picks N colours (default 7) from every image in features/ and writes, to
palettes/, a strip of side-by-side swatches with their hex codes, plus a JSON
sidecar listing the colours.

Methods (all work in CIELAB, so "evenly" means perceptually even; swatches
are ordered dark to light):
  spread (default)  Farthest-point sampling over the image's common colours
                    (those covering at least --min-frac of the pixels): start
                    from the colour farthest from the mean and repeatedly add
                    the colour farthest from every colour chosen so far. Every
                    colour is a real pixel, none is a stray, and the set is
                    maximally spread through the image's colour space: red,
                    yellow and blue blobs each get a swatch.
  axis              Project the pixels onto the image's principal colour axis
                    (as forms.py does), split that range into N equal
                    intervals, and take the pixel colour nearest each
                    interval's mean: equal steps along the dominant range, but
                    averaging mixes hues, so the colours come out muted.
  kmeans            k-means with N clusters, each centre snapped to the
                    nearest real pixel colour. Representative, also muted.

Run:  python3 palettes.py                 # all of features/ → palettes/
      python3 palettes.py --method spread --n 5
Only images newer than their palette are redone unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2
from skimage import color

HERE = Path(__file__).resolve().parent
FONTS = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Helvetica.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]


def load_pixels(img: Image.Image, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb float HxWx3, lab float Nx3) after a light blur that removes hatching."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    if sigma > 0:
        rgb = np.stack([ndi.gaussian_filter(rgb[..., i], sigma) for i in range(3)], -1)
    lab = color.rgb2lab(rgb).reshape(-1, 3)
    return rgb, lab


def nearest_pixel(lab: np.ndarray, target: np.ndarray) -> int:
    return int(np.argmin(((lab - target) ** 2).sum(1)))


def pick_axis(lab: np.ndarray, n: int, clip: float = 1.0) -> list[int]:
    centered = lab - lab.mean(0)
    cov = centered.T @ centered / len(lab)
    vals, vecs = np.linalg.eigh(cov)
    proj = centered @ vecs[:, np.argmax(vals)]
    lo, hi = np.percentile(proj, [clip, 100 - clip])
    edges = np.linspace(lo, hi, n + 1)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        inside = (proj >= a) & (proj <= b)
        if not inside.any():
            inside = np.abs(proj - (a + b) / 2) <= (b - a)
        idx.append(nearest_pixel(lab, lab[inside].mean(0)))
    return idx


def common_colors(lab: np.ndarray, min_frac: float, cell: float = 5.0) -> np.ndarray:
    """Indices of pixels whose colour is common: its Lab cell (cell units wide)
    holds at least min_frac of all pixels. Keeps stray pixels out of the palette."""
    keys = np.floor(lab / cell).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    ok = counts[inv.ravel()] >= max(1, min_frac * len(lab))
    return np.flatnonzero(ok) if ok.any() else np.arange(len(lab))


def pick_spread(lab: np.ndarray, n: int, min_frac: float = 0.0002) -> list[int]:
    cand = common_colors(lab, min_frac)
    pts = lab[cand]
    first = int(np.argmax(((pts - pts.mean(0)) ** 2).sum(1)))
    chosen = [first]
    dmin = ((pts - pts[first]) ** 2).sum(1)
    while len(chosen) < min(n, len(pts)):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, ((pts - pts[j]) ** 2).sum(1))
    return [int(cand[j]) for j in chosen]


def pick_kmeans(lab: np.ndarray, n: int, seed: int = 0) -> list[int]:
    centres, _ = kmeans2(lab, n, minit="++", seed=seed)
    return [nearest_pixel(lab, c) for c in centres]


def order_dark_to_light(lab: np.ndarray, idx: list[int]) -> list[int]:
    return [idx[i] for i in np.argsort(lab[idx][:, 0])]


def palette(img: Image.Image, n: int, method: str, sigma: float, min_frac: float = 0.0002) -> list[str]:
    rgb, lab = load_pixels(img, sigma)
    if method == "axis":
        idx = pick_axis(lab, n)
        if lab[idx[0], 0] > lab[idx[-1], 0]:  # keep the axis order, but dark on the left
            idx = idx[::-1]
    elif method == "spread":
        idx = order_dark_to_light(lab, pick_spread(lab, n, min_frac))
    else:
        idx = order_dark_to_light(lab, pick_kmeans(lab, n))
    flat = (rgb.reshape(-1, 3) * 255).round().astype(int)
    return ["#%02x%02x%02x" % tuple(flat[i]) for i in idx]


def font(size: int) -> ImageFont.ImageFont:
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def strip(hexes: list[str], swatch: int = 160, label_h: int = 44) -> Image.Image:
    n = len(hexes)
    im = Image.new("RGB", (swatch * n, swatch + label_h), "white")
    pen = ImageDraw.Draw(im)
    f = font(18)
    for i, hx in enumerate(hexes):
        pen.rectangle([i * swatch, 0, (i + 1) * swatch - 1, swatch - 1], fill=hx)
        tw = pen.textlength(hx, font=f)
        pen.text((i * swatch + (swatch - tw) / 2, swatch + 12), hx, fill="#222", font=f)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "features"))
    ap.add_argument("--dst", default=str(HERE / "palettes"))
    ap.add_argument("--only", help="substring; process only files whose name contains it")
    ap.add_argument("--n", type=int, default=7, help="number of colours")
    ap.add_argument("--method", choices=["spread", "axis", "kmeans"], default="spread")
    ap.add_argument("--sigma", type=float, default=1.5, help="blur before sampling, to ignore the fine hatching")
    ap.add_argument("--min-frac", type=float, default=0.0002,
                    help="spread: only colours covering at least this fraction of the image are candidates "
                         "(0.0002 ≈ 10 pixels of a 224px image: excludes strays, keeps every blob)")
    ap.add_argument("--swatch", type=int, default=160, help="swatch size in pixels")
    ap.add_argument("--force", action="store_true", help="redo palettes that are already up to date")
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
        hexes = palette(Image.open(p), a.n, a.method, a.sigma, a.min_frac)
        strip(hexes, a.swatch).save(out, optimize=True)
        (dst / f"{p.stem}.json").write_text(json.dumps(
            {"source": p.name, "method": a.method, "colors": hexes}, indent=1))
        done += 1
        print(f"  {p.name}  {' '.join(hexes)}", flush=True)
    print(f"  {done} done, {skipped} already up to date, {time.time() - t0:.1f}s → {dst}")


if __name__ == "__main__":
    main()
