#!/usr/bin/env python3
"""
Grammar of Ornament — outlines.

Turns every image in features/ into a line drawing in outlines/, built on the
same continuous-tone field as dithered_forms.py: each pixel projected onto the
image's axis of greatest colour variation (first principal component in Lab).

Methods:
  xdog (default)     Extended difference-of-Gaussians (Winnemöller 2011): an
                     ink-drawing filter. Dark where the tone is locally darker
                     than its surroundings, white elsewhere, with a soft
                     threshold; reads like a woodcut. --xdog-* set its knobs.
  contour            Isolines of the smoothed tone field at a few brightness
                     levels, like a topographic map. Isolines run perpendicular
                     to the brightness gradient, so they follow the curves of
                     the shapes, and they are closed, continuous curves.
  canny              Classic Canny edges on the tone field (the old approach;
                     sharpest gradients only, tends to fragment soft shapes).

Run:  python3 outlines.py                                # all of features/ → outlines/
      python3 outlines.py --method contour               # topo-map isolines instead
      python3 outlines.py --method contour --levels 0.5  # a single contour per form
Only images newer than their outline are redone unless --force is given.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import feature, measure, morphology

from dithered_forms import tone

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
#  contour method
# ─────────────────────────────────────────────────────────────────────────────

def contour_drawing(t: np.ndarray, levels: list[float], sigma: float, min_len: float,
                    width: float, draw_scale: int = 2) -> Image.Image:
    """Draw isolines of the tone field t (values in [0,1]) at the given levels.

    The field is smoothed by sigma first; contours shorter than min_len pixels
    are dropped. Lines are drawn at draw_scale× and downsampled for anti-aliasing.
    """
    h, w = t.shape
    f = ndi.gaussian_filter(t, sigma) if sigma > 0 else t
    canvas = Image.new("L", (w * draw_scale, h * draw_scale), 255)
    pen = ImageDraw.Draw(canvas)
    for lv in levels:
        for c in measure.find_contours(f, lv):
            # c is (row, col) pairs; perimeter in pixels
            if len(c) < 3:
                continue
            length = np.sqrt((np.diff(c, axis=0) ** 2).sum(1)).sum()
            if length < min_len:
                continue
            pts = [(float(x) * draw_scale, float(y) * draw_scale) for y, x in c]
            pen.line(pts, fill=0, width=max(1, round(width * draw_scale)), joint="curve")
    return canvas.resize((w, h), Image.LANCZOS)


# ─────────────────────────────────────────────────────────────────────────────
#  xdog method
# ─────────────────────────────────────────────────────────────────────────────

def xdog(t: np.ndarray, sigma: float, k: float = 1.6, p: float = 20.0, eps: float = 0.0,
         phi: float = 10.0) -> Image.Image:
    g1 = ndi.gaussian_filter(t, sigma)
    g2 = ndi.gaussian_filter(t, sigma * k)
    d = (1 + p) * g1 - p * g2
    u = np.where(d >= eps, 1.0, 1.0 + np.tanh(phi * (d - eps)))
    return Image.fromarray((np.clip(u, 0, 1) * 255).astype(np.uint8), "L")


# ─────────────────────────────────────────────────────────────────────────────
#  canny method (kept for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def canny_drawing(t: np.ndarray, sigma: float, min_len: float, width: float) -> Image.Image:
    edges = feature.canny(t, sigma=sigma, low_threshold=0.5, high_threshold=0.9, use_quantiles=True)
    edges = morphology.remove_small_objects(edges, min_size=int(min_len), connectivity=2)
    edges = morphology.binary_closing(edges, morphology.disk(1))
    edges = morphology.skeletonize(edges)
    if width > 1:
        edges = morphology.binary_dilation(edges, morphology.disk((width - 1) / 2))
    return Image.fromarray(255 - edges.astype(np.uint8) * 255, "L")


# ─────────────────────────────────────────────────────────────────────────────

def outline(img: Image.Image, method: str, scale: int, sigma: float, levels: list[float],
            min_len: float, width: float, tone_sigma: float, gamma: float, invert: bool,
            xdog_params: dict | None = None) -> Image.Image:
    w, h = img.size
    t = tone(img, scale=scale, sigma=tone_sigma, gamma=gamma, invert=invert)
    if method == "contour":
        out = contour_drawing(t, levels, sigma, min_len, width)
    elif method == "xdog":
        out = xdog(t, sigma, **(xdog_params or {}))
    else:
        out = canny_drawing(t, sigma, min_len, width)
    return out.resize((w, h), Image.LANCZOS) if out.size != (w, h) else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "features"))
    ap.add_argument("--dst", default=str(HERE / "outlines"))
    ap.add_argument("--only", help="substring; process only files whose name contains it")
    ap.add_argument("--method", choices=["xdog", "contour", "canny"], default="xdog")
    ap.add_argument("--scale", type=int, default=2, help="work at this multiple of the image size")
    ap.add_argument("--levels", default="0.2,0.35,0.5,0.65,0.8",
                    help="contour method: comma-separated tone levels in 0–1 (0.5 alone gives the form boundary)")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="smoothing of the tone field before drawing, at the working size")
    ap.add_argument("--min-len", type=float, default=40, help="drop contours shorter than this (px at working size)")
    ap.add_argument("--width", type=float, default=1.8, help="line width at the working size")
    ap.add_argument("--tone-sigma", type=float, default=1.0, help="blur before the colour projection (as in dithered_forms)")
    ap.add_argument("--gamma", type=float, default=1.0, help="tone curve before contouring")
    ap.add_argument("--invert", action="store_true", help="swap light and dark in the tone field")
    ap.add_argument("--xdog-k", type=float, default=1.6, help="xdog: ratio of the two Gaussian scales")
    ap.add_argument("--xdog-p", type=float, default=40.0, help="xdog: edge sharpening strength")
    ap.add_argument("--xdog-eps", type=float, default=0.6, help="xdog: threshold; higher inks more of the image")
    ap.add_argument("--xdog-phi", type=float, default=5.0, help="xdog: softness of the threshold (higher = harder)")
    ap.add_argument("--force", action="store_true", help="redo outlines that are already up to date")
    a = ap.parse_args()

    levels = [float(v) for v in a.levels.split(",") if v.strip()]
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
        im = outline(Image.open(p), a.method, a.scale, a.sigma, levels, a.min_len, a.width,
                     a.tone_sigma, a.gamma, a.invert,
                     {"k": a.xdog_k, "p": a.xdog_p, "eps": a.xdog_eps, "phi": a.xdog_phi})
        im.save(out, optimize=True)
        done += 1
        print(f"  {p.name}", flush=True)
    print(f"  {done} outlined, {skipped} already up to date, {time.time() - t0:.1f}s → {dst}")


if __name__ == "__main__":
    main()
