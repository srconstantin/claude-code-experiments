"""
Neuron — parts of real compounds, at stroke weight.

A whole compound whose body matches a letter's stroke is as big as a letter. A
part is a short stretch of one: the piece's outline (silhouettes.py) cut to the
ink lying along a window a..b of its spine, branches and holes included, the cut
edges rounded. Sizes are in units of the piece's body, so 1.0 = one stroke width.

  .venv/bin/python typeface/parts.py     → typeface/proofs/parts/catalog-large.png, catalog-fine.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import silhouettes as SI    # noqa: E402

PX = 44                     # raster pixels per body width
ROUND = 0.13                # cut edges are rounded over this many body widths
SPAN = 3.2                  # a catalog part's length, in body widths
REFS = ["curved-branching-line/%d" % i for i in (0, 1, 3, 7, 8, 9, 10, 11)] + \
       ["curved-sprig/%d" % i for i in (0, 3, 8, 11, 12, 13, 14, 15)] + ["wavy-sprig/%d" % i for i in (1, 3, 6, 7, 8)] + \
       ["curved-comb/%d" % i for i in (0, 1, 3, 4, 5, 6, 7, 9)] + ["straight-comb/%d" % i for i in (0, 1, 2, 3, 6, 14, 15)] + \
       ["wavy-comb/%d" % i for i in (0, 1, 3, 4, 5, 6, 7, 15)] + ["straight-feather/%d" % i for i in (1, 2, 4, 5, 6, 7)] + \
       ["curved-feather/%d" % i for i in (0, 1, 2, 3, 4, 5, 6, 7)] + ["wavy-feather/%d" % i for i in (0, 1, 2, 3, 4, 5)] + \
       ["mesh/%d" % i for i in (0, 1, 2, 3, 4, 6, 7, 12, 14)] + ["loop/%d" % i for i in (1, 3, 4, 6, 14)] + \
       ["fork/%d" % i for i in (0, 1, 3, 6, 10, 11, 12, 13, 15)]


FINE = ["straight-comb/%d" % i for i in (8, 9, 10, 11, 12, 13)] + ["curved-comb/%d" % i for i in (8, 10, 11, 12, 13, 14, 15)] + \
       ["wavy-comb/%d" % i for i in (8, 9, 10, 11, 12, 13)] + ["straight-feather/%d" % i for i in (8, 9, 10, 11, 12, 14, 15)] + \
       ["curved-feather/%d" % i for i in (8, 9, 10, 11, 12, 13, 14, 15)] + \
       ["wavy-feather/%d" % i for i in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)] + ["mesh/%d" % i for i in (5, 8, 10, 12, 15)]


def _mask(ref, px=PX):
    s = SI.silhouette(ref)
    k = px / s["body"]
    P = np.vstack(s["contours"])
    lo = P.min(0) - 8 / k
    W, H = (np.ceil((P.max(0) - lo) * k).astype(int) + 8)
    m = np.zeros((H, W), bool)
    for C in s["contours"]:
        im = Image.new("1", (W, H), 0)
        ImageDraw.Draw(im).polygon([tuple(q) for q in (C - lo) * k], fill=1)
        m ^= np.asarray(im, bool)
    return m, lo, k, s


def part(ref: str, a: float, b: float, px: int = PX, rnd: float = ROUND, fat: float = 0.0) -> dict:
    """The stretch a..b of the piece's spine with what grows from it: outline contours in body widths
    (y up, origin at the stretch's start), its holes and branch tips counted. fat thickens the ink by that
    many body widths, keeping the piece's own shape, so fine teeth can stand beside a heavy stroke."""
    m, lo, k, s = _mask(ref, px)
    sp = (s["spine"] - lo) * k
    sa = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(sp, axis=0).T))])
    u = sa / sa[-1]
    ys, xs = np.nonzero(m)
    _, idx = cKDTree(sp).query(np.stack([xs, ys], 1))
    keep = np.zeros_like(m)
    ok = (u[idx] >= a) & (u[idx] <= b)
    keep[ys[ok], xs[ok]] = True
    lab, n = ndi.label(keep)
    if n > 1:
        mid = sp[np.argmin(np.abs(u - (a + b) / 2))].astype(int)
        L = lab[np.clip(mid[1], 0, m.shape[0] - 1), np.clip(mid[0], 0, m.shape[1] - 1)]
        L = L or 1 + int(np.argmax(ndi.sum(keep, lab, range(1, n + 1))))
        keep = lab == L
    keep = np.pad(keep, 12 + int(fat * px))
    if fat:
        keep = ndi.binary_dilation(keep, ndi.generate_binary_structure(2, 1), iterations=int(fat * px))
    soft = ndi.gaussian_filter(keep.astype(float), rnd * px)
    cs = measure.find_contours(soft, 0.5)
    o = (sp[np.argmin(np.abs(u - a))] + 12 + int(fat * px))
    contours = [np.stack([(c[:, 1] - o[0]), (c[:, 0] - o[1])], 1) / px for c in cs if len(c) > 20]
    solid = soft > 0.5
    sk = skeletonize(solid)
    deg = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    tips = int((sk & (deg == 1)).sum())
    holes = len(contours) - 1
    return {"ref": ref, "a": a, "b": b, "contours": contours, "tips": tips, "holes": holes,
            "ink": float(solid.mean()), "size": np.ptp(np.vstack(contours), axis=0) if contours else np.zeros(2)}


def catalog(out: Path, refs=REFS, span=6.0, fraction=None, px=PX, rnd=0.06, top=2, min_tips=3) -> None:
    """Parts `span` body widths long (or `fraction` of the spine), the most structured of each piece,
    each drawn to fill its cell and labelled with its length in strand widths."""
    found = []
    for ref in refs:
        try:
            s = SI.silhouette(ref)
        except Exception:
            continue
        L = float(np.sum(np.hypot(*np.diff(s["spine"], axis=0).T))) / s["body"]
        step = fraction or span / max(L, 1e-6)
        windows = [(0.0, 1.0)] if step >= 1 else [(x, x + step) for x in np.arange(0, 1 - step + 1e-6, step * 0.6)]
        best = []
        for a, b in windows:
            p = part(ref, a, b, px, rnd)
            if not p["contours"]:
                continue
            p["score"] = (p["tips"] >= min_tips) * min(p["tips"], 8) + 3 * min(p["holes"], 2)
            if p["score"] > 0:
                best.append(p)
        best.sort(key=lambda p: -p["score"])
        found += best[:top]
        print(f"  {ref}: {len(best)}", flush=True)
    found.sort(key=lambda p: -p["score"])
    found = found[:96]
    cell, cols = 186, 12
    rows = (len(found) + cols - 1) // cols
    sheet = Image.new("L", (cell * cols, cell * rows), 255)
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    for k, p in enumerate(found):
        x, y = (k % cols) * cell, (k // cols) * cell
        P = np.vstack(p["contours"])
        c, sz = (P.min(0) + P.max(0)) / 2, np.ptp(P, axis=0)
        unit = (cell - 26) / max(sz)
        m = np.zeros((cell, cell), bool)
        for C in p["contours"]:
            im = Image.new("1", (cell, cell), 0)
            ImageDraw.Draw(im).polygon([(cell / 2 + (q[0] - c[0]) * unit, cell / 2 - 8 - (q[1] - c[1]) * unit) for q in C], fill=1)
            m ^= np.asarray(im, bool)
        sheet.paste(Image.fromarray(~m).convert("L"), (x, y))
        name = p["ref"].replace("curved", "c").replace("straight", "s").replace("wavy", "w").replace("branching-line", "bl")
        d.text((x + 3, y + cell - 14), f"{name} {p['a']:.2f}-{p['b']:.2f} L{max(sz):.0f}", font=f, fill=90)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out} ({len(found)} parts)", flush=True)


if __name__ == "__main__":
    catalog(HERE / "proofs" / "parts" / "catalog-large.png", REFS, span=6.5, px=30)
    catalog(HERE / "proofs" / "parts" / "catalog-fine.png", FINE, fraction=0.3, px=14, rnd=0.1, min_tips=5)
