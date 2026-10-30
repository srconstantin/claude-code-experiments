"""
Neuron Formal — Curated's skeletons, redrawn with the manners of formal lettering.

What read as casual in Curated was the drawing, not the forms: round marker ends,
even weight, wobbling stems, spurs and round dots. Here the same centrelines are:

  settled    smoothed, with stems and bars truly straight
  penned     given thick and thin by one pen angle throughout (a broad nib)
  finished   their free ends tapered to a point
  dotted     with a lozenge for the dots of i and j

STYLES holds the treatments being compared.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curated as CU        # noqa: E402
import outline as OL        # noqa: E402

STEP = 6.0
SETTLE = 0.45               # centrelines are smoothed over this length, in weights
NO_SPUR = set("nmrhuNMU")   # these lose the arch's spur
PLAIN = set("35")           # these take the plain skeleton: a 5 needs its flat top
SPECK = 0.12                # contours smaller than this (in square weights) are slivers, not counters

# pen angle (degrees), thinnest stroke (share of the thickest), weight, the finest any line may get (in weights,
# away from its tips), and how fine a tip ends
STYLES = {"uncial": dict(pen=32, thin=0.2, k=1.12, floor=0.3, tip=0.16)}


def _centre(s, W):
    c = np.vstack([s.c, s.c[:1]]) if s.closed else s.c
    if s.kind == "crescent":
        c = np.linspace(c[0], c[-1], 40)
    sa = OL.arclen(c)
    t = np.arange(0, sa[-1], STEP)
    q = np.stack([np.interp(t, sa, c[:, 0]), np.interp(t, sa, c[:, 1])], 1)
    if not s.closed and len(q) > 8 and s.kind != "crescent":
        sig = SETTLE * W / STEP
        Q = ndi.gaussian_filter1d(q, sig, axis=0, mode="nearest")
        n = len(q)
        h = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / (2 * sig), 0, 1)[:, None]
        q = h * Q + (1 - h) * q
    elif s.closed:
        q = ndi.gaussian_filter1d(q, SETTLE * W / STEP, axis=0, mode="wrap")
    return np.vstack([q, c[-1]]) if not s.closed else q


def _stroke(s, W, st):
    q = _centre(s, W)
    tg = OL.tangents(q, s.closed)
    th = np.arctan2(tg[:, 1], tg[:, 0])
    pen = np.radians(st["pen"])
    full = abs(np.sin(np.pi / 2 - pen))
    w = W * st["k"] * max(s.w, 0.9) * (st["thin"] + (1 - st["thin"]) * np.abs(np.sin(th - pen)) / full)
    mode = "wrap" if s.closed else "nearest"
    w = ndi.gaussian_filter1d(w, 0.4 * W / STEP, mode=mode)
    w = np.maximum(w, st["floor"] * W)                # no line goes hairline
    n = np.stack([-tg[:, 1], tg[:, 0]], 1)
    if s.closed:
        return [[q + n * (w / 2)[:, None], (q - n * (w / 2)[:, None])[::-1]]]
    sa = OL.arclen(q)
    d = {0: sa, 1: sa[-1] - sa}
    for end, joined in ((0, s.taper0), (1, s.taper1)):
        x = d[end]
        if joined:                                  # a curve tucks into the stroke it joins; a bar stays firm
            neck = 0.62 if s.kind == "crescent" else 0.25
            w = w * (neck + (1 - neck) * np.clip(x / (1.4 * W), 0, 1))
        else:                                       # bars end quickly: a long taper would swallow a short bar
            flat = s.kind == "crescent" and abs(tg[len(tg) // 2][0]) > 0.8
            w = w * (st["tip"] + (1 - st["tip"]) * np.clip(x / ((0.8 if flat else 1.8) * W), 0, 1) ** 0.75)
    if s.kind == "crescent" and sa[-1] > 5 * W:     # entasis: a stem is a touch narrower at the waist
        w = w * (1 - 0.07 * np.sin(np.pi * sa / sa[-1]) ** 2)
    return [np.vstack([q + n * (w / 2)[:, None], (q - n * (w / 2)[:, None])[::-1]])]


def _lozenge(d, W):
    r = 0.72 * W * 1.0
    return np.array([[d.x, d.y - r], [d.x + 0.8 * r, d.y], [d.x, d.y + r], [d.x - 0.8 * r, d.y]])


def _strokes(ch, W):
    if ch not in PLAIN:
        S = CU._fitted(ch, W)
    else:                                       # a plain skeleton reads better than the piece here
        S = copy.deepcopy(CU.GL.GLYPHS[ch])
        CU.DR.dress(ch, S, W, None, None)
        CU.fit(ch, S, W)
    if ch == "f":                               # a crossbar long enough to read
        b = S[1]
        b.P0 = b.P0.mean(0) + (b.P0 - b.P0.mean(0)) * [1.5, 1]
        b.prepare(W)
    return S


def glyph(ch: str, W: float, style: str) -> list[np.ndarray]:
    st = STYLES[style]
    S = _strokes(ch, W)
    if ch in NO_SPUR:
        S = [s for s in S if not getattr(s, "free", False)]
    OL.outline(S, W)
    polys, body = [], []
    for s in S:
        p = [_lozenge(s, W)] if isinstance(s, OL.Dot) else _stroke(s, W, st)
        polys += p
        if not getattr(s, "free", False) and not isinstance(s, OL.Dot):
            body += p
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    cs = [C for C in OL.union(polys) if abs(area(C)) > SPECK * W ** 2]
    B = np.vstack(OL.union(body))
    top = CU.TOPS.get(ch, OL_CAP()) + (CU.OVER if ch in CU.TOP_ROUND else 0)
    bot = CU.BOTTOMS.get(ch, 0) - (CU.OVER if ch in CU.BOT_ROUND else 0)
    k = (top - bot) / (B[:, 1].max() - B[:, 1].min())
    return [np.stack([C[:, 0], bot + (C[:, 1] - B[:, 1].min()) * k], 1) for C in cs]


def OL_CAP():
    return CU.GL.CAP
