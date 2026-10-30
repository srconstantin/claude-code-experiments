"""
Neuron Club — Uncial's skeletons with thickness taken from strokelib rather than from a pen.

A broad pen makes thickness depend on direction, which is what reads as medieval. The library's strokes vary along
their length instead. Here every stroke takes a real tapered stroke's measured width profile (exaggerated for display
sizes): it swells toward one end and runs out to a fine point. The heavy end sits near a join, but narrows as it
arrives, so joins stay open; a stroke free at both ends is heavy low down. One profile each for stems and curves
keeps the letters related.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curated as CU        # noqa: E402
import formal as FM         # noqa: E402
import fuse as FU           # noqa: E402
import outline as OL        # noqa: E402
import silhouettes as SI    # noqa: E402

FAMILY = "strokelib:family-08-short-tapered-straight-branch-fine"
STEM, CURVE = f"{FAMILY}/7", f"{FAMILY}/2"
SWELL = 2.0                 # exponent on the measured width profile
HEAVY = 1.32                # thickest part of a stroke, in weights
FLOOR = 0.34                # no line thinner than this, in weights
BAR = 0.52                  # … and no short bar or arm thinner than this
TIP = 0.14                  # tip width, as a share of the stroke there
TAPER = 1.7                 # tips form over this length, in weights
NECK, NECK_LEN = 0.55, 1.5  # a stroke narrows to this share over this length (weights) as it arrives at a join
OVER = 12                   # every letter overshoots its lines: its ends are points or curves
PLAIN = set("2xX")          # these read better on the plain skeleton here
SEATED = set("TYLF")        # stems that end on another stroke rather than in a point
TRIM = {"a": (0, 0.1), "c": (0, 0.13), "t": (0, 0.14)}   # stroke, share cut from its lower end: curled feet go
DOT_DROP = 0.35             # … and it sits this much (weights) lower than the skeleton's dot
BOWLS = {"p": 1, "b": 1, "q": 0, "d": 0, "a": 0, "g": 0}   # which stroke is the bowl
TUCK = 0.1                  # share cut from a bowl's tucked inner end
DOT = 0.74                  # radius of the dot of i and j, in weights
FILLET = 0.1                # inside corners narrower than this (weights) are rounded, so joins leave no nicks
SPECK = 0.12                # contours smaller than this (square weights) are slivers


@lru_cache(maxsize=None)
def profile(ref):
    """The stroke's width along its length, thickest = 1, heaviest end last."""
    p = SI.silhouette(ref)["profile"]
    p = np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, len(p)), p)
    p = ndi.gaussian_filter1d(p, 3, mode="nearest") / p.max()
    return p if p[-12:].mean() >= p[:12].mean() else p[::-1]


def stroke(s, W, others, seated=False):
    q = FM._centre(s, W)
    if s.closed:                                 # the library's loops: heavier on one side
        tg = OL.tangents(q, True)
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        a = np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean())
        w = W * (0.68 + 0.5 * np.cos(a + 2.4))
        return [[q + nr * (w / 2)[:, None], (q - nr * (w / 2)[:, None])[::-1]]]
    d = q[-1] - q[0]
    straight = s.kind == "crescent"
    upright = straight and abs(d[1]) > 0.8 * np.hypot(*d)
    L = OL.arclen(q)
    near = lambda e: bool(others) and min(float(np.min(np.hypot(*(o - e).T))) for o in others) < 0.6 * W
    join0, join1 = (near(q[0]), near(q[-1])) if upright and seated else \
        (False, False) if upright else (s.taper0 or near(q[0]), s.taper1 or near(q[-1]))
    p = np.interp(L / L[-1], np.linspace(0, 1, 60), profile(STEM if straight else CURVE)) ** SWELL
    if upright and seated:                       # a stem still swells low down, and stays firm at its seat
        p = np.maximum(p[::-1] if q[0, 1] < q[-1, 1] else p, 0.62)
    elif join0 and join1:                        # held at both ends: heaviest in the middle
        p = 0.6 + 0.4 * np.sin(np.pi * L / L[-1])
    elif join0 or (not join1 and q[0, 1] < q[-1, 1]):
        p = p[::-1]
    short = getattr(s, "free", False) or (straight and not upright and s.w < 1)
    w = W * HEAVY * s.w * np.maximum(p, (BAR if short else FLOOR) / HEAVY)
    for x, joined in ((L, join0), (L[-1] - L, join1)):
        if joined:
            n = 0.85 if straight else NECK       # a bar stays firm where it leaves a stem
            w = w * (n + (1 - n) * FM._ease(x / (NECK_LEN * W)))
        else:
            w = w * (TIP + (1 - TIP) * FM._ease(x / (TAPER * W)))
    tg = OL.tangents(q, False)
    nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
    return [np.vstack([q + nr * (w / 2)[:, None], (q - nr * (w / 2)[:, None])[::-1]])]


def dot(d) -> np.ndarray:
    """The dot of i and j: a soft, lopsided egg, heavier low on the left like the strokes."""
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    r = DOT * CU.GL.W_LOW * (1 + 0.13 * np.cos(t + 0.8) + 0.07 * np.cos(2 * t - 0.6) + 0.03 * np.cos(3 * t + 2.0))
    return np.stack([d.x + r * np.cos(t), d.y - DOT_DROP * CU.GL.W_LOW + r * np.sin(t)], 1)


def glyph(ch: str, W: float) -> list[np.ndarray]:
    S = FM._strokes(ch, W, False, PLAIN)
    if ch in TRIM:
        s, cut = S[TRIM[ch][0]], TRIM[ch][1]
        n = int(cut * len(s.P0))
        s.P0 = s.P0[n:] if s.P0[0, 1] < s.P0[-1, 1] else s.P0[:-n]
        s.prepare(W)
    if ch in BOWLS:                              # the bowl's tucked end stops short of the stem
        s = S[BOWLS[ch]]
        n = int(TUCK * len(s.P0))
        m = s.P0.mean(0)
        inner_last = np.hypot(*(s.P0[-1] - m)) < np.hypot(*(s.P0[0] - m))
        s.P0 = s.P0[:-n] if inner_last else s.P0[n:]
        s.prepare(W)
    if ch == "4":                                # the diagonal turns into the bar in one stroke, leaving no corner block
        diag, bar = S[1], S[2]
        S = [S[0], OL.Stroke(np.vstack([diag.P0[::-1], bar.P0[1:]]))]
    if ch == "F":                                # a top bar long enough to balance the stem
        b = S[1]
        b.P0 = np.stack([b.P0[:, 0].min() + (b.P0[:, 0] - b.P0[:, 0].min()) * 1.25, b.P0[:, 1]], 1)
        b.prepare(W)
    if ch in FM.NO_SPUR:
        S = [s for s in S if not getattr(s, "free", False)]
    OL.outline(S, W)
    if ch in "Kk":                               # the curve stops at the stem; its arms open a little wider
        a = S[1]
        x0 = float(a.c[:, 0].min())
        a.P0 = np.stack([x0 + (a.P0[:, 0] - x0) * 1.14, a.P0[:, 1]], 1)
        a.prepare(W)
        a.P0 = a.P0 + [float(S[0].c[:, 0].mean()) - float(a.c[:, 0].min()), 0]
        a.prepare(W)
    lines = [s.c for s in S if isinstance(s, OL.Stroke)]
    polys, body = [], []
    for s in S:
        if isinstance(s, OL.Dot):
            polys.append(dot(s))
        else:
            p = stroke(s, W, [c for c in lines if c is not s.c], ch in SEATED)
            polys += p
            body += p
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    cs = [C for C in OL.union(polys) if abs(area(C)) > SPECK * W ** 2]
    holes = [C for C in cs if area(C) < 0]
    cs = FU.raster_glyph([], [], [(cs, [])], [], W, counters=holes, inset=0.0, smooth=0.8, fillet=FILLET,
                         contour_smooth=0.015 * W)
    cs = [C for C in cs if abs(area(C)) > SPECK * W ** 2]
    if ch in "69":                               # only the main counter: where the spiral meets itself is ink
        holes = [C for C in cs if area(C) < 0]
        cs = [C for C in cs if area(C) > 0 or C is max(holes, key=lambda h: -area(h))]
    B = np.vstack(OL.union(body))
    top = CU.TOPS.get(ch, CU.GL.CAP) + OVER
    bot = CU.BOTTOMS.get(ch, 0) - OVER
    k = (top - bot) / (B[:, 1].max() - B[:, 1].min())
    return [np.stack([C[:, 0], bot + (C[:, 1] - B[:, 1].min()) * k], 1) for C in cs]


_WEIGHTS: dict = {}


def weight(ch: str, main_stroke) -> float:
    """Every letter's main strokes brought toward its reference letter's (n, or H for capitals and figures)."""
    if ch not in _WEIGHTS:
        base = CU.GL.W_LOW if ch.islower() else CU.GL.W_CAP
        ref = main_stroke(glyph("n" if ch.islower() else "H", base))
        f = ref / main_stroke(glyph(ch, base))
        _WEIGHTS[ch] = base * float(np.clip(f, 0.8, 1.25)) ** 0.85
    return _WEIGHTS[ch]
