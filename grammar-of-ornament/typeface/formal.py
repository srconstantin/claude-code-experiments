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
import ornate as OR         # noqa: E402
import outline as OL        # noqa: E402

# Illuminated: Uncial with two kinds of real fragment, kept mostly to the capitals: the doubled arc on rounds and
# arches, and one comb, thickened so its teeth have body beside the stroke. Lowercase and figures stay plain, but
# the descending hooks roll into curved-sprig/13's open spiral.
BOLD = ("wavy-feather/6", 0.0, 0.30, False, 0.3)
ARC = OR.ARC
ILLUMINATED = {c: None for c in CU.GL.GLYPHS}
ILLUMINATED.update({
    "C": (ARC, 0, "top", 0.8), "D": (ARC, 1, "top", 0.8), "G": (ARC, 0, "top", 0.8), "M": (ARC, 0, "top", 0.8),
    "N": (ARC, 0, "top", 0.8), "O": (ARC, 0, "top", 0.8), "P": (ARC, 1, "top", 0.8), "Q": (ARC, 0, "top", 0.8),
    "R": (ARC, 1, "top", 0.8), "S": (ARC, 0, "top", 0.8, 3.6), "T": (ARC, 1, "top", 0.8),
    "A": (BOLD, 0, "left", 0.7), "B": (BOLD, 0, (0.12, 0.88), 0.7), "E": (BOLD, 0, "left", 0.7), "F": (BOLD, 0, (0.12, 0.8), 0.7),
    "H": (BOLD, 0, (0.12, 0.88), 0.7), "K": (BOLD, 0, (0.12, 0.88), 0.7), "L": (BOLD, 0, (0.15, 0.9), 0.7),
    "U": (BOLD, 0, "left", 0.7), "Z": (BOLD, 0, "top", 0.6)})
SPIRAL = ("curved-sprig/13", 0.56)          # the piece, and where along its spine its spiral is taken from
# hook → (steps carried out along the hook before it rolls up, length of the rolled end in weights)
ROLL = {"j": (3, 6.2), "J": (3, 6.2), "g": (3, 6.2), "y": (3, 6.2)}
DEEP = {"g": -420, "y": -420}               # these run deeper, to give the spiral room below the bowl
SEAT = 0.34                                 # share of an arc's length, at each end, that folds into the stroke
CEILING = 130               # no fragment rises more than this above its letter's top line
STEP = 6.0
SETTLE = 0.45               # centrelines are smoothed over this length, in weights
NO_SPUR = set("nmrhuNMU")   # these lose the arch's spur
PLAIN = set("35D")          # these take the plain skeleton: a 5 needs its flat top, a D a closed bowl
HOOKS = {"g": 1, "y": 1, "j": 0, "J": 0}    # which stroke is the descending hook
SPECK = 0.12                # contours smaller than this (in square weights) are slivers, not counters

# pen angle (degrees), thinnest stroke (share of the thickest), weight, the finest any line may get (in weights,
# away from its tips), and how fine a tip ends
STYLES = {"uncial": dict(pen=32, thin=0.2, k=1.12, floor=0.3, tip=0.16)}


def _ease(x):
    """0 at the end of a stroke, rising to 1 with no corner where it meets the stroke's body."""
    return np.sin(0.5 * np.pi * np.clip(x, 0, 1)) ** 0.85


def _centre(s, W):
    c = np.vstack([s.c, s.c[:1]]) if s.closed else s.c
    if s.kind == "crescent":                        # straightened, but staying where its joins were placed
        c = np.linspace(c[0], c[-1], 40) + (c.mean(0) - (c[0] + c[-1]) / 2)
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


def _stroke(s, W, st, others=()):
    q = _centre(s, W)
    tg = OL.tangents(q, s.closed)
    th = np.arctan2(tg[:, 1], tg[:, 0])
    pen = np.radians(st["pen"])
    full = abs(np.sin(np.pi / 2 - pen))
    w = W * st["k"] * max(s.w, 0.9) * (st["thin"] + (1 - st["thin"]) * np.abs(np.sin(th - pen)) / full)
    mode = "wrap" if s.closed else "nearest"
    w = ndi.gaussian_filter1d(w, 0.4 * W / STEP, mode=mode)
    w = ndi.gaussian_filter1d(np.maximum(w, st["floor"] * W), 0.3 * W / STEP, mode=mode)     # no line goes hairline
    n = np.stack([-tg[:, 1], tg[:, 0]], 1)
    if s.closed:
        return [[q + n * (w / 2)[:, None], (q - n * (w / 2)[:, None])[::-1]]]
    sa = OL.arclen(q)
    d = {0: sa, 1: sa[-1] - sa}
    upright = s.kind == "crescent" and abs(tg[len(tg) // 2][1]) > 0.8
    for end, joined in ((0, s.taper0), (1, s.taper1)):
        x = d[end]
        if upright:                                 # a stem always finishes in a point, even where a curve meets it
            joined = False
        elif others and min(float(np.min(np.hypot(*(o - q[-end]).T))) for o in others) < 0.6 * W:
            joined = True                           # a curve tucks in wherever it lands on another stroke
        if joined:                                  # a curve tucks into the stroke it joins; a bar stays firm
            neck = 0.62 if s.kind == "crescent" else 0.25
            w = w * (neck + (1 - neck) * _ease(x / (1.4 * W)))
        else:                                       # bars end quickly: a long taper would swallow a short bar
            flat = s.kind == "crescent" and abs(tg[len(tg) // 2][0]) > 0.8
            w = w * (st["tip"] + (1 - st["tip"]) * _ease(x / ((1.15 if flat else st.get("taper", 2.4)) * W)))
    if s.kind == "crescent" and sa[-1] > 5 * W:     # entasis: a stem is a touch narrower at the waist
        w = w * (1 - 0.07 * np.sin(np.pi * sa / sa[-1]) ** 2)
    return [np.vstack([q + n * (w / 2)[:, None], (q - n * (w / 2)[:, None])[::-1]])]


def _lozenge(d, W):
    r = 0.72 * W * 1.0
    return np.array([[d.x, d.y - r], [d.x + 0.8 * r, d.y], [d.x, d.y + r], [d.x - 0.8 * r, d.y]])


def _strokes(ch, W, ornament=False, plain=()):
    if ch not in PLAIN and ch not in plain:
        S = CU._fitted(ch, W)
    else:                                       # a plain skeleton reads better than the piece here
        S = copy.deepcopy(CU.GL.GLYPHS[ch])
        CU.DR.dress(ch, S, W, None, None)
        CU.fit(ch, S, W)
    if ch in HOOKS:                             # the descender's hook carried on round, and a little back up
        s = S[HOOKS[ch]]
        if ornament and ch in DEEP:
            k = (60 - DEEP[ch] - 150) / (60 - float(s.P0[:, 1].min()))
            s.P0 = np.stack([s.P0[:, 0], np.where(s.P0[:, 1] < 60, 60 + (s.P0[:, 1] - 60) * k, s.P0[:, 1])], 1)
        low_first = s.P0[0, 1] < s.P0[-1, 1]
        q = s.P0[::-1] if low_first else s.P0
        h = q[-1] - q[-5]
        a = np.arctan2(h[1], h[0])
        ext = [q[-1]]
        for _ in range(ROLL[ch][0] if ornament else 9):
            a -= np.radians(7)
            ext.append(ext[-1] + 0.34 * W * np.array([np.cos(a), np.sin(a)]))
        if ornament:                            # on into the real spiral, rolling the same way
            sp = CU.skel(SPIRAL[0])["spine"]
            z = (sp[:, 0] + 1j * sp[:, 1])[int(SPIRAL[1] * len(sp)):]
            if np.sign(np.unwrap(np.angle(np.diff(z)))[-1] - np.angle(z[1] - z[0])) > 0:
                z = np.conj(z)
            m = np.exp(1j * a) / ((z[2] - z[0]) / abs(z[2] - z[0])) * ROLL[ch][1] * W / np.sum(np.abs(np.diff(z)))
            r = complex(*ext[-1]) + m * (z - z[0])
            ext += list(np.stack([r.real, r.imag], 1)[1:])
        q = np.vstack([q, ext[1:]])
        s.P0 = q[::-1] if low_first else q
        s.prepare(W)
    if ch == "f":                               # a crossbar long enough to read
        b = S[1]
        b.P0 = b.P0.mean(0) + (b.P0 - b.P0.mean(0)) * [1.5, 1]
        b.prepare(W)
    return S


def glyph(ch: str, W: float, style: str, ornament: bool = False) -> list[np.ndarray]:
    st = STYLES[style]
    S = _strokes(ch, W, ornament)
    if ch in NO_SPUR:
        S = [s for s in S if not getattr(s, "free", False)]
    OL.outline(S, W)
    if ch in "Kk":                              # the curve stops at the stem; its arms open a little wider
        a = S[1]
        x0 = float(a.c[:, 0].min())
        a.P0 = np.stack([x0 + (a.P0[:, 0] - x0) * 1.14, a.P0[:, 1]], 1)
        a.prepare(W)
        a.P0 = a.P0 + [float(S[0].c[:, 0].mean()) - float(a.c[:, 0].min()), 0]
        a.prepare(W)
    grafts = []
    if ornament:                                # the fragment follows the settled stroke, not the raw one
        for s in S:
            if isinstance(s, OL.Stroke) and s.kind == "crescent":
                s.c = _centre(s, W)
        piece = OR.fragment(ch, S, W, ILLUMINATED, SEAT)[1]
        if piece:
            top, spec = CU.TOPS.get(ch, CU.GL.CAP) + CEILING, ILLUMINATED[ch]
            over = float(np.vstack(piece)[:, 1].max())
            if over > top:                      # brought under the shared ceiling
                base = CU.TOPS.get(ch, CU.GL.CAP) - 0.5 * W
                k = (top - base) / (over - base)
                piece = OR.fragment(ch, S, W, {ch: (*spec[:3], spec[3] * k, *spec[4:])}, SEAT)[1]
            grafts = [piece]
    lines = [s.c for s in S if isinstance(s, OL.Stroke)]
    polys, body = [], []
    for s in S:
        rest = [c for c in lines if c is not getattr(s, "c", None)]
        firm = ch in "Ee" and getattr(s, "free", False) or ch in "Kk" and s is S[1]
        p = [_lozenge(s, W)] if isinstance(s, OL.Dot) else _stroke(s, W, dict(st, floor=0.52, taper=1.0) if firm else st, rest)
        polys += p
        if not getattr(s, "free", False) and not isinstance(s, OL.Dot):
            body += p
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    cs = [C for C in OL.union(polys, grafts) if abs(area(C)) > (0.02 if ornament else SPECK) * W ** 2]
    if ch in "69" and not ornament:             # only the main counter: where the spiral meets itself is ink
        holes = [C for C in cs if area(C) < 0]
        cs = [C for C in cs if area(C) > 0 or C is max(holes, key=lambda h: -area(h))]
    B = np.vstack(OL.union(body))
    top = CU.TOPS.get(ch, OL_CAP()) + (CU.OVER if ch in CU.TOP_ROUND else 0)
    bot = (DEEP.get(ch) if ornament and ch in DEEP else CU.BOTTOMS.get(ch, 0)) - (CU.OVER if ch in CU.BOT_ROUND else 0)
    k = (top - bot) / (B[:, 1].max() - B[:, 1].min())
    return [np.stack([C[:, 0], bot + (C[:, 1] - B[:, 1].min()) * k], 1) for C in cs]


def OL_CAP():
    return CU.GL.CAP
