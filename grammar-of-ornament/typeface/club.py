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
import ornate as OR         # noqa: E402
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
PLAIN = set("2xXMN")        # these read better on the plain skeleton here (M and N as true capitals)
CLOSED = set("PR")          # with an arc on top these need a closed bowl: the arc sits badly on an open spiral
SEATED = set("TYLFDMN")     # stems that end on another stroke rather than in a point
TRIM = {"a": (0, 0.1), "c": (0, 0.13), "t": (0, 0.14)}   # stroke, share cut from its lower end: curled feet go
DOT_DROP = 0.35             # … and it sits this much (weights) lower than the skeleton's dot
BOWLS = {"p": 1, "b": 1, "q": 0, "d": 0, "a": 0, "g": 0}   # which stroke is the bowl
BARS = {"f": 1.3, "t": 1.7}  # crossbars lengthened by this much
E_DEPTH = 0.45              # how much further the E's curve bows, in weights
SLIVERS = set("469")
CLASS = {"x": 0.92, "H": 1.04}   # lowercase a little lighter than capitals, as it should be
PUNCT_SB = 45               # sidebearing of punctuation
TUCK = 0.1                  # share cut from a bowl's tucked inner end
DOT = 0.6                   # radius of the dot of i and j, in weights
# Club Ornate: on capitals only. The doubled arc on rounds; on stems and diagonals, fine branched fragments kept to
# the thin parts of the strokes, so they don't pile onto the swell, and often a second one lower on the other side.
ARC = OR.ARC
WING = ("wavy-feather/6", 0.0, 0.30, False, 0.15)
ROUND = [(ARC, 0, "top", 0.8)]
BOWL = [(ARC, 1, "top", 0.8)]
SWEEP = ("curved-comb/14", 0.36, 0.66, False, 0.15)      # a sweeping frond, its strands doubled
COMB = ("straight-comb/12", 0.36, 0.66, False, 0.15)
FROND = [(SWEEP, 0, "upper", 1.0, 4.2)]
HIGH = [(WING, 0, "upper", 1.0, 4.0)]
ORNAMENT = {"A": [(WING, 0, "left", 1.0, 4.0)], "B": FROND, "C": ROUND, "D": BOWL, "E": [(WING, 0, (0.3, 0.7), 0.75)],
            "F": HIGH, "G": ROUND, "H": FROND, "I": FROND, "J": [(WING, 0, "upper", 1.15, 4.4)], "K": HIGH, "L": FROND,
            "M": FROND, "N": FROND, "O": ROUND, "P": BOWL, "Q": ROUND, "R": BOWL, "S": [(ARC, 0, "top", 0.8, 3.6)],
            "U": HIGH, "V": HIGH, "W": HIGH, "X": [(WING, 1, "upper", 0.9, 3.4)],
            "Y": [(WING, 0, "upper", 1.0, 3.8)], "Z": [(COMB, 1, "upper", 0.85, 3.4)]}
ASCENT = (WING, 0, "upper", 0.7, 3.0)       # small wings on the lowercase ascenders
ORNAMENT.update({"b": [ASCENT], "h": [ASCENT], "k": [ASCENT], "l": [ASCENT], "d": [(WING, 1, (0.62, 0.97), 0.7)]})
ARC_TOP = 105               # every arc tops out this far above its letter's top line
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
    if getattr(s, "top_heavy", False):
        p = p[::-1]
    short = getattr(s, "free", False) or (straight and not upright and s.w < 1)
    w = W * HEAVY * s.w * np.maximum(p, (BAR if short else FLOOR) / HEAVY)
    for x, joined in ((L, join0), (L[-1] - L, join1)):
        if joined:
            n = 0.85 if straight else NECK       # a bar stays firm where it leaves a stem
            w = w * (n + (1 - n) * FM._ease(x / (NECK_LEN * W)))
        else:
            w = w * (TIP + (1 - TIP) * FM._ease(x / ((0.8 if short else TAPER) * W)))    # bars end quickly
    tg = OL.tangents(q, False)
    nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
    return [np.vstack([q + nr * (w / 2)[:, None], OL._cap(q[-1], tg[-1], w[-1] / 2),      # every end is round, or a point
                       (q - nr * (w / 2)[:, None])[::-1], OL._cap(q[0], -tg[0], w[0] / 2)])]


def dot(d) -> np.ndarray:
    """The dot of i and j: a soft, lopsided egg, heavier low on the left like the strokes."""
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    r = DOT * CU.GL.W_LOW * (1 + 0.13 * np.cos(t + 0.8) + 0.07 * np.cos(2 * t - 0.6) + 0.03 * np.cos(3 * t + 2.0))
    return np.stack([d.x + r * np.cos(t), d.y - DOT_DROP * CU.GL.W_LOW + r * np.sin(t)], 1)


def _punct():
    C, K, D = CU.GL.C, CU.GL.K, OL.Dot
    def line(*pts, **kw):
        return OL.Stroke(np.array(pts, float), **kw)
    tail = lambda x: line((x + 22, 75), (x + 16, -30), (x - 42, -125), w=0.62)
    bang = C((110, 215), (110, 730))
    bang.top_heavy = True
    amp = [line((535, 15), (345, 235), (185, 430), (165, 590), (270, 695), w=0.92),      # two strokes: one would cross itself
           line((270, 695), (365, 605), (335, 480), (195, 360), (75, 205), (135, 45), (280, -5), (430, 110), (505, 275),
                w=0.92)]
    q = K([(85, 545), (165, 680), (305, 700), (395, 585), (325, 455), (225, 365), (215, 225)])
    arc = [(265, 770), (125, 560), (85, 250), (125, -60), (265, -240)]
    return {".": [D(100, 105)], ",": [D(105, 105), tail(105)], ":": [D(100, 105), D(100, 455)],
            ";": [D(105, 105), tail(105), D(105, 455)], "!": [bang, D(110, 105)], "?": [q, D(215, 105)],
            "'": [C((85, 725), (75, 490), w=0.62)], '"': [C((85, 725), (75, 490), w=0.62), C((210, 725), (200, 490), w=0.62)],
            "-": [C((50, 265), (300, 265), w=0.8)], "\u2013": [C((50, 265), (520, 265), w=0.8)],
            "\u2014": [C((50, 265), (920, 265), w=0.8)], "/": [C((40, -80), (410, 770), w=0.85)],
            "(": [line(*arc)], ")": [line(*[(300 - x, y) for x, y in arc])], "&": amp}


EXTRA = _punct()


def punct(ch: str, W: float) -> list[np.ndarray]:
    S = EXTRA[ch]
    OL.outline(S, W)
    lines = [s.c for s in S if isinstance(s, OL.Stroke)]
    polys = []
    for s in S:
        polys += [dot(s)] if isinstance(s, OL.Dot) else stroke(s, W, [c for c in lines if c is not s.c])
    return OL.union(polys)


def _piece(ch, S, W):
    """The letter's fragments, on these strokes, under the shared ceiling."""
    for s in S:
        if isinstance(s, OL.Stroke) and s.kind == "crescent":
            s.c = FM._centre(s, W)              # the fragment follows the settled stroke
    out, line = [], CU.TOPS.get(ch, CU.GL.CAP)
    for spec in ORNAMENT.get(ch, []):
        piece = OR.fragment(ch, S, W, {ch: spec}, FM.SEAT)[1]
        if spec[0] == ARC:                      # all arcs reach one height, so a title's top edge is even
            top = line + ARC_TOP
            for _ in range(3):
                over = float(np.vstack(piece)[:, 1].max())
                spec = (*spec[:3], spec[3] * (top - line + 0.5 * W) / (over - line + 0.5 * W), *spec[4:])
                piece = OR.fragment(ch, S, W, {ch: spec}, FM.SEAT)[1]
        out.append(piece)
    return out


def glyph(ch: str, W: float, ornament: bool = False) -> list[np.ndarray]:
    if ch in EXTRA:
        return punct(ch, W)
    S = FM._strokes(ch, W, False, PLAIN | CLOSED if ornament else PLAIN)
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
        S = [S[0], OL.Stroke(np.vstack([diag.P0[::-1], bar.P0[1:]]), w=0.68)]    # slim, so its counter stays open
    if ch in BARS:                               # crossbars long enough to read
        b = S[1]
        b.P0 = b.P0.mean(0) + (b.P0 - b.P0.mean(0)) * [BARS[ch], 1]
        b.prepare(W)
    if ornament and ch == "E":                   # a deeper curve: room for the frond, and the arms sweep clear of it
        ys = [s.P0[:, 1] for s in S]
        lo, hi = min(y.min() for y in ys), max(y.max() for y in ys)
        for s in S:
            s.P0 = s.P0 - np.stack([E_DEPTH * W * np.sin(np.pi * (s.P0[:, 1] - lo) / (hi - lo)) ** 1.2,
                                    np.zeros(len(s.P0))], 1)
            s.prepare(W)
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
    grafts = _piece(ch, S, W) if ornament else []
    keep = 0.02 if grafts else SPECK            # a fragment's own small holes stay
    lines = [s.c for s in S if isinstance(s, OL.Stroke)]
    for d in S:                                  # a dot sits centred over its stem
        if isinstance(d, OL.Dot):
            d.x = float(lines[0][np.argmax(lines[0][:, 1]), 0])
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
    tight = ch in SLIVERS                        # white slits running off these counters are closed too
    cs = FU.raster_glyph([], [], [(cs, [])], [], W, counters=[] if tight else holes, inset=0.0, smooth=0.8,
                         fillet=(0.32 if ch == "6" else 0.2) if tight else FILLET, contour_smooth=0.015 * W)
    cs = [C for C in cs if abs(area(C)) > SPECK * W ** 2]
    if grafts:                                   # fragments join after the rounding, so their teeth stay open
        cs = [C for C in OL.union([cs], grafts) if abs(area(C)) > keep * W ** 2]
    if ch in "469":                              # only the main counter: where the spiral meets itself is ink
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
    if ch in EXTRA:
        return CU.GL.W_LOW * CLASS["x"]
    if ch not in _WEIGHTS:
        base = CU.GL.W_LOW if ch.islower() else CU.GL.W_CAP
        ref = main_stroke(glyph("n" if ch.islower() else "H", base))
        f = ref / main_stroke(glyph(ch, base))
        k = float(np.clip(f, 0.72, 1.3)) ** (1.0 if ch.isdigit() else 0.85)      # figures are evened fully
        if ch in SLIVERS:                        # their measured stroke reads thin, but heavier closes their counters
            k = min(k, 0.9 if ch == "9" else 1.0)
        _WEIGHTS[ch] = base * k * CLASS["x" if ch.islower() else "H"]
    return _WEIGHTS[ch]
