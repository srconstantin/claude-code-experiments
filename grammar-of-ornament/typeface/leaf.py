"""
Neuron Leaf — Uncial's skeletons drawn as leaves rather than with a pen.

Every stroke swells in its middle and runs out to a fine point at each free end, like the leaf-shaped strokes of
strokelib; where it meets another stroke it narrows, so joins stay open. On top of that swelling, strokes running
across are thinner than strokes running up and down, so a word has calligraphic contrast without a pen's angle.
The proportions are long: tall ascenders, deep descenders, ascenders rising above the capitals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curated as CU        # noqa: E402
import formal as FM         # noqa: E402
import fuse as FU           # noqa: E402
import ornate as OR         # noqa: E402
import outline as OL        # noqa: E402

LOW = 0.1                   # a stroke's ends, before tapering, as a share of its swell
SWELL = 1.7                 # exponent of the swell along a stroke (higher: fuller middle, longer fine ends)
HEAVY = 1.5                 # thickest part of a stroke, in weights
ACROSS = 0.42               # strokes running across are this much thinner than strokes running up and down
FLOOR = 0.17                # no line thinner than this, in weights
BAR = 0.36                  # … and no bar thinner than this
TIP = 0.1                   # tip width, as a share of the stroke there
TAPER = 2.4                 # tips form over this length, in weights
NECK, NECK_LEN = 0.45, 1.5  # a stroke narrows to this share over this length (weights) as it arrives at a join
OVER = 12                   # every letter overshoots its lines: its ends are points or curves
ASC, DESC = 840, -320       # long ascenders (above the capitals) and descenders
TALL = 1.35                 # ascenders and t's stem stretched by this much above the x-height
DEEP = 1.22                 # descenders stretched by this much below the baseline
TOPS = {**{c: ASC for c in "bdfhkl"}, "t": 740}
PLAIN = set("2xXMN")        # these read better on the plain skeleton (M and N as true capitals)
SEATED = set("TYLFDMN")     # stems that end on another stroke rather than in a point
TRIM = {"a": (0, 0.1), "c": (0, 0.13), "t": (0, 0.14)}   # stroke, share cut from its lower end: curled feet go
DOT_DROP = 0.35             # the dot sits this much (weights) lower than the skeleton's dot
BOWLS = {"p": 1, "b": 1, "q": 0, "d": 0, "a": 0, "g": 0}   # which stroke is the bowl
BARS = {"f": 1.5, "t": 1.9}  # crossbars lengthened by this much …
LEAN = {"f": 0.25, "t": 0.2}  # … and centred on the stem, reaching this share of their half-length further right
HOLD = 0.7                  # a stem stays full where another stroke comes within this (weights)
CARRY = {"s": 7, "S": 5}    # these strokes' upper ends carried on round by this many steps
SLIVERS = set("469")
CLASS = {"x": 0.92, "H": 0.92}   # lowercase a little lighter than capitals
PUNCT_SB = 45               # sidebearing of punctuation
TUCK = 0.1                  # share cut from a bowl's tucked inner end
DOT = 0.6                   # radius of the dot of i and j, in weights
# Leaf Ornate: a fragment of a real compound (parts.py) grows out of one stroke of each letter: laid along the
# stroke and bent to its curve, its dense edge sunk in the stroke, its ends folding down into it, and standing out
# on the side away from the letter. Its strands are drawn at about the stroke's own weight. A piece is
# (ref, spine window a, b).
STRAND = 0.5               # the fragment's strand width, in weights
RISE = 2.2                  # … less if it would otherwise stand out further than this from the stroke (weights)
REACH = 0.42                # it covers at most this share of its stroke
SEAT = 0.22                 # share of its length, at each end, folding into the stroke
CROWN = 0.8                 # the fragment crowns the letter: its stretch reaches above this share of the letter's height
ECHO = ("straight-comb/9", 0.18, 0.48)      # a strand doubling the stroke, splitting from it


def piece_for(ch):
    return ECHO if ch.isalpha() else None


FILLET = 0.1                # inside corners narrower than this (weights) are rounded, so joins leave no nicks
SPECK = 0.12                # contours smaller than this (square weights) are slivers


def stroke(s, W, others, seated=False):
    q = FM._centre(s, W)
    tg = OL.tangents(q, s.closed)
    across = 1 - ACROSS * np.abs(tg[:, 0]) ** 2.5             # thinner running across
    if s.closed:                                 # a loop: heavier on one side, thin across the top and bottom
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        a = np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean())
        w = np.maximum(W * HEAVY * (0.55 + 0.45 * np.cos(a + 2.6)) * across, FLOOR * W)
        return [[q + nr * (w / 2)[:, None], (q - nr * (w / 2)[:, None])[::-1]]]
    d = q[-1] - q[0]
    straight = s.kind == "crescent"
    upright = straight and abs(d[1]) > 0.8 * np.hypot(*d)
    L = OL.arclen(q)
    u = L / L[-1]
    near = lambda e: bool(others) and min(float(np.min(np.hypot(*(o - e).T))) for o in others) < 0.6 * W
    join0, join1 = (near(q[0]), near(q[-1])) if upright and seated else \
        (False, False) if upright else (s.taper0 or near(q[0]), s.taper1 or near(q[-1]))
    p = LOW + (1 - LOW) * np.sin(np.pi * u) ** SWELL          # a leaf: full in the middle, fine at the ends
    if upright and others:                       # a stem stays full along the stretch where other strokes meet it
        dmin = np.min([np.min(np.hypot(*(q[:, None] - o[None]).transpose(2, 0, 1)), 1) for o in others], 0)
        ua, ub = (u[dmin < HOLD * W].min(), u[dmin < HOLD * W].max()) if (dmin < HOLD * W).any() else (0, 0)
        if ub - ua > 0.2:                        # full from the first join to the last, easing into the leaf beyond
            hold = FM._ease((u - ua) / 0.12 + 1) * FM._ease((ub - u) / 0.12 + 1)
            p = p + np.clip(0.85 - p, 0, None) * hold
    if join0 and join1 and not straight:         # a curve joined at both ends is part of a bowl: it stays closed
        p = 0.55 + 0.45 * np.sin(np.pi * u)
    elif join0 and join1:                        # a straight one stays a leaf, a little fuller at its ends
        p = 0.3 + 0.7 * np.sin(np.pi * u) ** SWELL
    short = getattr(s, "free", False) or (straight and not upright and s.w < 1)
    w = W * HEAVY * s.w * np.maximum(p * across, (BAR if short else FLOOR) / HEAVY)
    for k, (x, joined) in enumerate(((L, join0), (L[-1] - L, join1))):
        if joined:
            n = 0.85 if straight else NECK       # a bar stays firm where it leaves a stem
            w = w * (n + (1 - n) * FM._ease(x / (NECK_LEN * W)))
        else:
            w = w * (TIP + (1 - TIP) * FM._ease(x / ((0.8 if short else TAPER) * W)))
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
    tail = lambda x: line((x + 18, 100), (x + 20, -20), (x - 40, -140), w=0.95)
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


def _proportion(ch, S, W):
    """Long extenders: ascenders and t's stem drawn up, descenders down, the x-height left alone."""
    hi = CU.GL.XH * 0.6 if ch == "t" else CU.GL.XH
    for s in S:
        if isinstance(s, OL.Dot):
            continue
        y = s.P0[:, 1]
        if ch in "bdfhklt" and y.max() > hi + 0.4 * (CU.GL.CAP - CU.GL.XH):    # the ascender, not an arch
            y = np.where(y > hi, hi + (y - hi) * TALL, y)
        if ch in "gjpqy":
            y = np.where(y < 0, y * DEEP, y)
        s.P0 = np.stack([s.P0[:, 0], y], 1)
        s.prepare(W)
    if ch in CARRY:                              # the upper end carried on round, so the curve reads
        s = S[0]
        top_first = s.P0[0, 1] > s.P0[-1, 1]
        q = s.P0[::-1] if top_first else s.P0
        d1, d0 = q[-1] - q[-4], q[-4] - q[-7]
        turn = np.sign(d0[0] * d1[1] - d0[1] * d1[0]) or -1.0
        a = np.arctan2(d1[1], d1[0])
        ext = [q[-1]]
        for _ in range(CARRY[ch]):
            a += turn * np.radians(9)
            ext.append(ext[-1] + 0.3 * W * np.array([np.cos(a), np.sin(a)]))
        q = np.vstack([q, ext[1:]])
        s.P0 = q[::-1] if top_first else q
        s.prepare(W)


def _grow(S, W, pieces):
    """The fragment on the letter's longest curved stroke, centred on its highest stretch, standing out away from
    the letter. A letter of straight strokes stays plain: on a stem the fragment shrinks to a knot."""
    st = [s for s in S if isinstance(s, OL.Stroke) and not getattr(s, "free", False)]
    bent = [s for s in st if s.kind != "crescent"]
    if not bent:
        return None
    s = max(bent, key=lambda q: OL.arclen(q.c)[-1])
    q = FM._centre(s, W)
    sa = OL.arclen(q)
    u = sa / sa[-1]
    mid = u[(u > 0.25) & (u < 0.75)][np.argmax(q[(u > 0.25) & (u < 0.75), 1])]
    B = np.vstack([t.c for t in st])
    if np.interp(mid, u, q[:, 1]) < B[:, 1].min() + CROWN * np.ptp(B[:, 1]):    # a curve low in the letter stays plain
        return None
    spec = pieces
    P = np.vstack(OR._fragment(*spec))
    _, V = np.linalg.eigh(np.cov((P - P.mean(0)).T))
    strand = min(STRAND, RISE / float(np.ptp(P @ V[:, 0])))
    span = float(np.ptp(P @ V[:, 1])) * strand * W                # its own proportions
    reach = min(span, REACH * sa[-1])
    u0, u1 = max(mid - reach / 2 / sa[-1], 0.0), min(mid + reach / 2 / sa[-1], 1.0)
    centre = np.vstack([t.c for t in st]).mean(0)
    line = type("Line", (), {"c": q, "closed": s.closed})
    return OR.along(spec, line, u0, u1, W, centre, span / max(reach, 1e-9), seat=SEAT)


def glyph(ch: str, W: float, ornament: bool = False) -> list[np.ndarray]:
    if ch in EXTRA:
        return punct(ch, W)
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
        S = [S[0], OL.Stroke(np.vstack([diag.P0[::-1], bar.P0[1:]]), w=0.68)]    # slim, so its counter stays open
    if ch in BARS:                               # crossbars long enough to read
        b, st = S[1], S[0]
        y = float(b.P0[:, 1].mean())
        x = float(np.interp(y, *(st.P0[np.argsort(st.P0[:, 1])].T[::-1])))      # the stem where the bar crosses it
        half = (b.P0[:, 0].max() - b.P0[:, 0].min()) / 2 * BARS[ch]
        b.P0 = np.stack([x + LEAN[ch] * half + (b.P0[:, 0] - b.P0[:, 0].mean()) * BARS[ch], b.P0[:, 1]], 1)
        b.prepare(W)
    _proportion(ch, S, W)
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
    piece = piece_for(ch) if ornament else None
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
    grafts = [g for g in [_grow(S, W, piece) if piece else None] if g]
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    cs = [C for C in OL.union(polys) if abs(area(C)) > SPECK * W ** 2]
    holes = [C for C in cs if area(C) < 0]
    tight = ch in SLIVERS                        # white slits running off these counters are closed too
    cs = FU.raster_glyph([], [], [(cs, [])], [], W, counters=[] if tight else holes, inset=0.0, smooth=0.8,
                         fillet=(0.32 if ch == "6" else 0.2) if tight else FILLET, contour_smooth=0.015 * W)
    cs = [C for C in cs if abs(area(C)) > SPECK * W ** 2]
    if grafts:                                   # the compound joins after the rounding, its own small holes kept
        cs = [C for C in OL.union([cs], grafts) if abs(area(C)) > 0.01 * W ** 2]
    if ch in "469":                              # only the main counter: where the spiral meets itself is ink
        holes = [C for C in cs if area(C) < 0]
        cs = [C for C in cs if area(C) > 0 or C is max(holes, key=lambda h: -area(h))]
    B = np.vstack(OL.union(body))
    top = TOPS.get(ch, CU.TOPS.get(ch, CU.GL.CAP)) + OVER
    bot = (DESC if ch in "gjpqy" else 0) - OVER
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
