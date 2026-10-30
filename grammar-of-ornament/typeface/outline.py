"""
Neuron — from stroke specs to smooth font outlines.

A glyph is a few strokes, each a centreline through a handful of points
(a centripetal Catmull-Rom spline, so every stroke is a smooth curve) and a
width that varies along it. The width rules come from the traced strokes in
linework/vgg16features (52,579 strokes of 8+ points, their 5-point width
profiles clustered):

  even      41%  about the same width all along          → free strokes keep
                                                            their width, with a
                                                            slight waist
  taper     31%  thinning at one end to ~0.35 of the max → where a stroke ends
                                                            on another (a branch
                                                            peeling off), it
                                                            thins to TAPER
  waist     19%  pinched in the middle to ~0.6
  crescent   9%  thin at both ends                       → a stroke joined at
                                                            both ends (the bar
                                                            of H) is a crescent

On top of that, strokes running horizontally are thinner (CONTRAST), as in
most display type, so bars don't clot. Free ends are round; an end marked
"bulb" swells a little, like the blobs on the ends of the sprigs.

Each stroke becomes a polygon (both offset sides and round caps), the
polygons are unioned (skia-pathops), and each union contour is fitted with
cubic Béziers (Schneider's algorithm), split at its corners.
"""

from __future__ import annotations

import math

import numpy as np
import pathops

STEP = 3.0          # centreline sampling, font units
# the stroke model: every stroke is a leaf (a crescent), thickest at its belly and
# narrowing toward both ends: to END_JOIN where it grows out of another stroke (a
# hairline neck), to END_FREE at a free end; bulb ends instead end in a round blob
END_JOIN = 0.10
END_CORNER = 0.9    # where two strokes meet end to end (a corner), they keep nearly full weight
CORNER_REACH = 0.7  # a join within this many weights of the other stroke's end is a corner
END_FREE = 0.30
BELLY = 0.45        # where along a stroke it is thickest (0..1); a stroke may override
PROF_AMT = 0.45     # how much of a compound spine's width variation a stroke takes on (dress.py sets the profiles)
CONTRAST = 0.55     # horizontal strokes are this much thinner
BULB = 1.15         # a bulb end's width, relative to the belly
FIT_ERR = 1.2       # curve-fitting tolerance, font units


# ── centrelines ─────────────────────────────────────────────────────────────

def spline(P, closed: bool = False, step: float = STEP) -> np.ndarray:
    """A centripetal Catmull-Rom spline through the points P, sampled about `step` apart."""
    P = np.asarray(P, float)
    if closed:
        Q = np.vstack([P[-1:], P, P[:2]])
    else:
        Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(len(Q) - 3):
        p0, p1, p2, p3 = Q[i:i + 4]
        t0 = 0.0
        t1 = t0 + max(np.hypot(*(p1 - p0)), 1e-6) ** 0.5
        t2 = t1 + max(np.hypot(*(p2 - p1)), 1e-6) ** 0.5
        t3 = t2 + max(np.hypot(*(p3 - p2)), 1e-6) ** 0.5
        n = max(2, int(math.ceil(np.hypot(*(p2 - p1)) / step)))
        t = np.linspace(t1, t2, n, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    if not closed:
        out.append(P[-1:])
    return np.vstack(out)


def tangents(c: np.ndarray, closed: bool) -> np.ndarray:
    if closed:
        d = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    else:
        d = np.gradient(c, axis=0)
    return d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]


def arclen(c: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))])


# ── strokes ─────────────────────────────────────────────────────────────────

class Stroke:
    """A centreline through `pts`; w is a weight multiplier; ends may be 'bulb'."""

    def __init__(self, pts, closed=False, w=1.0, b0=False, b1=False, taper0=None, taper1=None, belly=None,
                 curl0=0, curl1=0, teeth=None, end0=None, end1=None, e0w=None, e1w=None):
        self.P0 = np.asarray(pts, float)
        self.curl0, self.curl1 = curl0, curl1
        self.closed, self.w = closed, w
        self.belly = belly
        self.e0w, self.e1w = e0w, e1w   # end widths overriding the join/free rule (relative to the belly)
        self.kind = "curve"             # "crescent" for the straight-ish strokes made by C()
        self.grafts, self.graft_ends = [], set()   # piece outlines grafted on (dress.py), and which ends carry one
        self.teeth = teeth          # list of (piece, side, size, (u0, u1)): carry compound pieces' teeth (carried())
        self.end0, self.end1 = end0, end1    # terminal markers, e.g. "sprig" (dress.py decides what goes there)
        self.b0, self.b1 = b0, b1
        self.taper0, self.taper1 = taper0, taper1       # None = decide from the joins
        self.taper0_set, self.taper1_set = taper0, taper1
        self.prepare(W_REF)

    def prepare(self, W: float) -> None:
        """The centreline at weight W (curls scale with the weight)."""
        P = self.P0
        if self.curl1:
            P = np.vstack([P, _spiral(P[-1], P[-1] - P[-2], self.curl1, CURL_R * W)])
        if self.curl0:
            P = np.vstack([_spiral(P[0], P[0] - P[1], self.curl0, CURL_R * W)[::-1], P])
        self.P = P
        self.c = spline(P, self.closed)

    def dist(self, q) -> float:
        return float(np.min(np.hypot(*(self.c - q).T)))


CURL_R = 0.56       # starting radius of a curled terminal, relative to the weight
CURL_TURN = 290.0   # how far it turns, degrees
W_REF = 140.0


def _spiral(e, heading, sign: int, R: float) -> np.ndarray:
    """Points continuing an end at e (travelling along `heading`) into a tightening spiral,
    turning left (sign +1) or right (−1): the curl of the loop and curved-sprig compounds."""
    h = math.atan2(heading[1], heading[0])
    n = 9
    pts, p = [], np.asarray(e, float)
    dth = math.radians(CURL_TURN) / n
    for k in range(n):
        r = R * (1 - 0.6 * k / n) * abs(sign)
        h += math.copysign(dth, sign)
        p = p + r * dth * np.array([math.cos(h), math.sin(h)])
        pts.append(p)
    return np.array(pts)


def C(p0, p1, bow: float = 0.065, **kw) -> Stroke:
    """A crescent from p0 to p1. Every crescent bows the same way: oriented to travel upward
    (or rightward if level), it bulges to its left — west for a stem, up for a bar."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    flip = d[1] < -1e-6 or (abs(d[1]) <= 1e-6 and d[0] < 0)
    a, b = (p1, p0) if flip else (p0, p1)
    d = b - a
    n = np.array([-d[1], d[0]])
    t = np.linspace(0, 1, 5)
    pts = a + np.outer(t, d) + np.outer(bow * np.sin(np.pi * t), n)
    if flip:
        pts = pts[::-1]                                  # back to p0 → p1, so curl0/curl1 keep their ends
    s = Stroke(pts, **kw)
    s.kind = "crescent"
    return s


class Dot:
    def __init__(self, x, y, d=1.3):
        self.x, self.y, self.d = x, y, d


def joins(strokes: list, W: float) -> None:
    """Mark each stroke end that lands on another stroke (a join) for tapering."""
    S = [s for s in strokes if isinstance(s, Stroke)]
    for s in S:
        if s.closed:
            s.taper0 = s.taper1 = False
            continue
        for end in (0, 1):
            if (s.taper0 if end == 0 else s.taper1) is not None:
                continue
            e = s.c[0] if end == 0 else s.c[-1]
            hit = False
            for o in S:
                if o is s or o.dist(e) >= 0.5 * W * o.w + 12:
                    continue
                # a corner (ends meeting ends, as in L, E, Z, V) keeps its weight; a branch off a stroke's side thins
                sa = arclen(o.c)
                k = int(np.argmin(np.hypot(*(o.c - e).T)))
                at_end = not o.closed and min(sa[k], sa[-1] - sa[k]) < CORNER_REACH * W
                hit = "corner" if at_end and hit is not True else True
                if hit is True:
                    break
            if end == 0:
                s.taper0 = hit
            else:
                s.taper1 = hit


def widths(s: Stroke, W: float) -> np.ndarray:
    c = s.c
    s_ = arclen(c)
    L = max(s_[-1], 1e-6)
    u = s_ / L
    w = np.ones(len(c))
    if not s.closed:
        b = s.belly if s.belly is not None else BELLY
        end_w = lambda tp: END_CORNER if tp == "corner" else (END_JOIN if tp else END_FREE)
        e0 = s.e0w if s.e0w is not None else end_w(s.taper0)
        e1 = s.e1w if s.e1w is not None else end_w(s.taper1)
        if s.graft_ends:                                 # an end carrying a grafted serif keeps its body
            e0 = max(e0, GRAFT_END) if 0 in s.graft_ends else e0
            e1 = max(e1, GRAFT_END) if 1 in s.graft_ends else e1
        if s.teeth:                                      # an end carrying a serif keeps some body
            ends = {tz[0] for tz in s.teeth if tz[0] in (0, 1)}
            e0 = max(e0, SERIF_END) if 0 in ends else e0
            e1 = max(e1, SERIF_END) if 1 in ends else e1
        w = np.where(u < b, e0 + (1 - e0) * _ease(u / b), e1 + (1 - e1) * _ease((1 - u) / (1 - b)))
        bl = min(110.0, 0.35 * L)
        if s.b0:
            w = np.maximum(w, BULB * _bump(s_ / bl))
        if s.b1:
            w = np.maximum(w, BULB * _bump((L - s_) / bl))
    prof = getattr(s, "prof", None)
    if prof is not None and len(prof) > 1:              # a compound piece's own swelling and pinching
        q = np.interp(np.linspace(0, 1, len(c)), np.linspace(0, 1, len(prof)), prof)
        w *= np.clip(q, 0.55, 1.6) ** PROF_AMT
    if not getattr(s, "no_contrast", False):          # (carried teeth keep one weight in every direction)
        t = tangents(c, s.closed)
        w *= 1 - CONTRAST * np.abs(t[:, 0]) ** 3
    return W * s.w * w


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _ease(x):
    """0 → 1, rising quickly then flattening: a leaf's outline."""
    x = np.clip(x, 0, 1)
    return np.sin(x * np.pi / 2) ** 0.8


def _bump(x):
    return np.where(x < 1, np.cos(np.clip(x, 0, 1) * np.pi / 2) ** 2, 0.0)


def polygons(s: Stroke, W: float) -> list[np.ndarray]:
    c, w = s.c, widths(s, W)
    t = tangents(c, s.closed)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    L = c + n * (w / 2)[:, None]
    R = c - n * (w / 2)[:, None]
    if s.closed:
        return [L, R[::-1]]
    def cap(p, tan, r, sgn):
        a0 = math.atan2(tan[1], tan[0])
        a = a0 + sgn * np.linspace(np.pi / 2, -np.pi / 2, 14)[1:-1]
        return p + r * np.stack([np.cos(a), np.sin(a)], 1)
    end = cap(c[-1], t[-1], w[-1] / 2, 1)
    start = cap(c[0], -t[0], w[0] / 2, 1)
    return [np.vstack([L, end, R[::-1], start])]


def _cap(p, tan, r, n: int = 14) -> np.ndarray:
    """A round cap: a half circle of radius r around p, bulging along tan."""
    a0 = math.atan2(tan[1], tan[0])
    a = a0 + np.linspace(np.pi / 2, -np.pi / 2, n)[1:-1]
    return p + r * np.stack([np.cos(a), np.sin(a)], 1)


def dot_polygon(d: Dot, W: float) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    r = d.d * W / 2
    return np.stack([d.x + r * np.cos(a), d.y + r * np.sin(a)], 1)


def _path(contours, fill=None) -> pathops.Path:
    path = pathops.Path(fillType=fill) if fill is not None else pathops.Path()
    pen = path.getPen()
    for P in contours:
        pen.moveTo(tuple(P[0]))
        for q in P[1:]:
            pen.lineTo(tuple(q))
        pen.closePath()
    return path


def union(polys: list[np.ndarray], grafts: list[list[np.ndarray]] = ()) -> list[np.ndarray]:
    """The union of polygons (and of graft outlines, each a set of contours that may have holes)
    as a list of contours (outer counter-clockwise, holes clockwise). Skia occasionally fails on
    near-coincident edges: then it is retried with a tiny jitter, and failing that done on a fine
    raster (union by pixels, outline traced back with marching squares)."""
    rng = np.random.default_rng(0)
    for jitter in (0.0, 0.01, 0.05):
        try:
            jit = lambda P: [Q + rng.normal(0, jitter, Q.shape) for Q in P] if isinstance(P, (list, tuple)) \
                else P + rng.normal(0, jitter, P.shape)
            jp = [jit(P) for P in polys] if jitter else polys
            jg = [[C + rng.normal(0, jitter, C.shape) for C in g] for g in grafts] if jitter else grafts
            return _union(jp, jg)
        except pathops.PathOpsError:
            continue
    return _union_raster(polys, grafts)


def _union_raster(polys, grafts, res: float = 2.0) -> list[np.ndarray]:
    from PIL import Image, ImageDraw
    from skimage import measure
    flat = [Q for P in polys for Q in (P if isinstance(P, (list, tuple)) else [P])]
    allp = flat + [C for g in grafts for C in g]
    P = np.vstack(allp)
    lo = P.min(0) - 10
    hi = P.max(0) + 10
    Wd, Hd = (np.ceil((hi - lo) * res)).astype(int)
    acc = np.zeros((Hd, Wd), bool)
    def fill(contours, evenodd):
        m = np.zeros((Hd, Wd), bool)
        for C in contours:
            im = Image.new("1", (Wd, Hd), 0)
            ImageDraw.Draw(im).polygon([tuple((q - lo) * res) for q in C], fill=1)
            m = m ^ np.asarray(im, bool) if evenodd else m | np.asarray(im, bool)
        return m
    for Pp in polys:                                  # each stroke polygon (or ring) filled
        acc |= fill(Pp if isinstance(Pp, (list, tuple)) else [Pp], True)
    for g in grafts:
        acc |= fill(g, True)
    from scipy.ndimage import gaussian_filter
    soft = gaussian_filter(acc.astype(float), 0.7)
    out = []
    for c in measure.find_contours(np.pad(soft, 1), 0.5):
        xy = np.stack([c[:, 1] - 1, c[:, 0] - 1], 1) / res + lo
        a = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
        out.append(xy)
    # orient: outer counter-clockwise, holes clockwise (by nesting depth)
    fixed = []
    for i, C in enumerate(out):
        inside = sum(_contains(D, C[0]) for j, D in enumerate(out) if j != i)
        a = 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
        want_ccw = inside % 2 == 0
        fixed.append(C if (a > 0) == want_ccw else C[::-1])
    return fixed


def _contains(C, p) -> bool:
    x, y = p
    a, b = C, np.roll(C, -1, 0)
    m = (a[:, 1] > y) != (b[:, 1] > y)
    xs = a[m, 0] + (y - a[m, 1]) * (b[m, 0] - a[m, 0]) / (b[m, 1] - a[m, 1])
    return bool(np.sum(xs > x) % 2)


def _union(polys, grafts) -> list[np.ndarray]:
    # each stroke polygon is made consistently wound on its own first (a stroke that doubles back on
    # itself can come out clockwise, and would then cancel where it overlaps others)
    # (an item may be a group of contours: a ring's outer and inner edges, normalized together)
    path = pathops.Path()
    for P in polys:
        group = P if isinstance(P, (list, tuple)) else [P]
        one = _path(group, pathops.FillType.EVEN_ODD)
        one.simplify(fix_winding=True, clockwise=False)
        one.draw(path.getPen())
    path.simplify(fix_winding=True, clockwise=False)
    if grafts:
        both = pathops.Path()
        path.draw(both.getPen())
        for g in grafts:                      # each graft normalized (even-odd → consistent winding) first
            gp = _path(g, pathops.FillType.EVEN_ODD)
            gp.simplify(fix_winding=True, clockwise=False)
            gp.draw(both.getPen())
        both.simplify(fix_winding=True, clockwise=False)
        path = both
    out, cur = [], []
    for verb, pts in path.segments:
        if verb == "moveTo":
            cur = [pts[0]]
        elif verb == "lineTo":
            cur.append(pts[0])
        elif verb in ("closePath", "endPath"):
            if len(cur) > 2:
                out.append(np.array(cur, float))
            cur = []
        else:                                   # simplify only makes lines from lines; keep any curve's end point
            cur.append(pts[-1])
    return out


TOOTH_W = (0.30, 0.62)    # tooth widths, relative to the weight, are kept in this range


TOOTH_LEN = 2.0           # a piece's median tooth is drawn this long, in weights (its other teeth in proportion)
TOOTH_MAX = 3.2           # longest tooth, in weights
TEETH_MAX = 5             # at most this many teeth per stroke (evenly chosen from the piece's)


def carried(host: "Stroke", piece: dict, side: int = 1, size: float = 1.0, span=(0.0, 1.0), W: float = 100.0) -> list:
    """The teeth of a compound piece, laid along the host stroke: each tooth leaves the host
    where it left the piece's spine (the same fraction of the way along), at the same angle,
    with the same curve and the same length relative to the spine, scaled by `size`. side
    −1 mirrors them across the host. They are smooth strokes, thinning to a hairline where
    they leave the host and to a point at their tips."""
    c = host.c
    s_ = arclen(c)
    L = s_[-1]
    t = tangents(c, host.closed)
    u0, u1 = span
    teeth = piece["teeth"]
    if len(teeth) > TEETH_MAX:
        teeth = [teeth[int(round(k))] for k in np.linspace(0, len(teeth) - 1, TEETH_MAX)]
    lens = [arclen(np.asarray(t["pts"], float))[-1] for t in teeth]
    k_len = TOOTH_LEN * W * size / max(float(np.median(lens)), 1e-6)
    out = []
    for tooth in teeth:
        u = u0 + (u1 - u0) * tooth["u"]
        i = int(np.searchsorted(s_, u * L).clip(0, len(c) - 1))
        tt, nn = t[i], np.array([-t[i][1], t[i][0]])
        if u1 < u0:                                      # laid backwards along the host
            tt = -tt
        xy = np.asarray(tooth["pts"], float) * k_len
        xy[:, 1] = np.abs(xy[:, 1])                      # every tooth to the same side (then `side` picks which)
        xy = _resmooth(xy)
        tl = arclen(xy)[-1]
        if tl > TOOTH_MAX * W:
            xy *= TOOTH_MAX * W / tl
        pts = c[i] + np.outer(xy[:, 0], tt) + np.outer(side * xy[:, 1], nn)
        keep = [0] + [k for k in range(1, len(pts)) if np.hypot(*(pts[k] - pts[k - 1])) > 4]
        pts = pts[keep]
        if len(pts) < 2 or arclen(pts)[-1] < 0.25 * W:
            continue
        w = float(np.clip(tooth["w"] * piece.get("wscale", 1.0), *TOOTH_W)) * host.w
        out.append(Stroke(pts, w=w, taper0=True, taper1=False, belly=0.3))
        out[-1].no_contrast = True
    for o in out:
        o.prepare(W)
    return out


GRAFT_END = 0.6           # a stroke end under a grafted serif narrows only to this
SERIF_END = 0.42          # a stroke end carrying a serif narrows only to this (it doesn't vanish to a point)
SERIF_W = 0.17            # barb width, relative to the weight: hairlines
SERIF_ZONE = 1.4          # the barbs leave the stroke within this many weights of its tip
SERIF_LEN = 1.25          # a piece's median barb is drawn this long, in weights


def serif(host: "Stroke", end: int, piece: dict, W: float, mode: str = "pair", size: float = 1.0) -> list:
    """A serif made of a compound piece's teeth, at one free end of the host stroke.

    The piece's teeth are laid within SERIF_ZONE weights of the tip, in the order and at the
    spacing they had along the piece's spine, leaning toward the tip (so they fan forward
    like barbs), with their own angles and curves. mode "pair" mirrors them to both sides
    of the stroke, as a bracketed serif is; "own" keeps the sides they had on the piece."""
    c = host.c
    s_ = arclen(c)
    L = s_[-1]
    t = tangents(c, host.closed)
    teeth = piece["teeth"]
    if len(teeth) > TEETH_MAX:
        teeth = [teeth[int(round(k))] for k in np.linspace(0, len(teeth) - 1, TEETH_MAX)]
    lens = [arclen(np.asarray(q["pts"], float))[-1] for q in teeth]
    k_len = SERIF_LEN * W * size / max(float(np.median(lens)), 1e-6)
    us = np.array([q["u"] for q in teeth])
    span = max(us.max() - us.min(), 1e-6)
    zone = min(SERIF_ZONE * W * size, 0.45 * L)
    # the piece's teeth nearer its spine's end go nearer the stroke's tip
    flip_u = np.mean([np.asarray(q["pts"], float)[-1][0] for q in teeth]) < 0   # teeth lean back along the spine
    out = []
    for q, u in zip(teeth, us):
        f = (u - us.min()) / span if len(teeth) > 1 else 1.0
        if flip_u:
            f = 1 - f
        d = zone * (1 - f)                                   # distance back from the tip
        pos = (L - d) if end == 1 else d
        i = int(np.searchsorted(s_, pos).clip(0, len(c) - 1))
        tt = t[i] if end == 1 else -t[i]                     # pointing toward the tip
        nn = np.array([-tt[1], tt[0]])
        xy = _arc(np.asarray(q["pts"], float) * k_len)
        if xy[-1][0] < 0:
            xy[:, 0] *= -1                                   # lean forward, toward the tip
        tl = arclen(xy)[-1]
        if tl > TOOTH_MAX * W:
            xy *= TOOTH_MAX * W / tl
        sides = (1, -1) if mode == "pair" else (1,)
        for sd in sides:
            y = np.abs(xy[:, 1]) * sd if mode == "pair" else xy[:, 1]
            pts = c[i] + np.outer(xy[:, 0], tt) + np.outer(y, nn)
            keep = [0] + [k for k in range(1, len(pts)) if np.hypot(*(pts[k] - pts[k - 1])) > 4]
            pts = pts[keep]
            if len(pts) < 3 or arclen(pts)[-1] < 0.3 * W:
                continue
            out.append(Stroke(pts, w=SERIF_W * host.w, taper0=False, taper1=False, belly=0.15, e0w=1.0, e1w=0.12))
    for o in out:
        o.prepare(W)
    return out


def _arc(xy: np.ndarray, n: int = 16) -> np.ndarray:
    """A barb reduced to one clean curve: from its root to its tip, bowed as much as the
    traced path bowed at its middle (its angle, length and bend kept; its wobble dropped)."""
    s_ = arclen(xy)
    if s_[-1] < 1e-6:
        return xy
    mid = np.array([np.interp(s_[-1] / 2, s_, xy[:, 0]), np.interp(s_[-1] / 2, s_, xy[:, 1])])
    a, b = xy[0], xy[-1]
    t = np.linspace(0, 1, n)[:, None]
    ctrl = 2 * mid - 0.5 * (a + b)                       # a quadratic through root, middle and tip
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b


def _resmooth(xy: np.ndarray, n: int = 24) -> np.ndarray:
    """A tooth's traced path, resampled evenly and smoothed (keeping its root in place)."""
    s_ = arclen(xy)
    if s_[-1] < 1e-6:
        return xy
    t = np.linspace(0, s_[-1], n)
    q = np.stack([np.interp(t, s_, xy[:, 0]), np.interp(t, s_, xy[:, 1])], 1)
    from scipy.ndimage import gaussian_filter1d
    q2 = gaussian_filter1d(q, 2.0, axis=0, mode="nearest")
    return q2 - q2[0] + q[0]


def outline(strokes: list, W: float, pinhole: float = 0.6) -> list[np.ndarray]:
    for s in strokes:
        if isinstance(s, Stroke):
            s.prepare(W)
            s.taper0 = s.taper0_set
            s.taper1 = s.taper1_set
    extra = []
    for s in strokes:
        if isinstance(s, Stroke) and s.teeth:
            for tz in s.teeth:
                extra += serif(s, tz[0], tz[1], W, *tz[2:]) if tz[0] in (0, 1) else carried(s, *tz, W=W)
    strokes = list(strokes) + extra
    joins(strokes, W)
    polys, grafts = [], []
    _W_OUTLINE[0] = W
    for s in strokes:
        if isinstance(s, Dot):
            polys.append(dot_polygon(s, W))
        else:
            pp = polygons(s, W)
            if s.closed:
                polys.append(pp)                      # a ring: outer and inner edges together
            else:
                polys.extend(pp)
            grafts.extend(getattr(s, "grafts", []))
    out = union(polys, grafts)
    # pinholes (where a tapered tip folds over itself) are filled
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    return [C for C in out if not (area(C) < 0 and -area(C) < (pinhole * W) ** 2)]


_W_OUTLINE = [100.0]


# ── curve fitting (Schneider, "An algorithm for automatically fitting digitized curves") ──

def _bez(ctrl, t):
    t = t[:, None]
    mt = 1 - t
    return mt ** 3 * ctrl[0] + 3 * mt ** 2 * t * ctrl[1] + 3 * mt * t ** 2 * ctrl[2] + t ** 3 * ctrl[3]


def _bez1(ctrl, t):
    t = t[:, None]
    mt = 1 - t
    return 3 * mt ** 2 * (ctrl[1] - ctrl[0]) + 6 * mt * t * (ctrl[2] - ctrl[1]) + 3 * t ** 2 * (ctrl[3] - ctrl[2])


def _bez2(ctrl, t):
    t = t[:, None]
    return 6 * (1 - t) * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * t * (ctrl[3] - 2 * ctrl[2] + ctrl[1])


def _generate(P, u, t1, t2):
    A1 = (3 * (1 - u) ** 2 * u)[:, None] * t1
    A2 = (3 * (1 - u) * u ** 2)[:, None] * t2
    C = np.array([[np.sum(A1 * A1), np.sum(A1 * A2)], [np.sum(A1 * A2), np.sum(A2 * A2)]])
    p0, p3 = P[0], P[-1]
    base = _bez(np.array([p0, p0, p3, p3]), u)
    X = np.array([np.sum(A1 * (P - base)), np.sum(A2 * (P - base))])
    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    seg = np.hypot(*(p3 - p0))
    if abs(det) > 1e-12:
        a1 = (X[0] * C[1, 1] - X[1] * C[0, 1]) / det
        a2 = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det
    else:
        a1 = a2 = seg / 3
    eps = 1e-6 * seg
    if a1 < eps or a2 < eps:
        a1 = a2 = seg / 3
    return np.array([p0, p0 + t1 * a1, p3 + t2 * a2, p3])


def _reparam(ctrl, P, u):
    d = _bez(ctrl, u) - P
    d1, d2 = _bez1(ctrl, u), _bez2(ctrl, u)
    num = np.sum(d * d1, 1)
    den = np.sum(d1 * d1, 1) + np.sum(d * d2, 1)
    return np.clip(u - np.where(np.abs(den) > 1e-12, num / np.where(den == 0, 1, den), 0), 0, 1)


def _fit(P, t1, t2, err, out):
    if len(P) == 2:
        d = np.hypot(*(P[1] - P[0])) / 3
        out.append(np.array([P[0], P[0] + t1 * d, P[1] + t2 * d, P[1]]))
        return
    u = arclen(P)
    u = u / u[-1]
    ctrl = _generate(P, u, t1, t2)
    for _ in range(6):
        e = np.hypot(*(_bez(ctrl, u) - P).T)
        if e.max() < err:
            out.append(ctrl)
            return
        u = _reparam(ctrl, P, u)
        ctrl = _generate(P, u, t1, t2)
    e = np.hypot(*(_bez(ctrl, u) - P).T)
    if e.max() < err:
        out.append(ctrl)
        return
    k = int(np.clip(np.argmax(e), 1, len(P) - 2))
    tc = P[k - 1] - P[k + 1]
    tc /= max(np.hypot(*tc), 1e-9)
    _fit(P[:k + 1], t1, tc, err, out)
    _fit(P[k:], -tc, t2, err, out)


def _unit(v):
    return v / max(np.hypot(*v), 1e-9)


def fit_contour(C: np.ndarray, err: float = FIT_ERR, corner_deg: float = 50.0) -> list[np.ndarray]:
    """A closed polyline as cubic Bézier segments, split at its corners."""
    # drop near-duplicate points
    keep = [0]
    for i in range(1, len(C)):
        if np.hypot(*(C[i] - C[keep[-1]])) > 0.4:
            keep.append(i)
    C = C[keep]
    n = len(C)
    if n < 4:
        return []
    prev = C - np.roll(C, 1, 0)
    nxt = np.roll(C, -1, 0) - C
    ang = np.degrees(np.abs(np.arctan2(prev[:, 0] * nxt[:, 1] - prev[:, 1] * nxt[:, 0], (prev * nxt).sum(1))))
    corners = list(np.flatnonzero(ang > corner_deg))
    if not corners:                                         # smooth loop: cut it at two opposite points
        corners = []
        start, mid = 0, n // 2
        segs = []
        for a, b in ((start, mid), (mid, n)):
            Pp = C[a:b + 1] if b < n else np.vstack([C[a:], C[:1]])
            t1 = _unit(C[(a + 1) % n] - C[(a - 1) % n])
            t2 = -_unit(C[(b + 1) % n] - C[(b - 1) % n])
            _fit(Pp, t1, t2, err, segs)
        return segs
    segs = []
    C2 = np.vstack([C, C])
    for i, a in enumerate(corners):
        b = corners[(i + 1) % len(corners)]
        if b <= a:
            b += n
        Pp = C2[a:b + 1]
        if len(Pp) < 2:
            continue
        t1 = _unit(Pp[min(1, len(Pp) - 1)] - Pp[0]) if len(Pp) < 4 else _unit(Pp[2] - Pp[0])
        t2 = _unit(Pp[-2] - Pp[-1]) if len(Pp) < 4 else _unit(Pp[-3] - Pp[-1])
        _fit(Pp, t1, t2, err, segs)
    return segs
