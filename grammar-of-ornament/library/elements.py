#!/usr/bin/env python3
"""
Grammar of Ornament — library elements: web-ready SVG pieces made from the real
shapes in compounds/, strokelib/ and vgg16motifs/.

Every source SVG is drawn in paint order (black ink, white cut-outs) onto a
fine raster and traced back as one even-odd outline, so each piece is a single
path in currentColor that any colorway can recolour. Pieces are measured
(principal axis, aspect, how much of their box they fill, how thick their
lines are) and chosen for each kind of element automatically, one compound
family per element:

  divider    an elongated piece laid flat; or two copies mirrored about a small
             centre piece of the same family; or a small stroke repeated
  frieze     one piece repeated in a seamless horizontal tile
  corner     a curved piece turned into the corner, mirrored across its diagonal;
             four corners and hairlines make a frame
  mark       a compact piece in a square (bullet, section mark)
  fleuron    a piece turned n-fold round a centre, with or without mirrors
  pointer    an elongated piece pointing right (for links and "next")
  emblem     a motif, whole, for large spots only

Shapes are lists of rings (N×2 arrays, y down) filled even-odd.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent.parent
# the feature sets whose compounds and strokes are drawn on, and the prefix their refs carry (vgg16's carry none,
# so the ids it had before the other networks were added still hold): "curved-comb/05", "r-curved-comb/05"
SETS = [("vgg16features", ""), ("resnet50features", "r-"), ("convnext_basefeatures", "c-")]
MODEL_OF = {"": "vgg16", "r-": "resnet50", "c-": "convnext"}
COMPOUNDS = ROOT / "compounds"
STROKES = ROOT / "strokelib"
MOTIFS = ROOT / "vgg16motifs"

RES = 640                   # a source is traced at this many pixels along its longer side


# ---------------------------------------------------------------- geometry

def read_svg(path: Path) -> tuple[list[tuple[str, str, list[np.ndarray]]], tuple[float, float]]:
    """(fill, fill-rule, rings) for each path in paint order, and the viewBox size. Only M/L/Z occur."""
    src = path.read_text()
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', src).group(1).split()]
    out = []
    for fill, rule, d in re.findall(r'<path fill="([^"]+)" fill-rule="([^"]+)" d="([^"]+)"', src):
        rings = []
        for sub in re.split(r"(?=M)", d):
            n = re.findall(r"-?\d+(?:\.\d+)?", sub)
            if len(n) >= 6:
                rings.append(np.array(n, float).reshape(-1, 2))
        out.append((fill, rule, rings))
    return out, (vb[2], vb[3])


def raster(paths, size, res=RES) -> tuple[np.ndarray, float]:
    """Ink mask of the paths painted in order; returns the mask and the pixels per unit."""
    k = res / max(size)
    Wd, Hd = int(np.ceil(size[0] * k)) + 8, int(np.ceil(size[1] * k)) + 8
    m = np.zeros((Hd, Wd), bool)
    for fill, rule, rings in paths:
        p = np.zeros((Hd, Wd), bool)
        for R in rings:
            im = Image.new("1", (Wd, Hd), 0)
            ImageDraw.Draw(im).polygon([tuple(q) for q in R * k + 4], fill=1)
            a = np.asarray(im, bool)
            p = p ^ a if rule == "evenodd" else p | a
        m = m & ~p if fill.lower() in ("#fff", "#ffffff", "white") else m | p
    return m, k


def simplify(P: np.ndarray, tol: float) -> np.ndarray:
    """Douglas–Peucker on a closed ring."""
    if len(P) < 5:
        return P
    keep = np.zeros(len(P), bool)
    far = int(np.argmax(np.hypot(*(P - P[0]).T)))     # a closed ring is split at its farthest point from the start
    keep[0] = keep[-1] = keep[far] = True
    stack = [(0, far), (far, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        d = P[b] - P[a]
        n = np.hypot(*d) or 1.0
        seg = P[a + 1:b] - P[a]
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / n
        i = int(np.argmax(dist))
        if dist[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return P[keep]


def trace(m: np.ndarray, tol: float = 0.6, min_area: float = 6.0) -> list[np.ndarray]:
    """Outline rings of a mask (pixel units, y down)."""
    soft = ndi.gaussian_filter(np.pad(m, 2).astype(float), 0.7)
    rings = []
    for c in measure.find_contours(soft, 0.5):
        P = simplify(c[:, ::-1] - 2, tol)
        a = 0.5 * abs(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
        if len(P) >= 4 and a >= min_area:
            rings.append(P)
    return rings


def smooth_ring(P, n=160, s=None):
    """A closed polygon resampled as a smooth periodic spline (the outline as a curve, not a chain of segments)."""
    from scipy.interpolate import splprep, splev
    P = P[np.r_[True, np.any(np.diff(P, axis=0) != 0, axis=1)]]      # drop repeated points
    if len(P) < 5:
        return P
    Q = np.vstack([P, P[:1]])
    size = float(np.ptp(Q, axis=0).max())
    try:
        tck, _ = splprep([Q[:, 0], Q[:, 1]], s=(0.004 * size) ** 2 * len(Q) if s is None else s, per=True, k=3)
    except Exception:
        return P
    u = np.linspace(0, 1, n, endpoint=False)
    x, y = splev(u, tck)
    return np.stack([x, y], 1)


def bbox(rings):
    P = np.vstack(rings)
    return P.min(0), P.max(0)


def fit(rings, box, pad=0.0, align="center"):
    """Rings scaled uniformly into box (x, y, w, h)."""
    lo, hi = bbox(rings)
    x, y, w, h = box
    s = min((w - 2 * pad) / max(hi[0] - lo[0], 1e-9), (h - 2 * pad) / max(hi[1] - lo[1], 1e-9))
    size = (hi - lo) * s
    off = np.array([x + (w - size[0]) / 2, y + (h - size[1]) / 2])
    if align == "left":
        off[0] = x + pad
    return [(R - lo) * s + off for R in rings]


def rotate(rings, theta, c=(0.0, 0.0)):
    c = np.asarray(c, float)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return [(P - c) @ R.T + c for P in rings]


def mirror_x(rings, x0):
    return [np.stack([2 * x0 - P[:, 0], P[:, 1]], 1)[::-1] for P in rings]


def merge(groups, w, h, res=1400) -> list[np.ndarray]:
    """The union of several copies (each a list of rings, filled even-odd on its own), traced as one outline:
    copies that overlap stay ink, instead of cancelling."""
    k = res / max(w, h)
    Wd, Hd = int(np.ceil(w * k)) + 8, int(np.ceil(h * k)) + 8
    m = np.zeros((Hd, Wd), bool)
    for g in groups:
        p = np.zeros((Hd, Wd), bool)
        for R in g:
            im = Image.new("1", (Wd, Hd), 0)
            ImageDraw.Draw(im).polygon([tuple(q) for q in R * k + 4], fill=1)
            p ^= np.asarray(im, bool)
        m |= p
    return [(R - 4) / k for R in trace(m, tol=0.5)]


def path_d(rings, nd=1) -> str:
    f = f"{{:.{nd}f}}"
    out = []
    for P in rings:
        pts = [f.format(v).rstrip("0").rstrip(".") for v in P.ravel()]
        out.append("M" + " ".join(f"{pts[2 * i]},{pts[2 * i + 1]}" for i in range(len(P))) + "Z")
    return "".join(out)


def svg(rings, w, h, title="", extra="", attrs="") -> str:
    t = f"<title>{title}</title>" if title else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" width="{w:g}" height="{h:g}" '
            f'fill="currentColor"{" " + attrs if attrs else ""}>{t}<path fill-rule="evenodd" d="{path_d(rings)}"/>{extra}</svg>\n')


# ---------------------------------------------------------------- pieces

class Piece:
    """A source shape traced to rings (pixel units), with its measurements."""

    def __init__(self, ref: str, path: Path, res: int = RES):
        self.ref = ref
        paths, size = read_svg(path)
        m, k = raster(paths, size, res)
        self.mask = m
        self.src = [R * k + 4 for fill, _, rings in paths if fill.lower() not in ("#fff", "#ffffff", "white") for R in rings]
        self.rings = trace(m)
        ys, xs = np.nonzero(m)
        self.ink = int(m.sum())
        if self.ink < 20 or not self.rings:
            self.ok = False
            return
        self.ok = True
        X = np.stack([xs, ys], 1).astype(float)
        self.centre = X.mean(0)
        ev, V = np.linalg.eigh(np.cov((X - self.centre).T))
        self.axis = V[:, 1]
        self.aspect = float(np.sqrt(ev[1] / max(ev[0], 1e-9)))       # elongation along the principal axis
        lo, hi = bbox(self.rings)
        self.size = float(max(hi - lo))
        self.fill = self.ink / max(np.prod(hi - lo), 1.0)
        sk = skeletonize(m)
        dt = ndi.distance_transform_edt(m)
        self.line = float(2 * np.median(dt[sk])) / self.size if sk.any() else 0.0    # line width / size
        lab, self.parts = ndi.label(m)
        self.holes = max(0, len(self.rings) - self.parts)

    @property
    def tips(self) -> int:
        """Branch ends of the piece's skeleton: how much structure it has."""
        if not hasattr(self, "_tips"):
            sk = skeletonize(self.mask)
            deg = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
            self._tips = int((sk & (deg == 1)).sum())
        return self._tips

    @property
    def family(self) -> str:
        return self.ref.split("/")[0]

    def flat(self, smooth=False):
        """Rings turned so the principal axis lies along x, heavier half to the left. smooth: the source outline
        fitted with a spline instead of the traced one (the strokes' own curves, without the raster's steps)."""
        th = -np.arctan2(self.axis[1], self.axis[0])
        rings = rotate([smooth_ring(R) for R in self.src] if smooth else self.rings, th, self.centre)
        ys, xs = np.nonzero(self.mask)
        X = rotate([np.stack([xs, ys], 1).astype(float)], th, self.centre)[0]
        if np.median(X[:, 0]) > self.centre[0]:
            rings = mirror_x(rings, self.centre[0])
        return rings


def compounds() -> list[Piece]:
    out = []
    for setname, pre in SETS:
        for f in sorted((COMPOUNDS / setname).glob("*/[0-9][0-9].svg")):
            p = Piece(f"{pre}{f.parent.name}/{f.stem}", f)
            if p.ok:
                out.append(p)
    return out


def strokes() -> list[Piece]:
    out = []
    for setname, pre in SETS:
        for f in sorted((STROKES / setname).glob("family-*/stroke-*.svg")):
            p = Piece(f"{pre}{f.parent.name}/{f.stem}", f, res=240)
            if p.ok:
                out.append(p)
    return out


def model_of(ref: str) -> str:
    """Which network a ref's piece was cut from, by its prefix."""
    return next((MODEL_OF[pre] for _, pre in SETS if pre and ref.startswith(pre)), "vgg16")


# ---------------------------------------------------------------- selection

def pick(pieces, score, n, per_family=1, exclude=()):
    """The best-scoring pieces, at most per_family from each family, skipping refs already used."""
    ranked = sorted((p for p in pieces if p.ref not in exclude), key=score, reverse=True)
    out, count = [], {}
    for p in ranked:
        if score(p) <= -1e8 or count.get(p.family, 0) >= per_family:
            continue
        out.append(p)
        count[p.family] = count.get(p.family, 0) + 1
        if len(out) == n:
            break
    return out


LIGHT = ("fine-", "hatching", "crosshatching")       # families too fine to read small


def readable(p, least=0.035):
    return p.line >= least and not any(p.family.startswith(x) or p.family == x for x in LIGHT)


def s_rule(p):            # long, clean, solid enough to read at 20–30 px tall
    return p.aspect + 4 * p.line - 2 * p.holes / 10 if readable(p) and p.aspect >= 2.6 else -1e9


def s_compact(p):          # sits in a square, one piece, with branches or holes but not a solid blob
    ok = readable(p, 0.045) and p.aspect <= 1.8 and p.parts <= 2 and 0.15 <= p.fill <= 0.45 and p.tips >= 4
    return min(p.tips, 10) / 10 + min(p.holes, 3) / 3 + 5 * p.line if ok else -1e9


def s_curved(p):           # a sweeping piece for corners: elongated but not straight
    curved = p.family.startswith(("curved", "wavy", "loop", "fork"))
    return p.aspect + 3 * p.line if curved and readable(p) and 1.8 <= p.aspect <= 6 and p.parts <= 2 else -1e9


def s_petal(p):            # an elongated piece to turn round a centre
    return 4 * p.line + min(p.aspect, 5) if readable(p) and 1.8 <= p.aspect <= 7 and p.parts <= 2 else -1e9


def s_tile(p):             # a piece for a repeating band
    return 2 * p.fill + 5 * p.line + min(p.aspect, 4) / 2 if readable(p) and p.parts <= 3 else -1e9


# ---------------------------------------------------------------- elements

def divider_flat(p: Piece, height=28.0):
    rings = p.flat()
    lo, hi = bbox(rings)
    w = height * (hi[0] - lo[0]) / (hi[1] - lo[1])
    return fit(rings, (0, 0, w, height)), w, height


def divider_mirror(arm: Piece, centre: Piece, height=32.0, arm_len=5.0):
    """Two copies of an arm meeting at a centre piece: ⟵ ✱ ⟶ , arm_len × height long each side."""
    rings = arm.flat()
    lo, hi = bbox(rings)
    if np.median(np.vstack(rings)[:, 0]) < (lo[0] + hi[0]) / 2:      # heavy end toward the middle
        rings = mirror_x(rings, (lo[0] + hi[0]) / 2)
    gap = height * 1.25
    aw = arm_len * height
    left = fit(rings, (0, height * 0.18, aw, height * 0.64))
    c = fit(centre.rings, (aw + 0.1 * height, 0, gap - 0.2 * height, height))
    W = 2 * aw + gap
    right = mirror_x(left, W / 2)
    return merge([left, c, right], W, height), W, height


def divider_repeat(st: Piece, height=14.0, n=24, step=1.6):
    rings = st.flat()
    unit = fit(rings, (0, 0, height * 1.3, height), pad=height * 0.1)
    W = n * height * step
    out = []
    for i in range(n):
        dx = i * height * step
        u = unit if i % 2 == 0 else [np.stack([P[:, 0], height - P[:, 1]], 1) for P in unit]   # glide reflection
        out += [P + [dx + 0.15 * height, 0] for P in u]
    return out, W, height


def frieze(p: Piece, height=40.0):
    """One tile of a seamless band: the piece flat, centred in a cell a little wider than itself."""
    rings = p.flat()
    lo, hi = bbox(rings)
    ar = (hi[0] - lo[0]) / (hi[1] - lo[1])
    w = height * 0.8 * ar
    cell = w + height * 0.35
    return fit(rings, (0, height * 0.1, cell, height * 0.8)), cell, height


def corner(p: Piece, size=96.0):
    """The piece along the top edge from the corner, and its mirror down the left edge (top-left corner)."""
    rings = p.flat()
    lo, hi = bbox(rings)
    if np.median(np.vstack(rings)[:, 0]) > (lo[0] + hi[0]) / 2:      # heavy end into the corner
        rings = mirror_x(rings, (lo[0] + hi[0]) / 2)
    arm = fit(rings, (0, 0, size, size * 0.3), align="left")
    down = [P[:, ::-1][::-1] for P in arm]                           # reflect across the diagonal
    return merge([arm, down], size, size), size, size


def turn_corner(rings, size, where):
    """A top-left corner turned to tl, tr, br or bl."""
    if where == "tl":
        return rings
    if where == "tr":
        return [np.stack([size - P[:, 0], P[:, 1]], 1)[::-1] for P in rings]
    if where == "bl":
        return [np.stack([P[:, 0], size - P[:, 1]], 1)[::-1] for P in rings]
    return [size - P for P in rings]


EDGE_THICK = 0.10           # a frame's edge stroke, as a fraction of its corner size
EDGE_INSET = 0.16           # where the edge runs, in from the border, as a fraction of the corner size


def frame(c_rings, csize, edge: Piece, w=480.0, h=320.0):
    """Four corners joined by a single stroke along each side (the stroke stretched to the side, its taper kept),
    the heavy ends placed with a half-turn symmetry: left on top, right on the bottom, bottom on the left side."""
    out = []
    for where, (x, y) in {"tl": (0, 0), "tr": (w - csize, 0), "bl": (0, h - csize), "br": (w - csize, h - csize)}.items():
        out.append([P + [x, y] for P in turn_corner(c_rings, csize, where)])
    ins, t = csize * EDGE_INSET, csize * EDGE_THICK
    out += [place(edge, (csize, ins), (w - csize, ins), t), place(edge, (w - csize, h - ins), (csize, h - ins), t),
            place(edge, (ins, h - csize), (ins, csize), t), place(edge, (w - ins, csize), (w - ins, h - csize), t)]
    return merge(out, w, h, res=2000), w, h


def line(st: Piece, orient="t", length=400.0, thick=12.0):
    """A stroke stretched into a long line, for rules and frame edges: t runs left to right with the heavy end
    left, b the reverse; l runs up the left side with the heavy end at the bottom, r down the right side."""
    if orient in ("t", "b"):
        a, b = ((0, thick / 2), (length, thick / 2)) if orient == "t" else ((length, thick / 2), (0, thick / 2))
        return place(st, a, b, thick), length, thick
    a, b = ((thick / 2, length), (thick / 2, 0)) if orient == "l" else ((thick / 2, 0), (thick / 2, length))
    return place(st, a, b, thick), thick, length


def mark(p: Piece, size=48.0):
    return fit(p.rings, (0, 0, size, size), pad=size * 0.06), size, size


def fleuron(p: Piece, n=6, mirror=False, size=96.0):
    """The piece laid along a radius (heavy end inward), turned n times round the centre."""
    rings = p.flat()
    lo, hi = bbox(rings)
    if np.median(np.vstack(rings)[:, 0]) > (lo[0] + hi[0]) / 2:
        rings = mirror_x(rings, (lo[0] + hi[0]) / 2)
    r0, r1 = size * 0.07, size * 0.5
    arm = fit(rings, (size / 2 + r0, size / 2 - size * 0.09, r1 - r0, size * 0.18), align="left")
    arms = arm + ([np.stack([P[:, 0], size - P[:, 1]], 1)[::-1] for P in arm] if mirror else [])
    groups = [rotate(a, 2 * np.pi * i / n, (size / 2, size / 2)) for i in range(n)
              for a in ([arm, arms[len(arm):]] if mirror else [arm])]
    return merge(groups, size, size), size, size


def main_part(p):
    """How much of the piece's ink is in its largest connected component (motifs carry many specks of texture)."""
    if not hasattr(p, "_main"):
        lab, k = ndi.label(p.mask)
        p._main = float(ndi.sum(p.mask, lab, range(1, k + 1)).max() / p.ink) if k else 0.0
    return p._main


def s_medallion(p):        # a motif to turn round a centre: elongated, and mostly one body rather than scattered bits
    return p.aspect + 2 * p.fill if p.aspect >= 1.35 and main_part(p) >= 0.6 else -1e9


def axis_deg(p: Piece) -> float:
    """The clockwise turn (CSS degrees) that lays the piece's long axis along x with its heavier end left."""
    th = -np.arctan2(p.axis[1], p.axis[0])
    ys, xs = np.nonzero(p.mask)
    X = rotate([np.stack([xs, ys], 1).astype(float)], th, p.centre)[0]
    return float(np.degrees(th) + (180 if np.median(X[:, 0]) > p.centre[0] else 0)) % 360


def medallion(p: Piece, n=6, mirror=False, size=480.0):
    """A motif laid along a radius (heavy end inward) and turned n times: a rosette for large spots. The arm is as
    wide as the ring of n arms allows, so the copies nearly touch."""
    rings = p.flat()
    lo, hi = bbox(rings)
    if np.median(np.vstack(rings)[:, 0]) > (lo[0] + hi[0]) / 2:
        rings = mirror_x(rings, (lo[0] + hi[0]) / 2)
    r0, r1 = size * 0.06, size * 0.5
    aw = min(size * 0.5, 2 * np.pi * 0.3 * size / n)
    arm = fit(rings, (size / 2 + r0, size / 2 - aw / 2, r1 - r0, aw), align="left")
    groups = []
    for i in range(n):
        groups.append(rotate(arm, 2 * np.pi * i / n, (size / 2, size / 2)))
        if mirror:
            groups.append(rotate([np.stack([P[:, 0], size - P[:, 1]], 1)[::-1] for P in arm], 2 * np.pi * i / n, (size / 2, size / 2)))
    return merge(groups, size, size, res=1600), size, size


def pointer(p: Piece, height=32.0):
    """The piece flat, its lighter (pointed) end to the right."""
    rings = p.flat()
    lo, hi = bbox(rings)
    if np.median(np.vstack(rings)[:, 0]) > (lo[0] + hi[0]) / 2:
        rings = mirror_x(rings, (lo[0] + hi[0]) / 2)
    ar = min((hi[0] - lo[0]) / (hi[1] - lo[1]), 3.2)
    return fit(rings, (0, 0, height * ar, height)), height * ar, height


def place(st: Piece, a, b, thick, smooth=True):
    """A stroke laid from point a to point b, `thick` across (its own taper kept), heavy end at a. The outline
    is the stroke's own curve (spline-smoothed), not the traced raster."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.hypot(*(b - a)))
    r = st.flat(smooth=smooth)
    lo, hi = bbox(r)
    r = [(P - lo) / (hi - lo) * [L, thick] - [0, thick / 2] for P in r]     # stretched to span a..b
    return rotate([P + a for P in r], np.arctan2(*(b - a)[::-1]), a)


def ring_of(st: Piece, c, r, thick, n=10):
    """A loop: the stroke repeated round a circle (for the search lens)."""
    out = []
    for i in range(n):
        t0, t1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1.25) / n
        out.append(place(st, c + r * np.array([np.cos(t0), np.sin(t0)]), c + r * np.array([np.cos(t1), np.sin(t1)]), thick))
    return out


ICONS = {   # name → strokes as (from, to) on a 24 grid; "lens" draws a loop
    "menu": [((4, 7), (20, 7)), ((4, 12), (20, 12)), ((4, 17), (20, 17))],
    "close": [((5, 5), (19, 19)), ((19, 5), (5, 19))],
    "plus": [((12, 4), (12, 20)), ((4, 12), (20, 12))],
    "minus": [((4, 12), (20, 12))],
    "arrow-right": [((3, 12), (20, 12)), ((13, 5), (20.5, 12.3)), ((13, 19), (20.5, 11.7))],
    "arrow-left": [((21, 12), (4, 12)), ((11, 5), (3.5, 12.3)), ((11, 19), (3.5, 11.7))],
    "chevron-down": [((5, 9), (12.3, 16.5)), ((19, 9), (11.7, 16.5))],
    "chevron-right": [((9, 5), (16.5, 12.3)), ((9, 19), (16.5, 11.7))],
    "check": [((4, 13), (9.5, 18.5)), ((9, 18.8), (20, 6))],
    "search": ["lens", ((14.5, 14.5), (20.5, 20.5))],
}


def icon(name: str, st: Piece, thick=2.6, size=24.0):
    groups = []
    for s in ICONS[name]:
        if s == "lens":
            groups += ring_of(st, np.array([10.0, 10.0]), 6.0, thick * 0.9)
        else:
            groups.append(place(st, s[0], s[1], thick))
    return merge(groups, size, size, res=480), size, size


def emblem(path: Path, size=480.0):
    paths, vb = read_svg(path)
    m, _ = raster(paths, vb, 900)
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rings = trace(m, tol=0.8, min_area=10)
    return fit(rings, (0, 0, size, size)), size, size


def preview(rings, w, h, px=200, pad=4) -> Image.Image:
    """A small black-on-white raster of rings (for contact sheets)."""
    k = (px - 2 * pad) / max(w, h)
    W, H = int(w * k) + 2 * pad, int(h * k) + 2 * pad
    m = np.zeros((H, W), bool)
    for P in rings:
        im = Image.new("1", (W, H), 0)
        ImageDraw.Draw(im).polygon([tuple(q) for q in P * k + pad], fill=1)
        m ^= np.asarray(im, bool)
    return Image.fromarray(~m).convert("L")


# ---------------------------------------------------------------- shapes (clip-paths)

def outer_ring(rings):
    area = lambda P: 0.5 * abs(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    return max(rings, key=area)


def solidity(rings):
    """Area of the outer outline over the area of its convex hull: 1 for a convex shape, lower the more it is cut into."""
    from scipy.spatial import ConvexHull
    P = outer_ring(rings)
    a = 0.5 * abs(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    return float(a / max(ConvexHull(P).volume, 1e-9))


def s_shape(p):            # one solid, roundish piece whose outline can hold a panel without spikes cutting into it
    ok = p.parts == 1 and p.aspect <= 1.7 and p.fill >= 0.42 and readable(p, 0.05)
    if not ok:
        return -1e9
    sol = solidity(p.rings)
    return sol + 0.2 * (1.7 - p.aspect) if sol >= 0.78 else -1e9


def shape_polygon(rings, n_max=80):
    """The outer outline as a polygon of at most n_max points, in percent of its box (for CSS clip-path)."""
    P = outer_ring(rings)
    lo, hi = P.min(0), P.max(0)
    tol = 0.002 * max(hi - lo)
    while True:
        Q = simplify(P, tol)
        if len(Q) <= n_max:
            break
        tol *= 1.4
    Q = (Q - lo) / (hi - lo) * 100
    return Q, float(np.ptp(P[:, 0]) / np.ptp(P[:, 1]))


def safe_area(Q, n=160):
    """The largest axis-aligned rectangle inside the outline (percent box Q), as insets (top, right, bottom, left)
    in percent of the box: where a shaped panel's content can sit on solid ground."""
    im = Image.new("1", (n, n), 0)
    ImageDraw.Draw(im).polygon([(x * n / 100, y * n / 100) for x, y in Q], fill=1)
    m = np.asarray(im, bool)
    best = (0, 0, 0, 0, 0)
    for y0 in range(0, n - 4, 2):
        run = np.ones(n, bool)
        for y1 in range(y0 + 4, n, 2):
            run &= m[y1 - 2:y1].all(0)
            if not run.any():
                break
            # longest run of columns that are inside for every row y0..y1
            pad = np.concatenate([[0], run.astype(int), [0]])
            d = np.diff(pad)
            starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
            k = int(np.argmax(ends - starts))
            w = ends[k] - starts[k]
            if w * (y1 - y0) > best[0]:
                best = (w * (y1 - y0), y0, y1, starts[k], ends[k])
    _, y0, y1, x0, x1 = best
    return (100 * y0 / n, 100 * (1 - x1 / n), 100 * (1 - y1 / n), 100 * x0 / n)


def clip_path(Q):
    return "polygon(" + ",".join(f"{x:.1f}% {y:.1f}%" for x, y in Q) + ")"


# ---------------------------------------------------------------- wallpapers

GROUPS = ("shift", "turn", "mirror", "glide", "quarter", "kaleido", "hex", "brick")


def family_members(pieces, seed, k=4):
    """The seed and its nearest pieces of the same family, by aspect and line (near-identical units)."""
    same = [p for p in pieces if p.family == seed.family and p is not seed and p.parts <= seed.parts + 1]
    same.sort(key=lambda p: abs(np.log(p.aspect / seed.aspect)) + 3 * abs(p.line - seed.line))
    return [seed] + same[:k - 1]


def _unit(p, h=100.0):
    rings = p.flat()
    lo, hi = bbox(rings)
    k = h / (hi[1] - lo[1])
    return [(R - lo) * k - [0, h / 2] for R in rings], (hi[0] - lo[0]) * k     # centred on y, x from 0


def _xf(rings, rot=0.0, scale=1.0, flip=(1, 1), at=(0.0, 0.0)):
    out = []
    for P in rings:
        Q = P * [flip[0] * scale, flip[1] * scale]
        if flip[0] * flip[1] < 0:
            Q = Q[::-1]
        out.append(rotate([Q], rot)[0] + at)
    return out


def wallpaper(members, group, gap=0.35, seed=0):
    """One seamless tile of a 2-D pattern from a family of units: the symmetry group places copies, each
    repeat takes the next member, with a small hand jitter. Returns (groups of rings, w, h)."""
    rng = np.random.default_rng(seed)
    units = [_unit(p) for p in members]
    uw = max(w for _, w in units)
    H = 100.0
    cw, ch = uw * (1 + gap), H * (1 + gap)
    nxt = [0]
    def take(at, rot=0.0, flip=(1, 1), scale=1.0):
        r, w = units[nxt[0] % len(units)]
        nxt[0] += 1
        j = np.radians(rng.normal(0, 2.5))
        s = scale * (1 + rng.normal(0, 0.04))
        return _xf([P - [w / 2, 0] for P in r], rot + j, s, flip, at)
    out = []
    if group in ("shift", "turn", "mirror"):
        W, Hh = 2 * cw, 2 * ch
        for i in range(2):
            for j in range(2):
                at = ((i + 0.5) * cw, (j + 0.5) * ch)
                if group == "shift":
                    out.append(take(at))
                elif group == "turn":
                    out.append(take(at, rot=np.pi * ((i + j) % 2)))
                else:
                    out.append(take(at, flip=(1 - 2 * i, 1 - 2 * j)))
    elif group == "glide":
        W, Hh = cw, 2 * ch
        out.append(take((cw * 0.5, ch * 0.5)))
        out.append(take((cw * 1.0, ch * 1.5), flip=(1, -1)))          # (the wrap supplies the copy at x = 0)
    elif group == "brick":
        W, Hh = 2 * cw, 2 * ch
        out.append(take((0.5 * cw, 0.5 * ch)))
        out.append(take((1.5 * cw, 0.5 * ch), flip=(-1, 1)))
        out.append(take((1.0 * cw, 1.5 * ch)))
        out.append(take((2.0 * cw, 1.5 * ch), flip=(-1, 1)))
    elif group in ("quarter", "kaleido", "hex"):   # rosettes: each arm a unit laid out from the centre
        n = 6 if group == "hex" else 4
        r0 = 0.12 * H
        R = r0 + uw                                  # rosette radius
        s = 2 * R * (1 + gap * 0.6)
        def arm(c, t, flip_y=False):
            r, w = units[nxt[0] % len(units)]
            nxt[0] += 1
            j = np.radians(rng.normal(0, 2.0))
            sc = 1 + rng.normal(0, 0.04)
            return _xf([P + [r0, 0] for P in r], t + j, sc, (1, -1 if flip_y else 1), c)
        if group == "hex":
            W, Hh = s, s * np.sqrt(3)
            centres = [np.array([0.0, 0.0]), np.array([s / 2, Hh / 2])]
        else:
            W = Hh = 2 * s
            centres = [np.array([(i + 0.5) * s, (j + 0.5) * s]) for i in range(2) for j in range(2)]
        for c in centres:
            for q in range(n):
                t = 2 * np.pi * q / n + (np.pi / n if group != "hex" else 0)
                out.append(arm(c, t))
                if group == "kaleido":
                    out.append(arm(c, t, flip_y=True))
    wrapped = []
    for g in out:
        for dx in (-W, 0, W):
            for dy in (-Hh, 0, Hh):
                wrapped.append([P + [dx, dy] for P in g])
    return wrapped, W, Hh


def svg_tile(groups, w, h, title=""):
    """Groups of rings as one SVG clipped to the tile, each group its own even-odd path, in currentColor."""
    body = "".join(f'<path fill-rule="evenodd" d="{path_d(g)}"/>' for g in groups)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}" width="{w:.1f}" height="{h:.1f}" fill="currentColor">'
            f'<title>{title}</title><defs><clipPath id="t"><rect width="{w:.1f}" height="{h:.1f}"/></clipPath></defs>'
            f'<g clip-path="url(#t)">{body}</g></svg>\n')


# ---------------------------------------------------------------- drawn boxes

def edge_profile(st: Piece, n=96):
    """A stroke's upper edge as a gentle waver: y against x along its length from the smoothed outline, the tilt
    removed, scaled to ±1."""
    P = outer_ring(st.flat(smooth=True))
    lo, hi = P.min(0), P.max(0)
    i0, i1 = int(np.argmin(P[:, 0])), int(np.argmax(P[:, 0]))
    a, b = sorted([i0, i1])
    arcs = [P[a:b + 1], np.vstack([P[b:], P[:a + 1]])]                 # the two boundaries between the ends
    top = min(arcs, key=lambda A: A[:, 1].mean())                      # the upper one (y is down)
    top = top[np.argsort(top[:, 0])]
    xs = np.linspace(lo[0] + 0.03 * (hi[0] - lo[0]), hi[0] - 0.03 * (hi[0] - lo[0]), n)
    y = np.interp(xs, top[:, 0], top[:, 1])
    y = ndi.gaussian_filter1d(y, 2.0, mode="nearest")
    y = y - np.polyval(np.polyfit(xs, y, 1), xs)
    return y / max(np.abs(y).max(), 1e-9)


def drawn_box(strokes, amp=0.75, corner=0.45, seed=0):
    """A near-rectangle (percent box) whose four edges waver like the strokes they're taken from: for a clip-path."""
    rng = np.random.default_rng(seed)
    prof = [edge_profile(s) for s in strokes[:4]]
    c = rng.uniform(-corner, corner, (4, 2))             # the corners sit a little off true, as a drawn box does
    tl, tr, br, bl = [0, 0] + c[0], [100, 0] + c[1], [100, 100] + c[2], [0, 100] + c[3]
    def edge(a, b, p, normal):
        t = np.linspace(0, 1, len(p))
        return np.array([a + (b - a) * ti + normal * amp * pi for ti, pi in zip(t, p)])
    top = edge(tl, tr, prof[0], np.array([0, 1.0]))
    right = edge(tr, br, prof[1], np.array([-0.7, 0.0]))
    bottom = edge(br, bl, prof[2], np.array([0, -1.0]))
    left = edge(bl, tl, prof[3], np.array([0.7, 0.0]))
    return np.vstack([top, right[1:], bottom[1:], left[1:-1]])
