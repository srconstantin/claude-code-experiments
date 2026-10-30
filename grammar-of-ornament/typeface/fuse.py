"""
Neuron — real compound pieces fused into the letters as serifs.

At a serif position (a stem's foot or top on its guide line, the tip of an
arm), the stroke runs into the branching end of an actual compound piece
(silhouettes.py), so the serif is the stroke splitting, the way the
compound's own strokes split:

  orient   the piece is laid along the stroke, its spine on the stroke's axis,
           with the end where it branches at the stroke's tip (the side where
           its outline spreads widest across the spine)
  cut      only the part from a little behind its first branch to its end is
           kept
  scale    so the piece's spine is as thick as the stroke where they meet
  distort  squashed along the stroke (SQUASH) so its branches splay sideways,
           as a serif does, rather than straight on
  mirror   for a two-sided serif, the piece and its mirror image across the
           stroke's axis
  trim     a serif on a guide line is cut flush with the line, so the letter
           stands on the baseline (or hangs from the cap height) like a
           classical letter

The whole glyph (strokes and pieces) is then drawn on a fine raster and
closed with a disc of radius FILLET × the weight: every crevice where a
piece meets a stroke, or two strokes meet, is filled with a smooth concave
fillet, which is what makes a classical bracketed serif look grown rather than
attached. The outline is traced back, and the build fits it with Béziers.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import measure

import outline as OL
import silhouettes as SI

RES = 2.0         # raster pixels per font unit
FILLET = 0.12     # closing radius, in weights
SQUASH = 0.65     # the piece is compressed to this along the stroke
BEHIND = 1.6      # keep this many piece-body-widths behind its first branch
OVERHANG = 0.35   # a line serif's piece reaches this many stroke widths past the line, then is trimmed
PINHOLE = 0.9     # holes smaller than this (in weights, squared) are filled
SPAN = 1.25       # a serif reaches this many weights to each side of the stroke's axis
BODY_MAX = 0.5    # …and the piece's own strokes are at most this thick, relative to the stroke


def _orient(sil: dict):
    """(x of the branching end, sign): +1 if the piece branches toward its spine's end (x=1), −1 if toward x=0."""
    P = np.vstack(sil["contours"])
    bx = np.asarray(sil["branch_x"]) if sil["branch_x"] else np.array([0.5])
    if len(bx) and (bx.min() < 0.25 or bx.max() > 0.75) and not (bx.min() < 0.25 and bx.max() > 0.75):
        side = 1 if bx.mean() > 0.5 else -1               # branches crowd one end
    else:
        mid = np.median(bx)                               # a fork: the prongs are where it spreads widest
        lo, hi = P[P[:, 0] < mid], P[P[:, 0] >= mid]
        spread = lambda Q: np.ptp(Q[:, 1]) if len(Q) else 0
        side = 1 if spread(hi) >= spread(lo) else -1
    first = (bx.min() if side == 1 else bx.max())
    return first, side


def serif_piece(ref: str, e, out, w_end: float, line: float | None, mirror: bool, W: float,
                span: float = 1.0, branches: int | None = None, body_max: float | None = None,
                behind: float | None = None, squash: float | None = None, whole: float | None = None,
                depth: int = 0):
    """The piece's outline placed at stroke end e (out: unit vector out of the stroke), as
    contours plus the half-planes to keep: [(contours, [(point, normal), ...]), ...]."""
    sil = SI.silhouette(ref)
    if whole:
        return _whole_piece(sil, e, out, w_end, whole * W, mirror, squash)
    first, side = _orient(sil)
    body = sil["body"]
    if branches:                                          # keep this many branch points back from the tip
        bx = sorted(set(np.round(sil["branch_x"], 2)), reverse=(side == 1))
        if len(bx) >= branches:
            first = bx[branches - 1]
    nrm = np.array([-out[1], out[0]])
    sp = sil["spine"]
    xc = first - side * (BEHIND if behind is None else behind) * body   # where the cut is
    y0 = float(np.interp(xc, sp[:, 0], sp[:, 1])) if side == 1 else float(np.interp(xc, sp[::-1, 0], sp[::-1, 1]))
    P = np.vstack(sil["contours"])
    kept = P[(P[:, 0] - xc) * side >= 0]
    half = float(np.max(np.abs(kept[:, 1] - y0))) if len(kept) else body
    # piece units → font units: the cut piece spreads SPAN weights to a side, unless that makes its strokes too heavy
    k = min(SPAN * span * W / max(half, 1e-6), (BODY_MAX if body_max is None else body_max) * w_end / max(body, 1e-6))
    tip = P[:, 0].max() if side == 1 else P[:, 0].min()
    sq = SQUASH if squash is None else squash
    reach = abs(tip - xc) * k * sq
    # the piece's tip sits OVERHANG stroke widths past the end (a line serif is then trimmed back to the line)
    tip_at = np.asarray(e, float) + out * (OVERHANG * w_end if line is not None else 0.0)
    out_list = []
    for sgn in ((1, -1) if mirror else (1,)):
        cs = []
        for C in sil["contours"]:
            along = (C[:, 0] - tip) * side * k * sq           # ≤ 0 behind the tip
            across = (C[:, 1] - y0) * side * sgn * k
            cs.append(tip_at + np.outer(along, out) + np.outer(across, nrm))
        keep = [(tip_at - out * reach, out)]                  # the cut: only the branching end
        if line is not None:
            keep.append((np.array([0.0, line]), np.array([0.0, -np.sign(out[1])])))   # flush with the line
        out_list.append((cs, keep))
        # dendrites: at each prong tip of this cut, a smaller copy of the same piece, continuing the prong
        if depth > 0:
            place = lambda x, y: tip_at + (x - tip) * side * k * sq * out + (y - y0) * side * sgn * k * nrm
            for tx, ty, dx, dy in sil["tips"]:
                if (tx - xc) * side < 0.5 * abs(tip - xc):       # only tips out in the branching part
                    continue
                pt = place(tx, ty)
                if line is not None and (pt[1] - line) * np.sign(out[1]) > 0:
                    continue                                     # trimmed away
                d = dx * side * sq * out + dy * side * sgn * nrm
                d /= max(np.hypot(*d), 1e-9)
                out_list += serif_piece(ref, pt - d * 0.15 * w_end, d, DENDRITE * w_end, None, False, W,
                                        span=span * DENDRITE, branches=branches, body_max=body_max,
                                        behind=behind, squash=squash, depth=depth - 1)
    return out_list


def _even(C: np.ndarray, step: float = 2.5, sigma: float = 1.2) -> np.ndarray:
    """A traced closed contour resampled every `step` units and lightly smoothed (no pixel stair-steps)."""
    C2 = np.vstack([C, C[:1]])
    s_ = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C2, axis=0).T))])
    n = max(8, int(s_[-1] / step))
    t = np.linspace(0, s_[-1], n, endpoint=False)
    q = np.stack([np.interp(t, s_, C2[:, 0]), np.interp(t, s_, C2[:, 1])], 1)
    return ndi.gaussian_filter1d(q, sigma, axis=0, mode="wrap")


def _whole_piece(sil: dict, e, out, w_end: float, length: float, mirror: bool, squash):
    """The whole piece, undistorted (unless squash is given), laid along the stroke so its spine runs
    down the stroke from the tip: the stroke's end becomes the piece. `length` is its extent along the
    stroke. Its branching end is at the tip; it reaches WHOLE_OVER of its length past the stroke's end."""
    first, side = _orient(sil)
    P = np.vstack(sil["contours"])
    xs = P[:, 0] * side
    tip, tail = xs.max(), xs.min()
    sq = 1.0 if squash is None else squash
    k = length / max((tip - tail) * sq, 1e-6)
    sp = sil["spine"]
    # line the spine up with the stroke's axis where the piece meets the stroke (its tail third)
    m = (sp[:, 0] * side) < tail + 0.35 * (tip - tail)
    y0 = float(np.median(sp[m, 1])) if m.any() else 0.0
    nrm = np.array([-out[1], out[0]])
    tip_at = np.asarray(e, float) + out * WHOLE_OVER * length
    res = []
    for sgn in ((1, -1) if mirror else (1,)):
        cs = [tip_at + np.outer((C[:, 0] * side - tip) * k * sq, out) + np.outer((C[:, 1] - y0) * side * sgn * k, nrm)
              for C in sil["contours"]]
        res.append((cs, []))
    return res


WHOLE_OVER = 0.25
DENDRITE = 0.75   # each generation of dendrites is this much smaller than its parent


def raster_glyph(polys: list, rings: list, grafts: list, dots: list, W: float,
                 counters: list | None = None, inset: float = 0.0, smooth: float = 0.8,
                 clean: float = 0.0, contour_smooth: float = 0.0, want_mask: bool = False,
                 drop_small: float = 0.0, fillet: float | None = None):
    """Fill strokes (simple polygons), rings (pairs: outer, inner), grafts (contours + half-planes)
    and dots on a raster, close it with the fillet disc, and trace the outline (font units)."""
    allp = list(polys) + [C for r in rings for C in r] + [C for g, _ in grafts for C in g] + [d for d in dots]
    Pts = np.vstack(allp)
    lo = Pts.min(0) - 2 * FILLET * W - 10
    hi = Pts.max(0) + 2 * FILLET * W + 10
    Wd, Hd = np.ceil((hi - lo) * RES).astype(int)

    def to_px(C):
        q = (C - lo) * RES
        return [(x, Hd - y) for x, y in q]

    def fill(contours, evenodd=True):
        m = np.zeros((Hd, Wd), bool)
        for C in contours:
            if len(C) < 3:                    # a degenerate fragment
                continue
            im = Image.new("1", (Wd, Hd), 0)
            ImageDraw.Draw(im).polygon(to_px(C), fill=1)
            a = np.asarray(im, bool)
            m = (m ^ a) if evenodd else (m | a)
        return m

    acc = np.zeros((Hd, Wd), bool)
    for P in polys:
        acc |= fill([P])
    for r in rings:
        acc |= fill(r)
    for d in dots:
        acc |= fill([d])
    ys, xs = np.mgrid[0:Hd, 0:Wd]
    X = xs / RES + lo[0]
    Y = (Hd - ys) / RES + lo[1]
    for contours, keep in grafts:
        g = fill(contours)
        for p, n in keep:
            g &= (X - p[0]) * n[0] + (Y - p[1]) * n[1] >= 0
        acc |= g
    r = (FILLET if fillet is None else fillet) * W * RES
    dil = ndi.distance_transform_edt(~acc) <= r
    closed = ndi.distance_transform_edt(dil) > r
    closed |= acc
    # pinholes the pieces leave (their own little counters, half-buried in a stroke) are filled; real counters stay
    holes = ndi.binary_fill_holes(closed) & ~closed
    lab, n = ndi.label(holes)
    if n:
        sz = ndi.sum(holes, lab, index=np.arange(1, n + 1))
        closed |= np.isin(lab, 1 + np.flatnonzero(sz < (PINHOLE * W * RES) ** 2))
    if counters:                      # the letter's own counters are kept open (shrunk by `inset` units)
        hole = fill(counters)
        if inset > 0:
            hole = ndi.distance_transform_edt(hole) > inset * RES
        closed &= ~hole
    if clean > 0:
        # remove what is smaller than the letter's own scale: specks and nubs (opening), nicks and
        # pinched gaps (closing), crumbs that came loose
        r = clean * RES
        closed = ndi.distance_transform_edt(closed) > r
        closed = ndi.distance_transform_edt(~closed) <= r
        closed = ndi.distance_transform_edt(~closed) <= r
        closed = ndi.distance_transform_edt(closed) > r
        lab, n = ndi.label(closed)
        if n > 1:
            sz = ndi.sum(closed, lab, index=np.arange(1, n + 1))
            closed = np.isin(lab, 1 + np.flatnonzero(sz >= (1.2 * W * RES) ** 2 * 0.25))
        if counters:
            closed &= ~hole
    if drop_small > 0:                # loose crumbs (inkblot dots) are dropped
        lab, n = ndi.label(closed)
        if n > 1:
            sz = ndi.sum(closed, lab, index=np.arange(1, n + 1))
            closed = np.isin(lab, 1 + np.flatnonzero(sz >= drop_small * (W * RES) ** 2))
    if want_mask:                     # the filled raster itself: (mask, origin in font units, height, px per unit)
        return closed, lo, Hd, RES
    soft = ndi.gaussian_filter(closed.astype(float), smooth * RES)
    out = []
    for c in measure.find_contours(np.pad(soft, 1), 0.5):
        xy = np.stack([(c[:, 1] - 1) / RES + lo[0], (Hd - (c[:, 0] - 1)) / RES + lo[1]], 1)
        if len(xy) > 8:
            out.append(_even(xy, sigma=max(1.2, contour_smooth / 2.5)))
    fixed = []
    for i, C in enumerate(out):
        depth = sum(OL._contains(D, C[0]) for j, D in enumerate(out) if j != i)
        a = 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
        fixed.append(C if (a > 0) == (depth % 2 == 0) else C[::-1])
    return fixed
