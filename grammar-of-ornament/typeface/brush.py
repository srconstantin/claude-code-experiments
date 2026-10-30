"""
Neuron Brush — every stroke of a letter is a compound piece, bent along it.

Like an art brush: each piece (silhouettes.py) is first straightened along
its own spine, so every point of its outline has a position along the spine
(u, 0..1) and a signed distance from it (d). Then it is laid onto a
skeleton stroke (glyphs.py): u runs along the stroke, d across it, scaled so
the piece's body is exactly as wide as the stroke is there. The piece's
branches, lobes and uneven edge become the letter's outline; they leave the
stroke wherever they left the piece's spine.

  family     every stroke in a font is drawn with pieces of one compound
             family (FAMILIES), chosen per stroke by a hash, among those of
             suitable proportions
  outward    a piece is flipped across the stroke if that sends its branches
             further from the letter's centre (counters stay open)
  bloom      and flipped end to end so its branchiest end is at the stroke's
             free end, not at a join
  reach      branches beyond the body are shortened by REACH, so they read
             without swamping the letter
  joins      strokes are unioned and the joins filled with fillets (fuse.py)

  .venv/bin/python typeface/brush.py FAMILY[,FAMILY…] [TEXT]   → typeface/proofs/brush/
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dress as DR          # noqa: E402
import fuse as FU           # noqa: E402
import glyphs as GL         # noqa: E402
import outline as OL        # noqa: E402
import pieces as PC         # noqa: E402

import silhouettes as SI    # noqa: E402

MAX_REACH = 9.0     # pieces with outline reaching beyond this many body-half-widths are left out
CORE_W = 0.55       # the slim Bare stroke drawn under every piece, relative to the weight
INSET = 0.12        # counters are kept open, shrunk by this many weights
SMOOTH_PX = 2.5     # final outline smoothing, font units
CLEAN = 0.09        # specks, nubs and nicks smaller than about twice this (in weights) are removed
CONTOUR_SMOOTH = 0.0   # outline smoothing along its length, in weights (0 = light)
# the cleaned-up Wavy: the richest wavy branching lines, plus a few strongly doubled pieces
WAVY_RICH = ["wavy-branching-line/0", "wavy-branching-line/1", "wavy-branching-line/5", "wavy-branching-line/11",
             "wavy-branching-line/12", "wavy-branching-line/15", "curved-branching-line/2",
             "curved-branching-line/10", "straight-feather/15", "straight-comb/0"]
REACH = 0.42        # beyond the body, branch offsets are scaled by this
CORE = 0.6          # within this many body-half-widths of the spine, offsets map exactly
STEP = 0.004        # contours are densified to this spacing (in spine lengths) before bending


def straightened(ref: str) -> dict:
    """The piece's outline in (u, d) coordinates along its own spine, in units of its body half-width."""
    s = SI.silhouette(ref)
    sp = s["spine"]
    seg = np.hypot(*np.diff(sp, axis=0).T)
    sa = np.concatenate([[0], np.cumsum(seg)])
    Ls = sa[-1]
    tan = np.gradient(sp, axis=0)
    tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)[:, None]
    tree = cKDTree(sp)
    half = s["body"] / 2
    out = []
    for C in s["contours"]:
        Cd = _densify(C, STEP)
        dist, idx = tree.query(Cd)
        rel = Cd - sp[idx]
        along = tan[idx, 0] * rel[:, 0] + tan[idx, 1] * rel[:, 1]
        across = tan[idx, 0] * rel[:, 1] - tan[idx, 1] * rel[:, 0]
        # past the spine's ends, a point keeps its distance along the spine's direction (it doesn't snap sideways)
        ends = (idx == 0) | (idx == len(sp) - 1)
        u = (sa[idx] + np.where(ends, along, 0.0)) / Ls
        d = np.where(ends, across, np.sign(across) * dist) / half
        out.append(np.stack([u, d], 1))
    # where the branching is: the mean u of points far from the spine
    P = np.vstack(out)
    far = np.abs(P[:, 1]) > 2.5
    bloom = float(P[far, 0].mean()) if far.any() else 0.5
    side = float(np.sign(np.sum(P[far, 1])) or 1) if far.any() else 1.0
    core = float(np.mean(np.abs(P[:, 1]) < 2.0))           # share of the outline near the spine
    # the piece is fitted to the stroke from its first point to its last (nothing overruns the stroke's ends)
    u0, u1 = float(P[:, 0].min()), float(P[:, 0].max())
    out = [np.stack([(uv[:, 0] - u0) / max(u1 - u0, 1e-9), uv[:, 1]], 1) for uv in out]
    # branches that run alongside the spine (instead of leaving it) read as a doubled stroke
    t = np.asarray(s["tips"]) if s["tips"] else np.zeros((0, 4))
    mid = (t[:, 0] > 0.08) & (t[:, 0] < 0.92) if len(t) else np.zeros(0, bool)
    parallel = float(np.mean(np.abs(t[mid, 3]) < 0.35)) if mid.any() else 0.0
    return {"uv": out, "bloom": bloom, "side": side, "aspect": s["aspect"], "core": core,
            "reach": float(np.percentile(np.abs(P[:, 1]), 98)), "branches": len(s["tips"]),
            "overrun": float(max(-P[:, 0].min(), P[:, 0].max() - 1, 0)), "parallel": parallel, "u0": u0, "u1": u1}


def _densify(C: np.ndarray, step: float) -> np.ndarray:
    C2 = np.vstack([C, C[:1]])
    sa = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C2, axis=0).T))])
    t = np.arange(0, sa[-1], step)
    return np.stack([np.interp(t, sa, C2[:, 0]), np.interp(t, sa, C2[:, 1])], 1)


def bend(piece: dict, c: np.ndarray, w: np.ndarray, closed: bool, flip_u: bool, flip_d: bool) -> list[np.ndarray]:
    """The straightened piece laid along centreline c (width w at each point)."""
    sa = OL.arclen(c)
    L = sa[-1]
    t = OL.tangents(c, closed)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    out = []
    for uv in piece["uv"]:
        u = 1 - uv[:, 0] if flip_u else uv[:, 0]
        d = -uv[:, 1] if flip_d else uv[:, 1]
        a = np.abs(d)
        d = np.sign(d) * np.where(a < CORE, a, CORE + (a - CORE) * REACH)       # branches shortened

        s = u * L
        cx = np.interp(s, sa, c[:, 0])
        cy = np.interp(s, sa, c[:, 1])
        nx = np.interp(s, sa, n[:, 0])
        ny = np.interp(s, sa, n[:, 1])
        ww = np.interp(s, sa, w) / 2
        out.append(np.stack([cx + nx * d * ww, cy + ny * d * ww], 1))
    return out


_CACHE: dict = {}


def family_pieces(family: str) -> list[dict]:
    if family not in _CACHE:
        out = []
        for f in sorted((PC.ROOT / family).glob("[0-9][0-9].svg")):
            ref = f"{family}/{int(f.stem)}"
            try:
                p = straightened(ref)
            except Exception:
                continue
            # careful selection: proportions like a stroke, a few branches, most of it near the spine,
            # nothing reaching absurdly far, nothing running far past its own ends
            if (4 <= p["aspect"] <= 30 and 2 <= p["branches"] <= 5 and p["core"] >= 0.45
                    and p["reach"] <= MAX_REACH and p["overrun"] <= 0.12):
                p["ref"] = ref
                out.append(p)
        _CACHE[family] = out
    return _CACHE[family]


# role of a stroke → the pieces whose effect it carries (Sarah's catalogue, 2026-10-27)
EFFECTS = {
    "stem":  ["straight-comb/0", "wavy-feather/1", "curved-comb/5"],     # thin line and blobs; ripple; grid
    "diag":  ["straight-comb/5", "straight-comb/8", "straight-comb/11"],  # brush-like split strokes
    "bar":   ["straight-comb/11"],                                        # the long split stroke
    "arch":  ["curved-comb/14", "curved-comb/15"],                        # line doubling, claws
    "curve": ["wavy-feather/14", "wavy-feather/15"],                      # vine and fruit
    "ring":  ["curved-branching-line/10"],                                # the spiral
}


def role(s) -> str:
    """What kind of stroke this is: a closed ring, a free curve (C, S), a curve joined to a stem
    (arches, attached bowls), or a straight stroke: stem, bar or diagonal."""
    if s.closed:
        return "ring"
    if s.kind != "crescent":
        return "arch" if (s.taper0 or s.taper1) else "curve"
    d = s.c[-1] - s.c[0]
    if abs(d[0]) < 0.25 * abs(d[1]):
        return "stem"
    if abs(d[1]) < 0.25 * abs(d[0]):
        return "bar"
    return "diag"


_PIECE: dict = {}


def piece(ref: str) -> dict:
    if ref not in _PIECE:
        _PIECE[ref] = straightened(ref)
        _PIECE[ref]["ref"] = ref
    return _PIECE[ref]


# letters drawn with one whole piece fitted to them (tried with wavy-comb/15 on m, w, M, W, E: the
# loops turned the letters into illegible clusters, so none for now)
EMBLEM: dict = {}
ENVELOPE_SIDE = 0.35   # ornament may reach this far (in weights) past the letter's sides …
ENVELOPE_VERT = 0.06   # … and this far past its top and bottom (no fake ascenders or descenders)
MIN_BRANCHES = 2       # a letter with fewer extra branch tips than this tries other pieces on its longest stroke
TRIES = 4


LAYERS = 1             # pieces laid on each long stroke (the extra ones go on the other side)
LAYER_MIN = 3.0        # strokes shorter than this (in weights) get one piece only
# the aggressive Vine: the most branched feathers and combs
THICKET = WAVY_RICH + ["curved-branching-line/3", "curved-branching-line/9", "curved-branching-line/13",
                       "straight-branching-line/15", "curved-comb/7", "straight-comb/2", "wavy-comb/3",
                       "straight-feather/1", "straight-feather/4", "straight-feather/15"]


def _graft(s, p, W: float, centre, side: int = 0) -> list[np.ndarray]:
    """Piece p bent along stroke s: branchy end at the stroke's free end, branches to the outside
    (side=0), or forced to the inside of the outward choice (side=1, for a second layer)."""
    w = OL.widths(s, W)
    free1 = not s.taper1 and not s.closed
    free0 = not s.taper0 and not s.closed
    want_end = 1.0 if free1 and not free0 else (0.0 if free0 and not free1 else (1.0 if p["bloom"] >= 0.5 else 0.0))
    flip_u = (p["bloom"] >= 0.5) != (want_end >= 0.5)
    opts = []
    for flip_d in (False, True):
        cs = bend(p, s.c, w, s.closed, flip_u, flip_d)
        opts.append((np.mean(np.hypot(*(np.vstack(cs) - centre).T)), cs))
    opts.sort(key=lambda o: -o[0])
    return opts[min(side, 1)][1]


def _emblem(ref: str, bare: list[np.ndarray]) -> list[np.ndarray]:
    """A whole piece fitted to the letter's box, in whichever of its eight orientations covers the
    letter best (stretched to the box: the piece keeps its loops and lobes, the letter its size)."""
    from PIL import Image, ImageDraw
    s = SI.silhouette(ref)
    P = np.vstack(bare)
    lo, hi = P.min(0), P.max(0)
    n = 96
    def mask(cs, l, h):
        m = Image.new("1", (n, n), 0)
        d = ImageDraw.Draw(m)
        for C in cs:
            if len(C) > 2:
                d.polygon([tuple((q - l) / (h - l) * (n - 1)) for q in C], fill=1)
        return np.asarray(m, bool)
    target = mask(bare, lo, hi)
    best = None
    for flip in (False, True):
        for r in range(4):
            cs = []
            for C in s["contours"]:
                q = C * [1, -1] if flip else C.copy()
                for _ in range(r):
                    q = np.stack([-q[:, 1], q[:, 0]], 1)
                cs.append(q)
            Q = np.vstack(cs)
            ql, qh = Q.min(0), Q.max(0)
            fitted = [lo + (C - ql) / (qh - ql) * (hi - lo) for C in cs]
            m = mask(fitted, lo, hi)
            iou = (m & target).sum() / max((m | target).sum(), 1)
            if best is None or iou > best[0]:
                best = (iou, fitted)
    return best[1]


def glyph(ch: str, family, W: float) -> list[np.ndarray]:
    strokes = GL.GLYPHS[ch]
    DR.dress(ch, strokes, W, None, None)
    bare = OL.outline(strokes, W)                   # (also sets the joins)
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    counters = [C for C in bare if area(C) < 0]     # the Bare letter's counters (clockwise contours)
    S = [s for s in strokes if isinstance(s, OL.Stroke)]
    centre = np.vstack([s.c for s in S]).mean(0)
    effects = family if isinstance(family, dict) else None
    if isinstance(family, (list, tuple)):          # an explicit list of pieces
        pool = [piece(r) for r in family]
    else:
        pool = None if effects else family_pieces(family)
    # the envelope: ornament stays near the letter (no fake ascenders, descenders or extra stems)
    B = np.vstack(bare)
    (bx0, by0), (bx1, by1) = B.min(0), B.max(0)
    env = [(np.array([bx0 - ENVELOPE_SIDE * W, 0.0]), np.array([1.0, 0.0])),
           (np.array([bx1 + ENVELOPE_SIDE * W, 0.0]), np.array([-1.0, 0.0])),
           (np.array([0.0, by0 - ENVELOPE_VERT * W]), np.array([0.0, 1.0])),
           (np.array([0.0, by1 + ENVELOPE_VERT * W]), np.array([0.0, -1.0]))]
    dots, polys, rings = [], [], []
    for s in strokes:
        if isinstance(s, OL.Dot):
            dots.append(OL.dot_polygon(s, W))
        elif CORE_W:                                # a slim Bare stroke under the pieces keeps the letter readable
            core = OL.polygons(s, W * CORE_W)
            if s.closed:
                rings.append(core)
            else:
                polys.extend(core)

    def pick(i, s, k=0):
        h = int(hashlib.md5(f"{family}/{ch}/{i}/{k}".encode()).hexdigest()[:8], 16)
        if effects:
            refs = effects[role(s)]
            return piece(refs[h % len(refs)])
        return pool[h % len(pool)]

    tendrils = _tendrils(ch, S, W, centre) if TENDRILS else []

    def render(choice):
        if ch in EMBLEM and RESTROKE:
            grafts = [(_emblem(EMBLEM[ch], bare), env)]
        else:
            grafts = [(_graft(s, choice[i], W, centre), env) for i, s in enumerate(S)]
            for i, s in enumerate(S):                  # extra layers on long strokes
                if OL.arclen(s.c)[-1] < LAYER_MIN * W:
                    continue
                for k in range(1, LAYERS):
                    grafts.append((_graft(s, pick(i, s, 50 + k), W, centre, side=k % 2), env))
        grafts += [(t, []) for t in tendrils]           # tendrils may leave the envelope
        return FU.raster_glyph(polys, rings, grafts, dots, W, counters=counters, inset=0.0, want_mask=True)

    choice = [pick(i, s) for i, s in enumerate(S)]
    if not RESTROKE:
        grafts = [(_graft(s, choice[i], W, centre), env) for i, s in enumerate(S)]
        return FU.raster_glyph(polys, rings, grafts, dots, W, counters=counters, inset=INSET * W,
                               smooth=SMOOTH_PX, clean=CLEAN * W, contour_smooth=CONTOUR_SMOOTH * W)
    m, lo, Hd, res = render(choice)
    if ch not in EMBLEM:
        # more branching: if few branch tips grew, try other pieces on the longest stroke
        base_tips = sum((not s.closed) and (not s.taper0) for s in S) + sum((not s.closed) and (not s.taper1) for s in S)
        tips = _tips(m) - base_tips
        if tips < MIN_BRANCHES:
            longest = max(range(len(S)), key=lambda i: OL.arclen(S[i].c)[-1])
            for k in range(1, TRIES + 1):
                alt = list(choice)
                alt[longest] = pick(longest, S[longest], k)
                m2, lo2, Hd2, res2 = render(alt)
                t2 = _tips(m2) - base_tips
                if t2 > tips:
                    tips, choice, m, lo, Hd, res = t2, alt, m2, lo2, Hd2, res2
                if tips >= MIN_BRANCHES:
                    break
    return restroke(m, lo, Hd, res, W, counters, dots, skel=np.vstack([s.c for s in S]))


# Tendrils: a few long branched pieces, at their own proportions, growing from chosen free ends.
TENDRILS: list = []          # piece refs (set by the build for the Tendril variant)
TENDRIL_LEN = 2.6            # tendril length, in weights
TENDRIL_MAX_N = 2            # at most this many per letter
TENDRIL_FAN = 28.0           # tendrils lean this many degrees away from the letter's centre
NO_TENDRIL = set("KMNVXWYZAkvwxyzas")   # diagonal letters (and a, s): legibility first


def _tendrils(ch: str, S: list, W: float, centre) -> list[list[np.ndarray]]:
    if ch in NO_TENDRIL or ch.isdigit():
        return []
    ends = []
    for i, s in enumerate(S):
        if s.closed or role(s) in ("diag", "bar"):
            continue
        t = OL.tangents(s.c, False)
        for end, joined in ((0, s.taper0), (1, s.taper1)):
            if joined:
                continue
            e = s.c[0] if end == 0 else s.c[-1]
            out = -t[0] if end == 0 else t[-1]
            if out[1] < 0.35:                            # only ends pointing up: tops, ascenders, upper terminals
                continue
            # prefer ends high up (ascenders, cap tops) and curve terminals, far from the centre
            score = e[1] + 0.5 * np.hypot(*(e - centre)) + (200 if s.kind != "crescent" else 0)
            ends.append((score, i, end, e, out))
    ends.sort(key=lambda z: -z[0])
    out_list = []
    for k, (_, i, end, e, out) in enumerate(ends[:TENDRIL_MAX_N]):
        h = int(hashlib.md5(f"tendril/{ch}/{i}/{end}".encode()).hexdigest()[:8], 16)
        ref = TENDRILS[h % len(TENDRILS)]
        sil = SI.silhouette(ref)
        p = piece(ref)
        # lean away from the centre
        away = np.sign((e - centre) @ np.array([-out[1], out[0]])) or 1.0
        a = np.radians(TENDRIL_FAN) * away
        d = np.array([out[0] * np.cos(a) - out[1] * np.sin(a), out[0] * np.sin(a) + out[1] * np.cos(a)])
        n = np.array([-d[1], d[0]])
        # the piece's plain end at the stroke, its branchy end outward; its branches toward `away`
        flip_u = p["bloom"] < 0.5
        flip_d = (p["side"] * (-1 if flip_u else 1)) * away < 0
        L = TENDRIL_LEN * W
        cs = []
        for C in sil["contours"]:
            x = 1 - C[:, 0] if flip_u else C[:, 0]
            y = -C[:, 1] if flip_d else C[:, 1]
            cs.append(e - d * 0.3 * W + np.outer(x * L, d) + np.outer(y * L, n))
        # snap the root onto the stroke: the outline point nearest the stroke's end (along the tendril's
        # base) is moved onto it, so the tendril grows from the letter and never floats
        P = np.vstack(cs)
        base = P[np.argsort((P - e) @ d)[:max(4, len(P) // 30)]].mean(0)
        shift = (e - d * 0.15 * W) - base
        out_list.append([C + shift for C in cs])
    return out_list


def _tips(m: np.ndarray) -> int:
    """Free ends of the mask's skeleton (after the redraw's spur pruning would keep them)."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    sk = skeletonize(m)
    deg = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    return int(np.sum(sk & (deg == 1)))


RESTROKE = False
ISOLATED = 1.2     # branches free at both ends and shorter than this (in weights) are dropped
JOIN_FILLET = 0.1  # inside corners are rounded with this radius (in weights)
SPUR = 0.85        # free-ended branches shorter than this (in weights) are dropped
LINE_SMOOTH = 0.22 # centrelines are smoothed over this length (in weights)
TIP = 0.9          # free ends taper to a point over this length (in weights)
W_MIN, W_MAX = 0.26, 1.05   # stroke widths are kept in this range (in weights)
W_GAMMA = 0.7      # and evened toward the weight (1 = as traced, 0 = all the same)


TENDRIL = 1.0      # branches away from the letter's skeleton are drawn at this share of their traced width
TENDRIL_MAX = 0.45 # … and at most this thick (in weights)
AWAY = 0.7         # a branch whose middle is further than this (in weights) from the skeleton is a tendril


def restroke(m: np.ndarray, lo, Hd: int, res: float, W: float, counters, dots=(), skel=None) -> list[np.ndarray]:
    """The letter redrawn as clean strokes along its own skeleton: every branch and doubled line of
    the brushed letter is kept, but drawn with a smooth centreline and an even, tapering width."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    sys.path.insert(0, str(HERE.parent))
    import linework as LW
    dt = ndi.distance_transform_edt(m)
    sk = skeletonize(m)
    brs, J = LW.branches(sk, dt)
    to_xy = lambda P: np.stack([P[:, 1] / res + lo[0], (Hd - P[:, 0]) / res + lo[1]], 1)
    polys = []
    for b in brs:
        P = b["pts"]
        if len(P) < 3:
            continue
        w = 2 * dt[P[:, 0].astype(int), P[:, 1].astype(int)] / res
        xy = to_xy(P)
        # reach back into the junctions, so strokes meet
        if b["jid0"] >= 0:
            jx, jy = J[b["jid0"]][:2]
            xy = np.vstack([to_xy(np.array([[jy, jx]])), xy]); w = np.concatenate([[w[0]], w])
        if b["jid1"] >= 0:
            jx, jy = J[b["jid1"]][:2]
            xy = np.vstack([xy, to_xy(np.array([[jy, jx]]))]); w = np.concatenate([w, [w[-1]]])
        L = OL.arclen(xy)[-1]
        free0, free1 = b["jid0"] < 0 and not b["closed"], b["jid1"] < 0 and not b["closed"]
        if (free0 or free1) and L < SPUR * W:
            continue
        if free0 and free1 and L < ISOLATED * W:      # a loose bit: an inkblot, not a stroke
            continue
        # resample, smooth the centreline and the width
        n = max(4, int(L / 4))
        t = np.linspace(0, L, n)
        sa = OL.arclen(xy)
        q = np.stack([np.interp(t, sa, xy[:, 0]), np.interp(t, sa, xy[:, 1])], 1)
        ww = np.interp(t, sa, w)
        sig = max(1.0, LINE_SMOOTH * W / 4)
        mode = "wrap" if b["closed"] else "nearest"
        qs = ndi.gaussian_filter1d(q, sig, axis=0, mode=mode)
        if not b["closed"]:                          # keep the ends where they are (joins stay joined)
            k = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / (2 * sig), 0, 1)[:, None]
            qs = k * qs + (1 - k) * q
        ww = ndi.gaussian_filter1d(ww, max(1.0, 0.6 * W / 4), mode=mode)
        ww = W * np.clip((ww / W) ** W_GAMMA, W_MIN, W_MAX)
        if skel is not None and TENDRIL < 1.0:
            far = np.min(np.hypot(*(skel[None, :, :] - q[len(q) // 2][None, None, :]).transpose(2, 0, 1)[:, 0]), axis=-1) \
                if False else float(np.min(np.hypot(*(skel - q[len(q) // 2]).T)))
            if far > AWAY * W:
                ww = np.minimum(ww * TENDRIL, TENDRIL_MAX * W)
        if free0:
            ww *= 0.12 + 0.88 * np.clip(t / (TIP * W), 0, 1) ** 0.7
        if free1:
            ww *= 0.12 + 0.88 * np.clip((L - t) / (TIP * W), 0, 1) ** 0.7
        tg = OL.tangents(qs, b["closed"])
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        left, right = qs + nr * (ww / 2)[:, None], qs - nr * (ww / 2)[:, None]
        if b["closed"]:
            polys.append([_ccw(left), _ccw(right)[::-1]])
        else:                                        # round caps, so strokes meeting at a joint blend
            polys.append(_ccw(np.vstack([left, OL._cap(qs[-1], tg[-1], ww[-1] / 2), right[::-1],
                                         OL._cap(qs[0], -tg[0], ww[0] / 2)])))
    clean = OL.union(polys + [_ccw(d) for d in dots])        # the dots of i and j are kept as they are
    # round the inside corners where branches meet strokes, drop crumbs, keep the counters clear
    return FU.raster_glyph([], [], [(clean, [])], [], W, counters=counters, inset=0.0, smooth=1.2,
                           drop_small=0.35, fillet=JOIN_FILLET, contour_smooth=0.06 * W)


def _ccw(P: np.ndarray) -> np.ndarray:
    a = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return P if a > 0 else P[::-1]


def sheet(families: list[str], text: str, out: Path, H: int = 170) -> None:
    from PIL import Image, ImageDraw, ImageFont
    lab = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    rows = []
    for fam in families:
        if isinstance(fam, str):
            print(f"  {fam}: {len(family_pieces(fam))} pieces pass the selection: {[p['ref'] for p in family_pieces(fam)]}", flush=True)
        x, polys = 40.0, []
        for ch in text:
            if ch == " ":
                x += 260
                continue
            W = GL.W_LOW if ch.islower() else GL.W_CAP
            cont = glyph(ch, fam, W)
            P = np.vstack(cont)
            dx = x - P[:, 0].min() + 50
            polys += [C + [dx, 0] for C in cont]
            x = dx + P[:, 0].max() + 50
        k = H / 1000
        im = Image.new("L", (int(x * k) + 20, int(1.45 * H)), 255)
        d = ImageDraw.Draw(im)
        area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
        for fill, sel in ((0, lambda a: a > 0), (255, lambda a: a <= 0)):
            for C in polys:
                if sel(area(C)):
                    d.polygon([(q[0] * k + 10, (1020 - q[1]) * k) for q in C], fill=fill)
        d.text((4, 2), fam if isinstance(fam, str) else "effects", font=lab, fill=140)
        rows.append(im)
        print(f"  {fam if isinstance(fam, str) else 'effects'}", flush=True)
    Wd = max(r.width for r in rows)
    sh = Image.new("L", (Wd, sum(r.height for r in rows)), 255)
    y = 0
    for r in rows:
        sh.paste(r, (0, y))
        y += r.height
    out.parent.mkdir(parents=True, exist_ok=True)
    sh.save(out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    fams = sys.argv[1].split(",")
    text = sys.argv[2] if len(sys.argv) > 2 else "HAMBURGS hamburgefonstiv"
    sheet(fams, text, HERE / "proofs" / "brush" / "families.png")
