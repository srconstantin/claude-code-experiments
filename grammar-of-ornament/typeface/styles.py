"""
Neuron styles — other letter constructions on the Bare skeleton (glyphs.py).

Each style rebuilds a glyph's strokes in a different way, using behaviours
seen in the compound strokes. None of them starts from an existing typeface.

  thorn   semi-serif. High contrast (horizontals hairline). Every stem end that
          sits on a guide line grows one hooked barb, like one prong of a
          fork: feet point right, tops point left, so the barbs turn around
          each letter
  script  slanted SLANT degrees; leaf-shaped strokes with pointed ends; a
          lowercase letter's rightmost foot sweeps up into an exit stroke, a
          capital's first top grows an entry hook
  split   every stroke is a separate pointed leaf; where strokes meet, they
          stop GAP short of each other, as if the letter were laid out from
          separate compound strokes; bowls break at top and bottom
  double  every stroke is drawn as two parallel lines (the compounds' line
          doubling), DOUBLE apart, each LINE of the weight
"""

from __future__ import annotations

import math

import numpy as np

import glyphs as GL
import outline as OL

SLANT = 13.0
GAP = 0.32        # in weights
DOUBLE = 0.34     # half the distance between the two lines, in weights
LINE = 0.4        # each line's weight, relative to the letter's

LINES = (0, GL.XH, GL.CAP, GL.ASC, GL.DESC)

PARAMS = {        # stroke-model settings per style (outline.py)
    "thorn":  dict(END_JOIN=0.3, END_FREE=0.95, CONTRAST=0.8),
    "script": dict(END_JOIN=0.3, END_FREE=0.12, CONTRAST=0.3),
    "split":  dict(END_JOIN=0.12, END_FREE=0.12, CONTRAST=0.4),
    "double": dict(END_JOIN=0.6, END_FREE=0.7, CONTRAST=0.3),
}


def _copy(s: OL.Stroke, P=None, **kw) -> OL.Stroke:
    n = OL.Stroke(s.P0 if P is None else P, closed=kw.pop("closed", s.closed), w=kw.pop("w", s.w),
                  belly=kw.pop("belly", s.belly), end0=s.end0, end1=s.end1, **kw)
    n.kind = s.kind
    return n


def _base(ch: str, W: float) -> list:
    """Fresh copies of the glyph's strokes, prepared, with their joins found."""
    out = []
    for s in GL.GLYPHS[ch]:
        out.append(OL.Dot(s.x, s.y, s.d) if isinstance(s, OL.Dot) else _copy(s))
    S = [s for s in out if isinstance(s, OL.Stroke)]
    for s in S:
        s.prepare(W)
    OL.joins(out, W)
    return out


def _ends(S):
    """Free ends: (stroke, end, point, outward unit vector)."""
    for s in S:
        if s.closed:
            continue
        t = OL.tangents(s.c, False)
        for end, joined in ((0, s.taper0), (1, s.taper1)):
            if not joined:
                yield s, end, (s.c[0] if end == 0 else s.c[-1]), (-t[0] if end == 0 else t[-1])


def _on_line(e, out) -> bool:
    line = min(LINES, key=lambda y: abs(e[1] - y))
    return abs(e[1] - line) < 45 and abs(out[1]) > 0.35


def styled(ch: str, style: str, W: float) -> list:
    return {"thorn": thorn, "script": script, "split": split, "double": double}[style](ch, W)


# ── thorn ───────────────────────────────────────────────────────────────────

def thorn(ch: str, W: float) -> list:
    strokes = _base(ch, W)
    S = [s for s in strokes if isinstance(s, OL.Stroke)]
    extra = []
    for s, end, e, out in list(_ends(S)):
        if s.kind != "crescent":                       # curve ends: a blunt wedge
            if end == 0:
                s.e0w = 0.7
            else:
                s.e1w = 0.7
            continue
        if not _on_line(e, out):
            continue
        side = 1.0 if out[1] < 0 else -1.0             # feet hook right, tops hook left
        x = np.array([side, 0.0])
        up = -np.array([0.0, np.sign(out[1])])          # back into the letter
        P = [e - x * 0.25 * W + up * 0.3 * W, e + x * 0.45 * W, e + x * 1.05 * W + up * 0.12 * W,
             e + x * 1.45 * W + up * 0.5 * W, e + x * 1.5 * W + up * 0.85 * W]
        b = OL.Stroke(P, w=0.85 * s.w, taper0=False, taper1=False, belly=0.2, e0w=1.0, e1w=0.04)
        extra.append(b)
    return strokes + extra


# ── script ──────────────────────────────────────────────────────────────────

def _shear(P, k):
    P = np.array(P, float)
    P[:, 0] += P[:, 1] * k
    return P


def script(ch: str, W: float) -> list:
    k = math.tan(math.radians(SLANT))
    strokes = _base(ch, W)
    S = [s for s in strokes if isinstance(s, OL.Stroke)]
    extra = []
    ends = list(_ends(S))
    if ch.islower():
        feet = [z for z in ends if z[2][1] < 60 and z[3][1] < -0.35]
        if feet:
            s, end, e, out = max(feet, key=lambda z: z[2][0])
            P = [e, e + np.array([0.3, -0.14]) * W, e + np.array([0.9, 0.02]) * W, e + np.array([1.3, 0.55]) * W]
            extra.append(OL.Stroke(P, w=0.7 * s.w, taper0=False, taper1=False, belly=0.2, e0w=1.0, e1w=0.08))
    elif ch.isalpha():
        heads = [z for z in ends if z[3][1] > 0.35]
        if heads:
            s, end, e, out = min(heads, key=lambda z: z[2][0])
            P = [e, e + np.array([-0.35, 0.16]) * W, e + np.array([-1.0, 0.08]) * W, e + np.array([-1.3, -0.45]) * W]
            extra.append(OL.Stroke(P, w=0.7 * s.w, taper0=False, taper1=False, belly=0.2, e0w=1.0, e1w=0.08))
    out = []
    for s in strokes + extra:
        if isinstance(s, OL.Dot):
            out.append(OL.Dot(s.x + s.y * k, s.y, s.d))
        else:
            n = _copy(s, _shear(s.c if s.closed else s.c, k), belly=0.62 if s.belly is None else s.belly,
                      taper0=s.taper0_set, taper1=s.taper1_set, e0w=s.e0w, e1w=s.e1w)
            out.append(n)
    return out


# ── split ───────────────────────────────────────────────────────────────────

def _trim(c: np.ndarray, d0: float, d1: float) -> np.ndarray | None:
    sa = OL.arclen(c)
    L = sa[-1]
    if L - d0 - d1 < 20:
        return None
    t = np.linspace(d0, L - d1, max(4, int((L - d0 - d1) / 4)))
    return np.stack([np.interp(t, sa, c[:, 0]), np.interp(t, sa, c[:, 1])], 1)


def split(ch: str, W: float) -> list:
    strokes = _base(ch, W)
    g = GAP * W
    out = []
    for s in strokes:
        if isinstance(s, OL.Dot):
            out.append(s)
            continue
        if s.closed:                                   # a bowl breaks at its top and bottom
            c = s.c
            top, bot = int(np.argmax(c[:, 1])), int(np.argmin(c[:, 1]))
            a, b = sorted((top, bot))
            for arc in (c[a:b + 1], np.vstack([c[b:], c[:a + 1]])):
                q = _trim(arc, g / 2, g / 2)
                if q is not None:
                    out.append(_copy(s, q, closed=False, taper0=False, taper1=False))
            continue
        q = _trim(s.c, g / 2 if s.taper0 else 0.0, g / 2 if s.taper1 else 0.0)
        if q is not None:
            out.append(_copy(s, q, taper0=False, taper1=False))
    return out


# ── double ──────────────────────────────────────────────────────────────────

def double(ch: str, W: float) -> list:
    strokes = _base(ch, W)
    out = []
    for s in strokes:
        if isinstance(s, OL.Dot):
            out.append(s)
            continue
        c = s.c
        t = OL.tangents(c, s.closed)
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        for sgn in (1, -1):
            q = c + sgn * DOUBLE * W * n
            out.append(_copy(s, q, w=LINE * s.w, taper0=s.taper0_set, taper1=s.taper1_set))
    return out


# ── more constructions ──────────────────────────────────────────────────────

PARAMS.update({
    "wedge":   dict(END_JOIN=0.3, END_FREE=0.05, CONTRAST=0.15),
    "reverse": dict(END_JOIN=0.5, END_FREE=0.8, CONTRAST=-3.2),
    "bulb":    dict(END_JOIN=0.6, END_FREE=0.6, CONTRAST=0.2, BULB=3.4),
    "bead":    dict(END_JOIN=0.4, END_FREE=0.35, CONTRAST=0.2, PROF_AMT=1.0),
})
BEAD = 1.5        # bead spacing along a stroke, in weights


def _wedge(ch: str, W: float) -> list:
    """Every stroke a thorn: full weight at its root (the joined end, else its lower or left end),
    tapering to a point."""
    strokes = _base(ch, W)
    for s in strokes:
        if isinstance(s, OL.Dot) or s.closed:
            continue
        if bool(s.taper0) != bool(s.taper1):
            root0 = bool(s.taper0)
        else:
            a, b = s.c[0], s.c[-1]
            root0 = (a[1] < b[1] - 30) or (abs(a[1] - b[1]) <= 30 and a[0] <= b[0])
        s.w *= 1.45
        s.belly = 0.1 if root0 else 0.9
        s.e0w, s.e1w = (1.0, 0.04) if root0 else (0.04, 1.0)
    return strokes


def _reverse(ch: str, W: float) -> list:
    strokes = _base(ch, W)
    for s in strokes:
        if isinstance(s, OL.Stroke):
            s.w *= 0.28
    return strokes


def _bulb(ch: str, W: float) -> list:
    strokes = _base(ch, W)
    for s in strokes:
        if isinstance(s, OL.Dot):
            continue
        s.w *= 0.36
        if s.closed:
            continue
        s.b0, s.b1 = not s.taper0, not s.taper1
    return strokes


def _bead(ch: str, W: float) -> list:
    strokes = _base(ch, W)
    for s in strokes:
        if isinstance(s, OL.Dot):
            continue
        L = OL.arclen(s.c)[-1]
        n = max(2, round(L / (BEAD * W)))
        u = np.linspace(0, 1, 200)
        s.prof = 1.0 + 0.5 * np.cos(2 * np.pi * n * u + (0 if s.closed else np.pi))
        s.w *= 0.85
    return strokes


_STYLES = {"wedge": _wedge, "reverse": _reverse, "bulb": _bulb, "bead": _bead}
_old_styled = styled


def styled(ch: str, style: str, W: float) -> list:      # noqa: F811
    return _STYLES[style](ch, W) if style in _STYLES else _old_styled(ch, style, W)


# ── reverse, ornamented ─────────────────────────────────────────────────────
# Reverse's weight is in its horizontals; those heavy strokes are brushed with real
# compound pieces (brush.py), teeth outward, while the hairline verticals stay clean.

ORN_POOLS = {
    "comb":    ["straight-comb/0", "straight-comb/2", "straight-comb/4", "straight-comb/7", "curved-comb/7",
                "wavy-comb/3", "wavy-comb/14"],
    "feather": ["straight-feather/1", "straight-feather/2", "straight-feather/4", "straight-feather/15",
                "straight-feather/3", "straight-feather/7"],
}
ORN_REACH = 0.6       # branch length beyond the stroke (brush.REACH)
ORN_SIDE, ORN_VERT = 0.45, 0.4   # how far ornament may leave the letter's box, in weights


def reverse_orn(ch: str, W: float, pool: str = "comb") -> list[np.ndarray]:
    import hashlib
    import brush as BR
    import fuse as FU
    strokes = _reverse(ch, W)
    bare = OL.outline(strokes, W)
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    counters = [C for C in bare if area(C) < 0]
    S = [s for s in strokes if isinstance(s, OL.Stroke)]
    centre = np.vstack([s.c for s in S]).mean(0)
    B = np.vstack(bare)
    (x0, y0), (x1, y1) = B.min(0), B.max(0)
    env = [(np.array([x0 - ORN_SIDE * W, 0.0]), np.array([1.0, 0.0])),
           (np.array([x1 + ORN_SIDE * W, 0.0]), np.array([-1.0, 0.0])),
           (np.array([0.0, y0 - ORN_VERT * W]), np.array([0.0, 1.0])),
           (np.array([0.0, y1 + ORN_VERT * W]), np.array([0.0, -1.0]))]
    old = BR.REACH
    BR.REACH = ORN_REACH
    refs = ORN_POOLS[pool]
    grafts = [(bare, [])]
    for i, s in enumerate(S):
        if s.closed:
            continue
        d = s.c[-1] - s.c[0]
        horizontal = s.kind != "crescent" or abs(d[1]) < 0.4 * abs(d[0])
        if not horizontal:
            continue
        h = int(hashlib.md5(f"{pool}/{ch}/{i}".encode()).hexdigest()[:8], 16)
        grafts.append((BR._graft(s, BR.piece(refs[h % len(refs)]), W, centre), env))
    BR.REACH = old
    return FU.raster_glyph([], [], grafts, [], W, counters=counters, inset=0.0, smooth=1.5,
                           clean=0.06 * W, contour_smooth=0.06 * W)


_COMBS = None


def _comb_pool() -> list[dict]:
    """Combs and feathers from pieces.py with 3–9 teeth, mostly to one side, of moderate length."""
    global _COMBS
    if _COMBS is None:
        import pieces as PC
        lib = PC.library()
        tl = lambda t: float(np.sum(np.hypot(*np.diff(np.asarray(t["pts"]), axis=0).T)))
        one = lambda p: abs(sum(np.sign(np.asarray(t["pts"])[-1][1]) for t in p["teeth"])) / len(p["teeth"])
        _COMBS = [p for k in ("straight-comb", "curved-comb", "wavy-comb", "straight-feather", "curved-feather",
                              "wavy-feather") for p in lib.get(k, [])
                  if 3 <= len(p["teeth"]) <= 9 and one(p) >= 0.4
                  and 0.06 <= np.median([tl(t) for t in p["teeth"]]) <= 0.4]
    return _COMBS


def _outward(host, piece, size, span, W, centre) -> int:
    best, side = -1e9, 1
    for sd in (1, -1):
        teeth = OL.carried(host, piece, sd, size, span, W)
        if teeth:
            d = np.mean([np.hypot(*(t.c[-1] - centre)) for t in teeth])
            if d > best:
                best, side = d, sd
    return side


def reverse_teeth(ch: str, W: float, size: float = 0.55, rings: bool = False) -> list:
    """Reverse, whose heavy bars and arches carry the teeth of real combs (pieces.py teeth, drawn
    by outline.carried) as hairlines, facing out of the letter."""
    import hashlib
    strokes = _reverse(ch, W)
    S = [s for s in strokes if isinstance(s, OL.Stroke)]
    centre = np.vstack([s.c for s in S]).mean(0)
    pool = _comb_pool()
    for i, s in enumerate(S):
        d = s.c[-1] - s.c[0]
        horizontal = (s.kind != "crescent" and (rings or not s.closed)) or (not s.closed and abs(d[1]) < 0.4 * abs(d[0]))
        if not horizontal:
            continue
        h = int(hashlib.md5(f"teeth/{ch}/{i}".encode()).hexdigest()[:8], 16)
        p = pool[h % len(pool)]
        span = (0.12, 0.88)
        side = _outward(s, p, size, span, W, centre)
        s.teeth = [(p, side, size, span)]
    return strokes

PARAMS["reverse-comb"] = PARAMS["reverse"]
_STYLES["reverse-comb"] = lambda ch, W: reverse_teeth(ch, W, size=0.6)


# ── medieval: compound sprigs at the stroke ends ────────────────────────────
# Each free stroke end grows a real sprig or fork: its skeleton (pieces.tree: the spine, and where,
# at what angle and how far each branch leaves it) redrawn as smooth tapering strokes. On a guide
# line it turns sideways, like a serif, its branches curling back into the letter; elsewhere it
# continues out of the stroke, leaning away from the letter.

MED_KINDS = ("straight-sprig", "curved-sprig", "wavy-sprig", "fork")
MED_LEN = 2.3        # spine length, in weights
MED_W = 0.6          # ornament weight at its root, at least this (in weights)
MED_TOOTH = 0.75     # a branch starts this wide relative to the spine there
MED_FAN = 22.0
MED_BRANCH = 2.2     # branch length multiplier       # free-end ornaments lean this many degrees away from the letter
_MED = None


def _med_pool() -> list[dict]:
    global _MED
    if _MED is None:
        import pieces as PC
        lib = PC.library()
        _MED = [p for k in MED_KINDS for p in lib.get(k, []) if 1 <= len(p["teeth"]) <= 3]
    return _MED


def _smooth_path(P: np.ndarray, n: int = 24, sigma: float = 1.6) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    sa = OL.arclen(P)
    if sa[-1] < 1e-9:
        return P
    t = np.linspace(0, sa[-1], n)
    q = np.stack([np.interp(t, sa, P[:, 0]), np.interp(t, sa, P[:, 1])], 1)
    q2 = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    return q2 - q2[0] + q[0]


def _sprig(p: dict, e, head, side: float, scale: float, w0: float, W: float, out=None) -> list:
    """Piece p's skeleton as strokes: spine from e along `head`, branches to `side` (+1 left)."""
    sp = np.asarray(p["spine"], float)
    teeth = p["teeth"]
    # root at the end with fewer branches
    if np.mean([t["u"] for t in teeth]) < 0.5:
        sp = sp[::-1] - sp[-1]
        teeth = [dict(t, u=1 - t["u"], pts=[[-x, -y] for x, y in t["pts"]]) for t in teeth]
    chord = sp[-1] - sp[0]
    a = -math.atan2(chord[1], chord[0])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    sp = (sp - sp[0]) @ R.T
    sp = _smooth_path(sp)
    L = OL.arclen(sp)[-1]
    k = scale / max(L, 1e-9)
    hd = np.asarray(head, float) / np.hypot(*head)
    nd = np.array([-hd[1], hd[0]]) * side
    world = lambda Q: e + np.outer(Q[:, 0] * k, hd) + np.outer(Q[:, 1] * k, nd)
    spine = world(sp)
    if out is not None:
        # the stroke flows into the ornament: the spine starts inside the stroke, travelling the
        # stroke's own direction, and turns toward the ornament's heading in a smooth elbow
        o = np.asarray(out, float)
        bridge = np.array([e - o * 0.6 * W, e - o * 0.25 * W])
        spine = np.vstack([bridge, spine + o * 0.12 * W])
    out = [OL.Stroke(spine, w=w0 / W, taper0=False, taper1=False, belly=0.08, e0w=1.0, e1w=0.06)]
    tg = OL.tangents(sp, False)
    sa = OL.arclen(sp) / max(L, 1e-9)
    for t in teeth:
        tp = np.asarray(t["pts"], float) @ R.T          # (the tooth frame turns with the spine)
        if OL.arclen(tp)[-1] < 0.15:
            continue
        i = int(np.clip(np.searchsorted(sa, t["u"]), 0, len(sp) - 1))
        tt, nn = tg[i], np.array([-tg[i][1], tg[i][0]])
        loc = (tp - tp[0]) * MED_BRANCH                  # branches drawn longer than traced, to read
        if loc[-1] @ nn * np.sign(np.sum(np.asarray(t["pts"])[:, 1]) or 1) < 0:
            pass
        Q = sp[i] + np.outer(loc[:, 0], tt) + np.outer(loc[:, 1], nn)
        Q = _smooth_path(Q, 16, 1.2)
        wt = w0 * MED_TOOTH * (1 - 0.6 * t["u"])
        out.append(OL.Stroke(world(Q), w=wt / W, taper0=False, taper1=False, belly=0.08, e0w=1.0, e1w=0.06))
    for o in out:
        o.no_contrast = True
    return out


def medieval(ch: str, W: float, base: str = "reverse") -> list:
    import hashlib
    strokes = _reverse(ch, W) if base == "reverse" else _base(ch, W)
    S = [s for s in strokes if isinstance(s, OL.Stroke)]
    for s in S:                                          # freeze the letter's own joins
        s.taper0_set, s.taper1_set = s.taper0, s.taper1
    centre = np.vstack([s.c for s in S]).mean(0)
    pool = _med_pool()
    extra = []
    for j, (s, end, e, out) in enumerate(_ends(S)):
        h = int(hashlib.md5(f"med/{ch}/{j}".encode()).hexdigest()[:8], 16)
        p = pool[h % len(pool)]
        w_end = float(OL.widths(s, W)[0 if end == 0 else -1])
        w0 = max(w_end, MED_W * W)
        if _on_line(e, out) and s.kind == "crescent":
            # a serif position: turn sideways, away from the letter's centre, branches back into the letter
            sx = 1.0 if e[0] >= centre[0] else -1.0
            head = np.array([sx, 0.0]) + 0.25 * out
            into = -np.sign(out[1])                       # +1: up from the baseline, −1: down from the top
            side = into * sx                              # left of a rightward heading is up
        else:
            away = np.sign((e - centre) @ np.array([-out[1], out[0]])) or 1.0
            a = math.radians(MED_FAN) * away
            head = np.array([out[0] * math.cos(a) - out[1] * math.sin(a), out[0] * math.sin(a) + out[1] * math.cos(a)])
            side = -away                                  # branches curl back toward the letter
        if end == 0:                                  # the stroke's own end is hidden inside the ornament
            s.e0w = 1.0
        else:
            s.e1w = 1.0
        extra += _sprig(p, e, head, side, MED_LEN * W, w0, W, out=out)
    return strokes + extra


PARAMS["medieval"] = PARAMS["reverse"]
PARAMS["medieval-bare"] = dict(END_JOIN=0.25, END_FREE=0.55, CONTRAST=0.38)
_STYLES["medieval"] = lambda ch, W: medieval(ch, W, "reverse")
_STYLES["medieval-bare"] = lambda ch, W: medieval(ch, W, "bare")


# ── medieval, richer: many-branched combs and feathers, branching again once ──

RICH_KINDS = ("straight-comb", "curved-comb", "wavy-comb", "straight-feather", "curved-feather", "wavy-feather",
              "curved-branching-line", "straight-branching-line")
RICH_MAX_TEETH = 6
RICH_BRANCH = 2.0    # branch length multiplier
RICH_TOOTH = 0.4     # branch root width relative to the spine
RICH_LEN = 2.8       # spine length, in weights
RICH_W = 0.5
RICH_SUB = 0.45       # a sub-branch is this size relative to its parent ornament
_RICH = None


def _rich_pool() -> list[dict]:
    global _RICH
    if _RICH is None:
        import pieces as PC
        lib = PC.library()
        _RICH = [p for k in RICH_KINDS for p in lib.get(k, []) if 3 <= len(p["teeth"]) <= 9]
    return _RICH


def _thin(p: dict) -> dict:
    t = p["teeth"]
    if len(t) > RICH_MAX_TEETH:
        t = [t[int(round(k))] for k in np.linspace(0, len(t) - 1, RICH_MAX_TEETH)]
    return dict(p, teeth=t)


def medieval_rich(ch: str, W: float, depth: int = 1) -> list:
    import hashlib
    global _med_pool
    old_pool, old_branch = _med_pool, MED_BRANCH
    strokes = medieval.__wrapped__(ch, W) if hasattr(medieval, "__wrapped__") else None
    # reuse medieval() with the rich pool and thinned pieces
    pool = [_thin(p) for p in _rich_pool()]
    saved = {k: globals()[k] for k in ("MED_BRANCH", "MED_TOOTH", "MED_LEN", "MED_W")}
    globals()["_med_pool"] = lambda: pool
    globals().update(MED_BRANCH=RICH_BRANCH, MED_TOOTH=RICH_TOOTH, MED_LEN=RICH_LEN, MED_W=RICH_W)
    try:
        strokes = medieval(ch, W, "bare")
    finally:
        globals()["_med_pool"] = old_pool
        globals().update(saved)
    if depth < 1:
        return strokes
    # second generation: at the tip of each long ornament branch, a small sprig of the same kind
    S = [s for s in strokes if isinstance(s, OL.Stroke) and getattr(s, "no_contrast", False)]
    extra = []
    for j, s in enumerate(S):
        P = s.P0
        L = OL.arclen(P)[-1]
        if L < 0.9 * W or s.w * W > 0.3 * W:            # only the branches (not spines), and only long ones
            continue
        h = int(hashlib.md5(f"rich/{ch}/{j}".encode()).hexdigest()[:8], 16)
        p = pool[h % len(pool)]
        tip = P[-1]
        d = P[-1] - P[-4]
        d = d / max(np.hypot(*d), 1e-9)
        side = 1.0 if h % 2 else -1.0
        extra += _sprig(p, tip - d * 0.05 * W, d, side, RICH_SUB * MED_LEN * W, 0.5 * s.w * W, W)
    return strokes + extra


PARAMS["medieval-rich"] = PARAMS["medieval-bare"]
_STYLES["medieval-rich"] = lambda ch, W: medieval_rich(ch, W, 1)
_STYLES["medieval-rich0"] = lambda ch, W: medieval_rich(ch, W, 0)
PARAMS["medieval-rich0"] = PARAMS["medieval-bare"]


# ── crowned: whole compound pieces, at the letter's own weight ──────────────
# The actual outline of a comb or feather (silhouettes.py, smoothed but not reshaped), scaled so
# its lines are about as thick as the letter's strokes, growing from the letter's highest
# upward-pointing ends. It may rise well beyond the letter; counters stay open.

CROWN_KINDS = ("straight-comb", "straight-feather", "curved-feather", "curved-comb", "wavy-feather", "wavy-comb")
CROWN_ASPECT = (5.0, 15.0)    # pieces with length/thickness in this range
CROWN_THICK = 0.62            # piece line thickness, in weights
CROWN_LEN = (260.0, 430.0)    # piece length is kept in this range (font units)
CROWN_N = 2                   # at most this many per letter
CROWN_LEAN = 12.0             # degrees, away from the letter's centre
_CROWN = None


def _crown_pool() -> list[str]:
    global _CROWN
    if _CROWN is None:
        import pieces as PC
        import silhouettes as SI
        _CROWN = []
        for k in CROWN_KINDS:
            for f in sorted((PC.ROOT / k).glob("[0-9][0-9].svg")):
                ref = f"{k}/{int(f.stem)}"
                try:
                    s = SI.silhouette(ref)
                except Exception:
                    continue
                if CROWN_ASPECT[0] <= s["aspect"] <= CROWN_ASPECT[1] and len(s["tips"]) >= 3:
                    _CROWN.append(ref)
    return _CROWN


def _crown_piece(ref: str, e, head, side: float, W: float, centre) -> list[np.ndarray]:
    import silhouettes as SI
    s = SI.silhouette(ref)
    C = [c.copy() for c in s["contours"]]
    P = np.vstack(C)
    # frame: the piece's own long axis (principal axis of its outline), from 0 to 1 along it
    mu = P.mean(0)
    ev, V = np.linalg.eigh(np.cov((P - mu).T))
    ax, nx = V[:, 1], V[:, 0]
    C = [np.stack([(c - mu) @ ax, (c - mu) @ nx], 1) for c in C]
    P = np.vstack(C)
    lo, hi = P[:, 0].min(), P[:, 0].max()
    C = [np.stack([(c[:, 0] - lo) / (hi - lo), c[:, 1] / (hi - lo)], 1) for c in C]
    P = np.vstack(C)
    spread = lambda m: np.ptp(P[m, 1]) if m.any() else 0.0
    root_at_0 = spread(P[:, 0] < 0.25) <= spread(P[:, 0] > 0.75)      # the narrower end joins the letter
    if not root_at_0:
        C = [np.stack([1 - c[:, 0], -c[:, 1]], 1) for c in C]
    L = float(np.clip(s["aspect"] * CROWN_THICK * W, *CROWN_LEN))
    hd = np.asarray(head, float) / np.hypot(*head)
    nd = np.array([-hd[1], hd[0]]) * side
    placed = [e + np.outer(c[:, 0] * L, hd) + np.outer(c[:, 1] * L, nd) for c in C]
    # snap: the piece's point nearest its root end overlaps the stroke's end
    Q = np.vstack(placed)
    base = Q[np.argsort((Q - e) @ hd)[:max(4, len(Q) // 40)]].mean(0)
    shift = (e - hd * 0.25 * W) - base
    return [c + shift for c in placed]


CROWN_MAX = (3, 2)            # ornaments per capital, per lowercase letter
CROWN_SCALE = {"top": 1.0, "terminal": 0.85, "arm": 0.8, "foot": 0.55}
CROWN_ORDER = ("top", "terminal", "arm", "foot")


def _rot(v, deg):
    a = math.radians(deg)
    return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])


def crowned(ch: str, W: float) -> list[np.ndarray]:
    import hashlib
    import fuse as FU
    strokes = _base(ch, W)
    bare = OL.outline(strokes, W)
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    counters = [C for C in bare if area(C) < 0]
    S = [s for s in strokes if isinstance(s, OL.Stroke)]
    centre = np.vstack([s.c for s in S]).mean(0)

    def upright(s):
        d = s.c[-1] - s.c[0]
        return abs(d[0]) < 0.12 * abs(d[1])

    places = []                                   # (kind, e, head, how to pick the side)
    for s, end, e, out in _ends(S):
        sx = 1.0 if e[0] >= centre[0] else -1.0
        if s.kind == "crescent" and upright(s) and out[1] > 0.8:
            places.append(("top", e, _rot(out, CROWN_LEAN * -sx), "away"))
        elif s.kind == "crescent" and upright(s) and out[1] < -0.8 and abs(e[1]) < 45:
            places.append(("foot", e, np.array([sx, 0.0]), "up"))
        elif (s.kind == "crescent" and abs(out[0]) > 0.8) or s.kind != "crescent":
            # arms and terminals: no sideways ornament (it would push the letters apart); in the upper
            # part of the letter it turns up steeply, at the baseline it becomes a small foot
            kind = "arm" if s.kind == "crescent" else "terminal"
            top = max(z.c[:, 1].max() for z in S)
            if e[1] > 0.62 * top:
                hx = np.sign(out[0]) or sx
                places.append((kind, e, np.array([0.35 * hx, 1.0]), "away"))
            elif e[1] < 0.2 * top:
                places.append(("foot", e, np.array([np.sign(out[0]) or sx, 0.0]), "up"))
    places.sort(key=lambda z: (CROWN_ORDER.index(z[0]), -z[1][1]))
    if ch.islower():                               # at most one foot on a lowercase letter
        feet = [z for z in places if z[0] == "foot"]
        places = [z for z in places if z[0] != "foot"] + feet[:1]
    places = places[:CROWN_MAX[1 if ch.islower() else 0]]
    grafts = [(bare, [])]
    pool = _crown_pool()
    for j, (kind, e, head, rule) in enumerate(places):
        h = int(hashlib.md5(f"crown/{ch}/{j}".encode()).hexdigest()[:8], 16)
        ref = pool[h % len(pool)]
        best = None
        for side in (1.0, -1.0):
            pc = _crown_piece(ref, e, head, side, W * CROWN_SCALE[kind], centre)
            Q = np.vstack(pc)
            score = Q[:, 1].mean() if rule == "up" else np.mean(np.hypot(*(Q - centre).T))
            if best is None or score > best[0]:
                best = (score, pc)
        pc = best[1]
        if kind == "foot":                        # a foot ornament sits on the baseline, not below it
            lift = max(0.0, -np.vstack(pc)[:, 1].min() - 0.02 * W)
            pc = [c + [0.0, lift] for c in pc]
        grafts.append((pc, []))
    return FU.raster_glyph([], [], grafts, [], W, counters=counters, inset=0.0, smooth=1.2,
                           clean=0.04 * W, contour_smooth=0.05 * W)


PARAMS["crowned"] = PARAMS["medieval-bare"]
