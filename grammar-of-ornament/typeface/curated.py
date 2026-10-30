"""
Neuron Curated — letters drawn from hand-chosen compound pieces.

Each letter is a recipe (RECIPES): a few components, each one a stretch of a
specific compound piece's skeleton (pieces.tree: its spine and its teeth),
chosen by eye for that letter. A component is laid either

  along      a target centreline (a Bare skeleton stroke or a drawn curve): the
             piece's spine stretch a..b runs along it, its teeth leave it where
             they left the spine, at the same angles, curves and proportions
  as is      the piece's own spine geometry, moved, turned, scaled (and maybe
             mirrored) into the letter by a similarity (place): the letter's
             curve is the compound's curve

and redrawn as clean strokes: the spine at the letter's weight with the piece's
own swelling and pinching, the teeth as tapering strokes (outline.py).

  .venv/bin/python typeface/curated.py sheet REF [REF…]   → proofs/curated/pieces.png (skeletons, numbered)
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import outline as OL        # noqa: E402
import pieces as PC         # noqa: E402
import silhouettes as SI    # noqa: E402

N_SPINE = 60
SPINE_SMOOTH = 0.035      # the traced spine is smoothed over this share of its length (its wobble goes, its curve stays)


def _even(P: np.ndarray, n: int) -> np.ndarray:
    sa = OL.arclen(P)
    q = np.linspace(0, sa[-1], n)
    return np.stack([np.interp(q, sa, P[:, 0]), np.interp(q, sa, P[:, 1])], 1)


def _smooth(P: np.ndarray, sig: float) -> np.ndarray:
    """Gaussian smoothing of an open polyline, its end points held in place."""
    from scipy import ndimage as ndi
    if sig <= 0 or len(P) < 4:
        return P
    Q = ndi.gaussian_filter1d(P, sig, axis=0, mode="nearest")
    n = len(P)
    k = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / (2 * sig), 0, 1)[:, None]
    Q = k * Q + (1 - k) * P
    Q[0], Q[-1] = P[0], P[-1]
    return Q


@lru_cache(maxsize=None)
def skel(ref: str) -> dict:
    """A piece's skeleton in its own geometry, in units of its spine length, spine starting at (0, 0):
    spine (N×2, evenly spaced), its width profile, and its teeth (u along the spine, global points, width)."""
    kind, idx = ref.rsplit("/", 1)
    t = PC.tree(kind, int(idx))
    sp = np.asarray(t["spine"], float)
    sa = OL.arclen(sp)
    L = sa[-1]
    q = np.linspace(0, L, N_SPINE)
    spine = np.stack([np.interp(q, sa, sp[:, 0]), np.interp(q, sa, sp[:, 1])], 1) / L
    spine = _smooth(spine, SPINE_SMOOTH * N_SPINE)
    tg = OL.tangents(spine, False)
    teeth = []
    for tooth in t["teeth"]:
        i = int(round(tooth["u"] * (N_SPINE - 1)))
        tt = tg[i]
        nn = np.array([-tt[1], tt[0]])
        xy = _smooth(_even(np.asarray(tooth["pts"], float), 16), 1.5)
        teeth.append({"u": tooth["u"], "pts": spine[i] + np.outer(xy[:, 0], tt) + np.outer(xy[:, 1], nn),
                      "w": tooth["w"]})
    try:
        prof = SI.silhouette(ref)["profile"]
    except Exception:
        prof = np.ones(10)
    return {"spine": spine, "teeth": teeth, "w": t["spine_w_rel"], "prof": prof}


# ── laying pieces into letters ──────────────────────────────────────────────────────────────────────
TOOTH_W = (0.3, 0.6)      # tooth root widths, relative to the letter's weight
PLACED_W = 1.12           # pieces drawn as whole letters run a little heavier (their curves lose more to contrast)
TOOTH_TIP = 0.45          # tooth tip width, relative to its root


def _frame(sp: np.ndarray, i: int):
    tg = OL.tangents(sp, False)[i]
    return tg, np.array([-tg[1], tg[0]])


def _tooth_stroke(P: np.ndarray, w: float) -> OL.Stroke:
    keep = [0] + [j for j in range(1, len(P)) if np.hypot(*(P[j] - P[j - 1])) > 3]
    if len(keep) < 3:
        return None
    st = OL.Stroke(P[keep], w=w, taper0=False, taper1=False, belly=0.15, e0w=1.0, e1w=TOOTH_TIP)
    st.no_contrast = True
    return st


def teeth_on(target, ref: str, a: float = 0.0, b: float = 1.0, which=None, flip: bool = False,
             rev: bool = False, scale: float = 1.0, w: float = 1.0, maxlen: float | None = None, W: float = 100.0,
             on=(0.0, 1.0)):
    """The teeth of piece `ref` in its spine stretch a..b, laid along target (a Stroke or a centreline):
    each leaves the target where it left the spine, at the same angle and curve, its length in the same
    proportion to the stretch (× scale). on: the part of the target the stretch is laid along. rev runs the stretch backwards along the target; flip mirrors
    the teeth across it. which: indices of the teeth to keep (default all in the stretch)."""
    c = target.c if isinstance(target, OL.Stroke) else np.asarray(target, float)
    sa0 = OL.arclen(c)
    keep = (sa0 >= on[0] * sa0[-1] - 1e-6) & (sa0 <= on[1] * sa0[-1] + 1e-6)
    c = c[keep]
    sk = skel(ref)
    sp = sk["spine"]
    sa = OL.arclen(c)
    L = sa[-1]
    tc = OL.tangents(c, False)
    sel = [t for j, t in enumerate(sk["teeth"]) if (j in which if which is not None else a <= t["u"] <= b)]
    med = float(np.median([OL.arclen(t["pts"])[-1] for t in sel])) if sel else 1.0
    k = scale * W / max(med, 1e-6)            # the median tooth is `scale` weights long, the others in proportion
    out = []
    for j, t in enumerate(sk["teeth"]):
        if which is not None and j not in which:
            continue
        if which is None and not (a <= t["u"] <= b):
            continue
        v = np.clip((t["u"] - a) / (b - a), 0, 1)
        if rev:
            v = 1 - v
        i = int(round(t["u"] * (N_SPINE - 1)))
        tp, npn = _frame(sp, i)
        rel = t["pts"] - sp[i]
        x, y = rel @ tp, rel @ npn
        if rev:
            x, y = -x, -y
        if flip:
            y = -y
        ci = int(np.clip(np.searchsorted(sa, v * L), 0, len(c) - 1))
        tt = tc[ci]
        nn = np.array([-tt[1], tt[0]])
        xy = np.stack([x, y], 1) * k
        if maxlen:
            tl = OL.arclen(xy)[-1]
            if tl > maxlen * W:
                xy *= maxlen * W / tl
        P = c[ci] + np.outer(xy[:, 0], tt) + np.outer(xy[:, 1], nn)
        out.append(_tooth_stroke(P, float(np.clip(t["w"], *TOOTH_W)) * w))
    return [o for o in out if o is not None]


def placed_box(ref: str, box, a: float = 0.0, b: float = 1.0, rot: float = 0.0, mirror: bool = False,
               stretch: bool = True, w: float = PLACED_W, teeth=True, which=None, tw: float = 1.0,
               tlen: float = 1.0, hflip: bool = False, **kw) -> list:
    """The piece's own spine stretch a..b (and its teeth), mirrored, turned by rot degrees, and fitted
    to box (x0, y0, x1, y1) (stretched to fill it, or with its proportions kept)."""
    sk = skel(ref)
    sp = sk["spine"]
    ia, ib = int(round(a * (N_SPINE - 1))), int(round(b * (N_SPINE - 1)))
    z = sp[:, 0] + 1j * sp[:, 1]
    tz = []
    for j, t in enumerate(sk["teeth"]):
        if not teeth or (which is not None and j not in which) or (which is None and not (a <= t["u"] <= b)):
            continue
        tz.append((t, t["pts"][:, 0] + 1j * t["pts"][:, 1]))
    def rotz(q):  # noqa: E306
        q = (np.conj(q) if mirror else q) * np.exp(1j * np.radians(rot))
        return -np.conj(q) if hflip else q                 # hflip: left and right swapped, after turning
    seg = rotz(z[ia:ib + 1])
    tz = [(t, rotz(q)) for t, q in tz]
    allz = np.concatenate([seg] + [q for _, q in tz])
    lo, hi = np.array([allz.real.min(), allz.imag.min()]), np.array([allz.real.max(), allz.imag.max()])
    x0, y0, x1, y1 = box
    sy = (y1 - y0) / max(hi[1] - lo[1], 1e-9)
    if x1 is None:                             # width from the piece's own proportions (× stretch, if a number)
        sx = sy * (stretch if not isinstance(stretch, bool) else 1.0)
        x1 = x0 + sx * (hi[0] - lo[0])
    else:
        sx = (x1 - x0) / max(hi[0] - lo[0], 1e-9)
        if stretch is False:
            sx = sy = min(sx, sy)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    mc = (lo + hi) / 2
    f = lambda q: np.stack([cx + (q.real - mc[0]) * sx, cy + (q.imag - mc[1]) * sy], 1)
    out = [OL.Stroke(f(seg), w=w, **kw)]
    for t, q in tz:
        P = f(q)
        P = P[0] + (P - P[0]) * tlen
        o = _tooth_stroke(P, float(np.clip(t["w"], *TOOTH_W)) * w * tw)
        if o is not None:
            o.free = True                      # a spur hangs into the space: it doesn't set the letter's size or spacing
        out.append(o)
    return [o for o in out if o is not None]


def placed(ref: str, a: float, b: float, p0, p1, mirror: bool = False, w: float = 1.0, teeth=True,
           which=None, tscale: float = 1.0, **kw) -> list:
    """The piece's own spine stretch a..b (and its teeth) moved, turned and scaled so that spine(a)
    lands on p0 and spine(b) on p1 (mirror: reflected first). Returns strokes."""
    sk = skel(ref)
    sp = sk["spine"]
    z = sp[:, 0] + 1j * sp[:, 1]
    if mirror:
        z = np.conj(z)
    ia, ib = int(round(a * (N_SPINE - 1))), int(round(b * (N_SPINE - 1)))
    za, zb = z[ia], z[ib]
    q0, q1 = complex(*p0), complex(*p1)
    m = (q1 - q0) / (zb - za)
    f = lambda zz: q0 + m * (zz - za)
    seg = f(z[ia:ib + 1])
    out = [OL.Stroke(np.stack([seg.real, seg.imag], 1), w=w, **kw)]
    if teeth:
        for j, t in enumerate(sk["teeth"]):
            if which is not None and j not in which:
                continue
            if which is None and not (a <= t["u"] <= b):
                continue
            tz = t["pts"][:, 0] + 1j * t["pts"][:, 1]
            if mirror:
                tz = np.conj(tz)
            root = f(z[int(round(t["u"] * (N_SPINE - 1)))])
            tz = f(tz)
            tz = root + (tz - tz[0]) * tscale
            out.append(_tooth_stroke(np.stack([tz.real, tz.imag], 1), float(np.clip(t["w"], *TOOTH_W)) * w))
    return out


# ── the letters ─────────────────────────────────────────────────────────────────────────────────────
import copy                  # noqa: E402

import dress as DR           # noqa: E402
import glyphs as GL          # noqa: E402

RECIPES: dict = {}


def recipe(chars: str):
    def deco(fn):
        for ch in chars:
            RECIPES[ch] = fn
        return fn
    return deco


# Alignment: ink edges (not centrelines) land on these lines; round forms overshoot by OVER.
OVER = 12
TOP_ROUND = set("BCDGOPQRSTabcdefghmnopqrs035689")
BOT_ROUND = set("BCDEGJOQSUabcdeostu035689")
TOPS = {**{c: GL.ASC for c in "bdfhkl"}, **{c: GL.XH for c in "acegimnopqrsuvwxyzj"}, "t": 640}
BOTTOMS = {c: GL.DESC for c in "gjpqy"}
# ink widths: the letters' proportions, set together (absent: keep the recipe's)
WIDTHS = dict(zip("abcdefghjkmnopqrstuvwxyz",
                  (470, 480, 400, 480, 420, 330, 470, 470, 240, 440, 720, 470, 470, 480, 480, 340, 390, 290, 470, 440,
                   660, 430, 450, 400)))
WIDTHS.update(zip("ABCDEFGHJKLMNOPQRSTUVWXYZ",
                  (640, 520, 560, 600, 450, 440, 600, 590, 330, 540, 430, 800, 620, 640, 500, 640, 540, 510, 520, 600,
                   600, 860, 580, 560, 520)))
WIDTHS.update(zip("0123456789", (500, 280, 470, 440, 500, 460, 490, 460, 470, 490)))


def _ink(strokes, W):
    P = np.vstack(OL.outline(strokes, W))
    return P.min(0), P.max(0)


def fit(ch: str, S: list, W: float) -> None:
    """Move and scale the centrelines (strokes keep their widths) so the ink lands on the letter's lines
    and takes its width. Strokes marked `free` (a Q's tail) and dots go along without being measured."""
    body = [s for s in S if not isinstance(s, OL.Dot) and not getattr(s, "free", False)]
    top = TOPS.get(ch, GL.CAP) + (OVER if ch in TOP_ROUND else 0)
    bot = BOTTOMS.get(ch, 0) - (OVER if ch in BOT_ROUND else 0)
    wt = WIDTHS.get(ch)
    for _ in range(4):
        lo, hi = _ink(body, W)
        sy = (top - bot - W) / max(hi[1] - lo[1] - W, 1e-6)
        sx = (wt - W) / max(hi[0] - lo[0] - W, 1e-6) if wt else 1.0
        for s in S:
            if isinstance(s, OL.Dot):
                s.x = lo[0] + (s.x - lo[0]) * sx
                continue
            s.P0 = np.stack([lo[0] + (s.P0[:, 0] - lo[0]) * sx, bot + (s.P0[:, 1] - lo[1]) * sy], 1)
            s.prepare(W)


def _fitted(ch: str, W: float) -> list:
    S = copy.deepcopy(GL.GLYPHS[ch])
    DR.dress(ch, S, W, None, None)
    if ch in RECIPES:
        S = RECIPES[ch](ch, S, W)
    fit(ch, S, W)
    return S


def glyph(ch: str, W: float) -> list[np.ndarray]:
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    return [C for C in OL.outline(_fitted(ch, W), W) if abs(area(C)) > 200]      # no specks where strokes graze


def body(ch: str, W: float) -> list[np.ndarray]:
    """The letter without its spur or tail: what the spacing is measured on."""
    return OL.outline([s for s in _fitted(ch, W) if not getattr(s, "free", False)], W)


# Weight: every letter's main strokes are brought toward its reference letter's, measured on these
# letters themselves (n for lowercase, H for capitals and figures).
_WEIGHTS: dict = {}


def weight(ch: str, main_stroke) -> float:
    if ch not in _WEIGHTS:
        base = GL.W_LOW if ch.islower() else GL.W_CAP
        ref = main_stroke(glyph("n" if ch.islower() else "H", base))
        f = ref / main_stroke(glyph(ch, base))
        _WEIGHTS[ch] = base * float(np.clip(f, 0.8, 1.25)) ** 0.8
    return _WEIGHTS[ch]


# crest: comb teeth standing up along the top of an arch (curved-comb/8's upright teeth)
def crest(stroke, W, on=(0.12, 0.6), scale=1.1, flip=False, rev=True):
    return teeth_on(stroke, "curved-comb/8", 0.72, 1.0, which=[9, 10, 11, 12, 13], rev=rev, flip=flip,
                    scale=scale, w=0.42, W=W, on=on)


# fringe: straight-comb/12's long row of teeth hanging from a descender
def fringe(stroke, W, on=(0.4, 1.0), flip=False, scale=1.0, rev=False):
    return teeth_on(stroke, "straight-comb/12", 0.2, 1.0, which=list(range(3, 16)), flip=flip, rev=rev,
                    scale=scale, w=0.34, W=W, on=on)


XH = GL.XH
ARCH = ("curved-branching-line/3", 300)     # the n: an arch with a spur on its left leg


SPUR = 0.55                                  # the spur is the arch's own branch, cut back to this share


def _arch(x0, rot=ARCH[1], h=XH, y0=0, **kw):
    kw.setdefault("tlen", SPUR)
    return placed_box(ARCH[0], (x0, y0, None, y0 + h), rot=rot, **kw)


def _xmax(strokes):
    return max(float(s.c[:, 0].max()) for s in strokes if isinstance(s, OL.Stroke))


@recipe("n")
def _n(ch, S, W):
    return _arch(40)


def _sprout(base: OL.Stroke, at: float, piece: list, a: float, W: float) -> list:
    """Piece (its spine stroke first) cut to start at a (fraction along its spine), moved so that start
    lands on base at fraction `at`: a second arch springing from the first one's leg."""
    sp = piece[0].c
    i = int(a * (len(sp) - 1))
    j = int(at * (len(base.c) - 1))
    d = base.c[j] - sp[i]
    Q = sp[i:] + d
    y0, yb = Q[0, 1], float(base.c[:, 1].min())
    Q[:, 1] = y0 + (Q[:, 1] - y0) * (y0 - yb) / max(y0 - Q[:, 1].min(), 1e-6)     # its leg lands on the same line
    return [OL.Stroke(Q, w=piece[0].w)]


@recipe("m")
def _m(ch, S, W):
    a = _arch(40, stretch=0.72)
    return a + _sprout(a[0], 0.66, _arch(40, stretch=0.72, teeth=False), 0.36, W)


@recipe("h")
def _h(ch, S, W):
    # the arch springs from the ascender (no second left leg: it's the stem)
    x = 70.0
    stem = GL.C((x, 0), (x, GL.ASC))
    stem.prepare(W)
    return [stem] + _sprout(stem, 0.62, _arch(40, teeth=False), 0.33, W)


@recipe("u")
def _u(ch, S, W):
    return _arch(40, rot=ARCH[1] - 180)


@recipe("e")
def _e(ch, S, W):
    # fork/1 on its side: a c whose prong is the bar (the uncial e)
    return placed_box("fork/1", (50, 0, None, XH), rot=265, tlen=1.45, tw=1.3)


@recipe("c")
def _c(ch, S, W):
    # wavy-sprig/6 turned: a C whose lower end rolls into a curl
    return placed_box("wavy-sprig/6", (50, 0, None, XH), rot=195, stretch=0.8)


BOWL = ("curved-sprig/13", 0.0, 0.75)     # the spiral cut to one turn: a ring whose end tucks inside


def _bowl(x0, y0=0, h=XH, stretch=0.85, rot=0, mirror=False):
    return placed_box(BOWL[0], (x0, y0, None, y0 + h), a=BOWL[1], b=BOWL[2], rot=rot, mirror=mirror,
                      stretch=stretch, teeth=False)


def _stem(x, y0, y1, W):
    st = GL.C((x, y0), (x, y1))
    st.prepare(W)
    return st


HOOK = ("wavy-band/2", 0)                   # a j: straight down, curling left at the bottom


def _hook(x_top, y_top, y_bot, W, mirror=False):
    """The hook, placed so its top end is at (x_top, y_top)."""
    h = placed_box(HOOK[0], (0, y_bot, None, y_top), rot=HOOK[1], mirror=mirror, stretch=1.0, teeth=False)
    c = h[0].c
    top = c[np.argmax(c[:, 1])]
    return [OL.Stroke(c + [x_top - top[0], 0], w=h[0].w)]


@recipe("o")
def _o(ch, S, W):
    return _bowl(40)


@recipe("b")
def _b(ch, S, W):
    x = 80.0
    return [_stem(x, 0, GL.ASC, W)] + _bowl(x - 0.35 * W)


@recipe("p")
def _p(ch, S, W):
    x = 80.0
    return [_stem(x, GL.DESC, XH, W)] + _bowl(x - 0.35 * W)


@recipe("dqa")
def _dq(ch, S, W):
    bowl = _bowl(40, rot=180)
    x = _xmax(bowl) - 0.35 * W
    y0, y1 = {"d": (0, GL.ASC), "q": (GL.DESC, XH), "a": (0, XH)}[ch]
    return bowl + [_stem(x, y0, y1, W)]


@recipe("g")
def _g(ch, S, W):
    bowl = _bowl(40, rot=180)
    x = _xmax(bowl) - 0.35 * W
    return bowl + _hook(x, XH, GL.DESC, W)


@recipe("j")
def _j(ch, S, W):
    return _hook(250, XH, GL.DESC, W) + [OL.Dot(250, 668)]


@recipe("i")
def _i(ch, S, W):
    return [_stem(115, 0, XH, W), OL.Dot(115, 668)]


@recipe("l")
def _l(ch, S, W):
    return [_stem(115, 0, GL.ASC, W)]


@recipe("t")
def _t(ch, S, W):
    h = placed_box(HOOK[0], (60, 0, None, 650), rot=HOOK[1] + 180, mirror=True, stretch=1.0, teeth=False)
    c = _smooth(_even(h[0].c, 60), 7)               # the hook's wobble settled: t needs a calm stem
    h = [OL.Stroke(c, w=h[0].w)]
    xt = float(c[np.argmax(c[:, 1]), 0])
    bar = GL.C((xt - 120, 470), (xt + 150, 470), w=0.78)
    return h + [bar]


@recipe("f")
def _f(ch, S, W):
    f = placed_box("wavy-band/1", (60, 0, None, GL.ASC), rot=105, stretch=0.86, teeth=False)
    c = f[0].c
    i = int(np.argmin(np.abs(c[:, 1] - 470)))
    xb = float(c[i, 0])
    return f + [GL.C((xb - 125, 470), (xb + 140, 470), w=0.78)]


@recipe("r")
def _r(ch, S, W):
    return _arch(40, b=0.66)


@recipe("k")
def _k(ch, S, W):
    # the e's crescent turned to face the stem: its prong reaches back to it, its horns are the arms
    x = 80.0
    arms = placed_box("fork/1", (x, 0, None, XH + 30), rot=265, teeth=False, stretch=1.2)  # the c, its belly on the stem
    return [_stem(x, 0, GL.ASC, W)] + arms


def _ess(h, stretch):
    """curved-sprig/3 as an S, its top carried on over to the right (the piece stops short there)."""
    st = placed_box("curved-sprig/3", (40, 0, None, h), rot=330, stretch=stretch, teeth=False)[0]
    c = st.P0
    top_first = c[0, 1] > c[-1, 1]
    q = c[::-1] if top_first else c
    d = q[-1] - q[-5]
    a = np.arctan2(d[1], d[0])
    step = 0.17 * h / 6
    ext = [q[-1]]
    for da in np.radians(np.linspace(-4, -30, 7)):       # keeps turning clockwise, over the top and down
        a += da
        ext.append(ext[-1] + step * np.array([np.cos(a), np.sin(a)]))
    q = np.vstack([q, ext[1:]])
    return [OL.Stroke(q[::-1] if top_first else q, w=st.w)]


@recipe("s")
def _s(ch, S, W):
    return _ess(XH, 1.25)


VEE = ("curved-band/0", 90)                 # a V, one arm swelling


@recipe("v")
def _v(ch, S, W):
    return placed_box(VEE[0], (30, 0, None, XH), rot=VEE[1], mirror=True, stretch=0.75)


@recipe("w")
def _w(ch, S, W):
    v = placed_box(VEE[0], (30, 0, None, XH), rot=VEE[1], mirror=True, stretch=0.6)
    return v + placed_box(VEE[0], (_xmax(v) - 1.1 * W, 0, None, XH), rot=VEE[1], mirror=True, stretch=0.6)


@recipe("x")
def _x(ch, S, W):
    # loop/11 from past its hairpin (which left a hole): two strokes crossing
    return placed_box("loop/11", (30, 0, None, XH), a=0.27, rot=240, stretch=1.0)


@recipe("y")
def _y(ch, S, W):
    # the u with the g's hook for its right leg
    u = _arch(40, rot=ARCH[1] - 180, teeth=False)
    return u + _hook(_xmax(u) - 0.1 * W, XH, GL.DESC, W)


@recipe("z")
def _z(ch, S, W):
    return [_bar(60, 400, XH - 0.3 * W), GL.C((70, 0), (400, XH - 0.3 * W)), _bar(70, 420, 0.3 * W)]


@recipe("6")
def _6(ch, S, W):
    return placed_box("curved-sprig/13", (40, 0, None, GL.CAP), a=0.1, rot=180, w=0.9, stretch=0.85)


@recipe("9")
def _9(ch, S, W):
    return placed_box("curved-sprig/13", (40, 0, None, GL.CAP), a=0.1, rot=0, w=0.9, stretch=0.85)


# ── capitals: the same parts at cap height (uncial forms where a part makes a better capital) ─────────
CAP = GL.CAP


def _bar(x0, x1, y, w=0.8):
    return GL.C((x0, y), (x1, y), w=w)


@recipe("A")
def _A(ch, S, W):
    v = placed_box(VEE[0], (30, 0, None, CAP), rot=VEE[1] + 180, stretch=0.75)
    x1 = _xmax(v)
    return v + [_bar(30 + 0.1 * (x1 - 30), x1 - 0.12 * (x1 - 30), 250)]


@recipe("B")
def _B(ch, S, W):
    x = 90.0
    def lobe(y0, h):
        return placed_box("fork/1", (x - 0.2 * W, y0, None, y0 + h), rot=265, hflip=True, teeth=False, stretch=1.5)
    return [_stem(x, 0, CAP, W)] + lobe(CAP * 0.5, CAP * 0.5) + lobe(0, CAP * 0.56)


@recipe("C")
def _C(ch, S, W):
    return placed_box("wavy-sprig/6", (50, 0, None, CAP), rot=195, stretch=0.8)


@recipe("D")
def _D(ch, S, W):
    x = 90.0
    # the C's curl turned to face the stem
    bowl = placed_box("wavy-sprig/6", (x - 0.45 * W, 0, None, CAP), rot=195, hflip=True, stretch=0.95)
    c = bowl[0].c
    end = 0 if c[0, 0] < c[-1, 0] else len(c) - 1      # the curl's tip would pile up on the stem: it's cut back
    cut = int(0.16 * len(c))
    bowl[0] = OL.Stroke(c[cut:] if end == 0 else c[:-cut], w=bowl[0].w)
    return [_stem(x, 0, CAP, W)] + bowl


@recipe("E")
def _E(ch, S, W):
    return placed_box("fork/1", (50, 0, None, CAP), rot=265, tlen=1.3, tw=1.3, stretch=0.85)


@recipe("F")
def _F(ch, S, W):
    x = 90.0
    return [_stem(x, 0, CAP, W), _bar(x, x + 330, CAP - 0.3 * W), _bar(x, x + 260, 370)]


@recipe("G")
def _G(ch, S, W):
    c = placed_box("wavy-sprig/6", (50, 0, None, CAP), rot=195, stretch=0.8)
    x1 = _xmax(c)
    # the curl is cut off, and the curve instead rises and turns in as the bar: one stroke
    q = c[0].c
    cut = int(0.16 * len(q))
    q = q[cut:][::-1] if q[0, 0] > q[-1, 0] else q[:-cut]
    e = q[-1]
    return [OL.Stroke(np.vstack([q[::6], [[e[0] + 30, 170], [e[0] + 20, 290], [e[0] - 170, 305]]]), w=c[0].w)]


@recipe("H")
def _H(ch, S, W):
    return [_stem(90, 0, CAP, W), _stem(500, 0, CAP, W), _bar(90, 500, 360)]


@recipe("I")
def _I(ch, S, W):
    return [_stem(120, 0, CAP, W)]


@recipe("J")
def _J(ch, S, W):
    return _hook(380, CAP, -40, W)


@recipe("K")
def _K(ch, S, W):
    x = 90.0
    arms = placed_box("fork/1", (x, 0, None, CAP + 30), rot=265, teeth=False, stretch=1.2)
    return [_stem(x, 0, CAP, W)] + arms


@recipe("L")
def _L(ch, S, W):
    x = 90.0
    return [_stem(x, 0, CAP, W), _bar(x, x + 360, 0.3 * W)]


@recipe("M")
def _M(ch, S, W):
    a = _arch(40, h=CAP, stretch=0.72)
    return a + _sprout(a[0], 0.66, _arch(40, h=CAP, stretch=0.72, teeth=False), 0.36, W)


@recipe("N")
def _N(ch, S, W):
    return _arch(40, h=CAP, stretch=0.9)


@recipe("O")
def _O(ch, S, W):
    return _bowl(40, h=CAP, stretch=0.8)


@recipe("P")
def _P(ch, S, W):
    x = 90.0
    return [_stem(x, 0, CAP, W)] + _bowl(x - 0.35 * W, y0=CAP * 0.4, h=CAP * 0.6, stretch=1.0)


@recipe("Q")
def _Q(ch, S, W):
    o = _bowl(40, h=CAP, stretch=0.8)
    x1 = _xmax(o)
    tail = GL.C((x1 - 210, 170), (x1 + 50, -110), w=0.8)
    tail.free = True
    return o + [tail]


@recipe("R")
def _R(ch, S, W):
    x = 90.0
    bowl = _bowl(x - 0.35 * W, y0=CAP * 0.4, h=CAP * 0.6, stretch=1.0)
    return [_stem(x, 0, CAP, W)] + bowl + [GL.C((x + 150, CAP * 0.42), (_xmax(bowl) + 40, 0))]


@recipe("S")
def _S(ch, S, W):
    return _ess(CAP, 1.2)


@recipe("T")
def _T(ch, S, W):
    x = 300.0
    return [_stem(x, 0, CAP, W), _bar(x - 250, x + 250, CAP - 0.3 * W)]


@recipe("U")
def _U(ch, S, W):
    return _arch(40, rot=ARCH[1] - 180, h=CAP, stretch=0.9)


@recipe("V")
def _V(ch, S, W):
    return placed_box(VEE[0], (30, 0, None, CAP), rot=VEE[1], mirror=True, stretch=0.75)


@recipe("W")
def _W(ch, S, W):
    v = placed_box(VEE[0], (30, 0, None, CAP), rot=VEE[1], mirror=True, stretch=0.6)
    return v + placed_box(VEE[0], (_xmax(v) - 1.1 * W, 0, None, CAP), rot=VEE[1], mirror=True, stretch=0.6)


@recipe("X")
def _X(ch, S, W):
    return placed_box("loop/11", (30, 0, None, CAP), a=0.27, rot=240, stretch=1.0)


@recipe("Y")
def _Y(ch, S, W):
    v = placed_box(VEE[0], (30, 300, None, CAP), rot=VEE[1], mirror=True, stretch=0.9)
    c = v[0].c
    x = float(c[np.argmin(c[:, 1]), 0])
    return v + [_stem(x, 0, 340, W)]


@recipe("Z")
def _Z(ch, S, W):
    return [_bar(60, 500, CAP - 0.3 * W), GL.C((70, 0), (500, CAP - 0.3 * W)), _bar(70, 520, 0.3 * W)]


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────
@recipe("0")
def _0(ch, S, W):
    return _bowl(40, h=CAP, stretch=0.7)


@recipe("1")
def _1(ch, S, W):
    return placed_box("wavy-branching-line/15", (60, 0, None, CAP), rot=270, stretch=1.0)


@recipe("2")
def _2(ch, S, W):
    # loop/2 is a 2 without its foot
    a = placed_box("loop/2", (50, 0, None, CAP), rot=240, stretch=0.86, teeth=False)
    c = a[0].c if a[0].c[-1, 1] < a[0].c[0, 1] else a[0].c[::-1]
    c = c[:int(0.9 * np.argmin(c[:, 1]))]          # to just short of its lowest point, then round onto the foot
    e = c[-1]
    return [OL.Stroke(np.vstack([c[::6], [[e[0] - 40, 40], [e[0] + 10, 0], [e[0] + 150, 22], [e[0] + 420, 10]]]), w=a[0].w)]


@recipe("3")
def _3(ch, S, W):
    # two of the c's curls, turned to face left, stacked
    top = placed_box("fork/1", (80, CAP * 0.45, None, CAP), rot=265, hflip=True, teeth=False, stretch=1.1)
    return top + placed_box("fork/1", (60, 0, None, CAP * 0.57), rot=265, hflip=True, teeth=False, stretch=1.1)


@recipe("4")
def _4(ch, S, W):
    return [_stem(380, 0, CAP, W), GL.C((60, 230), (380, CAP), w=0.8), _bar(60, 480, 230)]


@recipe("5")
def _5(ch, S, W):
    return placed_box("curved-sprig/3", (40, 0, None, CAP), rot=0, stretch=0.9)


@recipe("7")
def _7(ch, S, W):
    return [_bar(60, 470, CAP - 0.3 * W), GL.C((180, 0), (470, CAP - 0.3 * W))]


def sheet(refs: list[str], out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    cell, pad = 420, 40
    cols = 4
    rows = (len(refs) + cols - 1) // cols
    im = Image.new("RGB", (cell * cols, cell * rows), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    for k, ref in enumerate(refs):
        ox, oy = (k % cols) * cell, (k // cols) * cell
        try:
            s = skel(ref)
        except Exception as e:
            d.text((ox + 10, oy + 10), f"{ref}: {e}", fill="red", font=f)
            continue
        P = np.vstack([s["spine"]] + [t["pts"] for t in s["teeth"]])
        lo, hi = P.min(0), P.max(0)
        sc = (cell - 2 * pad) / max(hi - lo)
        to = lambda Q: [(ox + pad + (x - lo[0]) * sc, oy + cell - pad - (y - lo[1]) * sc) for x, y in Q]
        ww = max(2, int(s["w"] * sc))
        d.line(to(s["spine"]), fill="black", width=ww, joint="curve")
        for j, t in enumerate(s["teeth"]):
            d.line(to(t["pts"]), fill=(40, 90, 200), width=max(2, int(ww * min(t["w"], 1))), joint="curve")
            x, y = to(t["pts"][-1:])[0]
            d.text((x + 3, y - 8), str(j), fill=(200, 40, 40), font=f)
        for u in (0, 0.25, 0.5, 0.75, 1.0):
            x, y = to(s["spine"][int(u * (N_SPINE - 1)):][:1])[0]
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 160, 0))
            d.text((x + 5, y + 2), f"{u:g}", fill=(0, 130, 0), font=f)
        d.text((ox + 8, oy + 6), ref, fill="black", font=f)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "sheet":
        sheet(sys.argv[3:], Path(sys.argv[2]))
