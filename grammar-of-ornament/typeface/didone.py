"""
Neuron Didone — a rigorous high-contrast letter with the compounds' own
shapes grafted on where its serifs and terminals would be.

The body is classical: straight vertical stems, hairline thins, true
ellipses with vertical stress (thick where a curve runs vertically, hairline
where it runs across), strokes that join another thinning to a hairline
there. Nothing in the body wobbles.

The strangeness is all in the graft. Each free end of a straight stroke
carries a serif that is the actual outline of one compound piece
(silhouettes.py), laid across the end; each free end of a curve carries a
terminal that is the actual outline of another piece, continuing the curve.
A font uses one serif piece and one terminal piece throughout.

  .venv/bin/python typeface/didone.py SERIF TERMINAL [TEXT]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pathops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import outline as OL       # noqa: E402
import silhouettes as SI   # noqa: E402

CAP, XH, ASC, DESC = 700, 500, 740, -220
STEP = 3.0


# ── strokes ─────────────────────────────────────────────────────────────────

class K:
    """A stroke: a centreline and a role ("T" thick, "t" thin, "c" a curve with vertical stress).
    end0/end1: "auto" (serif on straight strokes, terminal on curves), "serif", "term" or None."""

    def __init__(self, pts, role="c", closed=False, end0="auto", end1="auto"):
        self.c = _dense(np.asarray(pts, float), closed)
        self.role, self.closed = role, closed
        self.end0, self.end1 = end0, end1


def _dense(P, closed=False):
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.hypot(*np.diff(P, axis=0).T)
    t = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(t[-1] / STEP) + 1)
    s = np.linspace(0, t[-1], n)
    out = np.stack([np.interp(s, t, P[:, 0]), np.interp(s, t, P[:, 1])], 1)
    return out[:-1] if closed else out


def L(p0, p1, role="T", **kw):
    return K([p0, p1], role, **kw)


def arc(cx, cy, rx, ry, a0, a1, n=None):
    n = n or max(12, int(abs(a1 - a0) / 2))
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def A(cx, cy, rx, ry, a0, a1, **kw):
    return K(arc(cx, cy, rx, ry, a0, a1), "c", **kw)


def chain(*parts, role="c", **kw):
    return K(np.vstack([np.asarray(p, float) for p in parts]), role, **kw)


def ell(cx, cy, rx, ry):
    return K(arc(cx, cy, rx, ry, 90, 450)[:-1], "c", closed=True)


# ── glyphs (a sample to judge the direction) ────────────────────────────────

G = {}
G["N"] = [L((80, 0), (80, 700), "t"), L((80, 700), (520, 0), "T", end0=None), L((520, 0), (520, 700), "t", end0=None)]
G["E"] = [L((80, 0), (80, 700)), L((80, 700), (450, 700), "t"), L((80, 360), (385, 360), "t"), L((80, 0), (470, 0), "t")]
G["U"] = [chain([(85, 700), (85, 240)], arc(300, 240, 215, 245, 180, 360), [(515, 240), (515, 700)])]
G["R"] = [L((80, 0), (80, 700)),
          chain([(80, 700), (320, 700)], arc(320, 527, 190, 173, 90, -90), [(320, 354), (80, 354)]),
          L((300, 354), (545, 0))]
G["O"] = [ell(310, 350, 248, 352)]
G["H"] = [L((80, 0), (80, 700)), L((520, 0), (520, 700)), L((80, 365), (520, 365), "t")]
G["A"] = [L((40, 0), (300, 712), "t", end1=None), L((300, 712), (560, 0), "T", end0=None), L((140, 240), (470, 240), "t")]
G["M"] = [L((70, 0), (70, 700), "t"), L((70, 700), (330, 0), "T", end0=None, end1=None),
          L((330, 0), (590, 700), "t", end0=None, end1=None), L((590, 0), (590, 700), "T", end1=None)]
G["G"] = [chain(arc(310, 350, 245, 352, 45, 330)), L((545, 0), (545, 290), "T", end0=None), L((380, 290), (560, 290), "t", end1=None)]
G["B"] = [L((80, 0), (80, 700)),
          chain([(80, 700), (300, 700)], arc(300, 538, 165, 162, 90, -90), [(300, 376), (80, 376)]),
          chain([(80, 376), (320, 376)], arc(320, 188, 190, 188, 90, -90), [(320, 0), (80, 0)])]
G["S"] = [chain(arc(290, 525, 205, 175, 25, 270), arc(290, 175, 220, 175, 90, -160))]

_arch = lambda x0, x1, top: chain(arc((x0 + x1) / 2, top - 175, (x1 - x0) / 2, 175, 165, 0), [(x1, top - 175), (x1, 0)])
G["n"] = [L((80, 0), (80, 500)), _arch(80, 430, 500)]
G["h"] = [L((80, 0), (80, 740)), _arch(80, 430, 500)]
G["m"] = [L((75, 0), (75, 500)), _arch(75, 370, 500), _arch(370, 665, 500)]
G["u"] = [chain([(80, 500), (80, 175)], arc(255, 175, 175, 175, 180, 345)), L((430, 0), (430, 500))]
G["o"] = [ell(250, 250, 192, 256)]
G["b"] = [L((80, 0), (80, 740)), A(265, 250, 185, 255, 150, -150)]
G["d"] = [L((420, 0), (420, 740)), A(235, 250, 185, 255, 30, 330)]
G["p"] = [L((80, -220), (80, 500)), A(265, 250, 185, 255, 150, -150)]
G["e"] = [L((62, 262), (440, 262), "t", end0=None), A(252, 250, 190, 256, 0, 318)]
G["c"] = [A(255, 250, 188, 256, 42, 322)]
G["a"] = [L((400, 0), (400, 330)), A(255, 330, 148, 172, 0, 152),
          A(238, 150, 165, 150, 68, 340, end0=None)]
G["r"] = [L((80, 0), (80, 500)), A(255, 330, 172, 172, 162, 62)]
G["f"] = [L((150, 0), (150, 560)), A(300, 560, 152, 180, 180, 32), L((40, 482), (300, 482), "t")]
G["t"] = [chain([(140, 640), (140, 120)], arc(245, 120, 105, 120, 180, 300)), L((35, 482), (300, 482), "t")]
G["s"] = [chain(arc(245, 378, 160, 124, 22, 270), arc(245, 128, 178, 128, 90, -158))]
G["g"] = [ell(235, 318, 160, 190), chain([(398, 505), (398, -30)], arc(228, -30, 170, 172, 0, -150))]
G["i"] = [L((110, 0), (110, 500))]
G["v"] = [L((20, 500), (230, -6), "T", end1=None), L((230, -6), (440, 500), "t", end0=None)]
DOTS = {"i": (110, 660)}


# ── geometry ────────────────────────────────────────────────────────────────

class Style:
    def __init__(self, serif: str, term: str, T=118, t=17, span=2.5, term_len=1.7, side=1, lower=0.9):
        self.serif, self.term = serif, term
        self.T, self.t, self.span, self.term_len, self.side, self.lower = T, t, span, term_len, side, lower


def widths(k: K, T, t, joined0, joined1):
    if k.role == "T":
        w = np.full(len(k.c), T)
    elif k.role == "t":
        w = np.full(len(k.c), t)
    else:
        tg = OL.tangents(k.c, k.closed)
        w = t + (T - t) * np.abs(tg[:, 1]) ** 2.2
    if k.role == "c" and not k.closed:                      # a curve thins to a hairline where it joins
        s = OL.arclen(k.c)
        for joined, d in ((joined0, s), (joined1, s[-1] - s)):
            if joined:
                r = np.clip(d / 90.0, 0, 1)
                w = t + (w - t) * (r * r * (3 - 2 * r))
    return w


def stroke_poly(k: K, w, caps):
    c = k.c
    tg = OL.tangents(c, k.closed)
    n = np.stack([-tg[:, 1], tg[:, 0]], 1)
    Lf, R = c + n * (w / 2)[:, None], c - n * (w / 2)[:, None]
    if k.closed:
        return [Lf, R[::-1]]
    out = [Lf]
    if caps[1] == "round":
        out.append(_cap(c[-1], tg[-1], w[-1] / 2))
    out.append(R[::-1])
    if caps[0] == "round":
        out.append(_cap(c[0], -tg[0], w[0] / 2))
    return [np.vstack(out)]


def _cap(p, tan, r):
    a0 = math.atan2(tan[1], tan[0])
    a = a0 + np.linspace(np.pi / 2, -np.pi / 2, 14)[1:-1]
    return p + r * np.stack([np.cos(a), np.sin(a)], 1)


def graft(piece: str, origin, ax, ay, length) -> list[np.ndarray]:
    """A piece's outline placed so its spine chord runs from origin along ax for `length`,
    with its left side toward ay."""
    kind, idx = piece.rsplit("/", 1)
    s = SI.silhouette(kind, int(idx))
    return [origin + length * (np.outer(C[:, 0], ax) + np.outer(C[:, 1], ay)) for C in s["contours"]]


def glyph(ch: str, st: Style) -> list[np.ndarray]:
    T = st.T * (st.lower if ch.islower() else 1)
    t = st.t * (st.lower if ch.islower() else 1)
    ks = G[ch]
    polys, grafts = [], []

    def touches(e, me):
        return any(np.min(np.hypot(*(o.c - e).T)) < 0.5 * T + 6 for o in ks if o is not me)

    centre = np.vstack([k.c for k in ks]).mean(0)
    for k in ks:
        j0 = not k.closed and touches(k.c[0], k)
        j1 = not k.closed and touches(k.c[-1], k)
        w = widths(k, T, t, j0, j1)
        caps = []
        for end, joined, mark in ((0, j0, k.end0), (1, j1, k.end1)):
            kind = None
            if not k.closed and not joined and mark is not None:
                kind = ("serif" if k.role in "Tt" else "term") if mark == "auto" else mark
            caps.append("round" if kind == "term" or (kind is None and not k.closed) else "butt")
            if kind is None or k.closed:
                continue
            tg = OL.tangents(k.c, False)
            e = k.c[0] if end == 0 else k.c[-1]
            out = -tg[0] if end == 0 else tg[-1]               # pointing out of the stroke
            nrm = np.array([-out[1], out[0]])
            we = w[0] if end == 0 else w[-1]
            if kind == "serif":
                span = st.span * T * (0.75 if k.role == "t" else 1.0)
                toward = -out * st.side                       # the piece's left side faces into the stroke (or away)
                grafts += graft(st.serif, e - nrm * span / 2 + out * 0.0, nrm, toward, span)
            else:
                # the terminal continues the curve; its left side turns the way the curve was turning
                inward = centre - e
                ay = nrm if nrm @ inward > 0 else -nrm
                grafts += graft(st.term, e - out * 0.15 * we, out, ay, st.term_len * T)
        polys += stroke_poly(k, w, caps)
    if ch in DOTS:
        x, y = DOTS[ch]
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        polys.append(np.stack([x + 0.62 * T * np.cos(a), y + 0.62 * T * np.sin(a)], 1))
    return _union(polys, grafts)


def _path(contours, fill=pathops.FillType.WINDING):
    p = pathops.Path(fillType=fill)
    pen = p.getPen()
    for C in contours:
        pen.moveTo(tuple(C[0]))
        for q in C[1:]:
            pen.lineTo(tuple(q))
        pen.closePath()
    return p


def _union(polys, grafts):
    body = _path(polys)
    body.simplify(fix_winding=True, clockwise=False)
    if grafts:
        g = _path(grafts, pathops.FillType.EVEN_ODD)
        g.simplify(fix_winding=True, clockwise=False)
        both = pathops.Path()                         # both consistently wound, so one simplify unions them
        body.draw(both.getPen())
        g.draw(both.getPen())
        both.simplify(fix_winding=True, clockwise=False)
        body = both
    out, cur = [], []
    for verb, pts in body.segments:
        if verb == "moveTo":
            cur = [pts[0]]
        elif verb in ("closePath", "endPath"):
            if len(cur) > 2:
                out.append(np.array(cur, float))
            cur = []
        else:
            cur.append(pts[-1])
    return out


# ── proof ───────────────────────────────────────────────────────────────────

def render(text: str, st: Style, H: int = 160):
    from PIL import Image, ImageDraw
    x, polys = 0.0, []
    for ch in text:
        if ch == " ":
            x += 280
            continue
        cont = glyph(ch, st)
        P = np.vstack(cont)
        dx = x - P[:, 0].min() + 55
        polys += [C + [dx, 0] for C in cont]
        x = dx + P[:, 0].max() + 55
    k = H / 1000
    im = Image.new("L", (int(x * k) + 40, int(1.2 * H)), 255)
    d = ImageDraw.Draw(im)
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    for fill, sel in ((0, lambda a: a > 0), (255, lambda a: a <= 0)):
        for C in polys:
            if sel(area(C)):
                d.polygon([(q[0] * k + 20, (880 - q[1]) * k) for q in C], fill=fill)
    return im


if __name__ == "__main__":
    serif, term = sys.argv[1], sys.argv[2]
    text = sys.argv[3] if len(sys.argv) > 3 else "NEURO HAMBURGS neuron hamburgefonstiv"
    out = sys.argv[4] if len(sys.argv) > 4 else "didone.png"
    render(text, Style(serif, term)).save(out)
