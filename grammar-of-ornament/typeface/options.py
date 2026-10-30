"""
Neuron — ornament trials on Curated's letters: sample strips only, no fonts.

Each option is a rule applied to the sample letters, so options can be compared
before any of them is drawn letter by letter.

  eyes     a stroke swells and is pierced by an oval hole (mesh/0, mesh/3, loop/1)
  fruit    stems ending in round fruit (wavy-feather/14 and 15's teeth)
  fingers  thick blunt branches (wavy-feather/1's teeth)
  knots    thin strokes with blobs at their ends and along them (straight-comb/0)
  inline   a slit down the middle of every stroke (loop/11, loop/15)
  hatched  fine slits across every stroke (the hatched bands)
  lobed    heavy strokes with undulating edges (lobed-wavy-band, wavy-band)
  echo     a full-length second strand beside the main strokes (curved-sprig/12)
  roots    free ends dividing into three rounded prongs (curved-branching-line/9)
  curls    free ends rolling up in the spiral of curved-sprig/13

Real geometry (the pieces' own outlines from silhouettes.py, not redrawn):

  graft    branching pieces carried on out of a letter's free ends
  stems    a whole feather or comb standing in as a letter's stem
  meshes   the same with pieces that have holes
  coils    the thick spirals growing from free ends, at their own proportions
  hand     pieces chosen for each letter, sized so their body matches the stroke (HAND)

  .venv/bin/python typeface/options.py         → typeface/proofs/options/<option>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as BD          # noqa: E402
import curated as CU        # noqa: E402
import outline as OL        # noqa: E402
import silhouettes as SI    # noqa: E402

SAMPLE = ["Grammar of Ornament", "hamburgefonstiv", "THE QUICK 2946"]
SCALE, GAP, SPACE = 0.34, 55, 280


def _main(S):
    st = [s for s in S if isinstance(s, OL.Stroke) and not getattr(s, "free", False)]
    return sorted(st, key=lambda s: -OL.arclen(s.c)[-1]), np.vstack([s.c for s in st]).mean(0)


def _at(s, u):
    c = s.c
    sa = OL.arclen(c)
    i = int(np.clip(np.searchsorted(sa, u * sa[-1]), 3, len(c) - 4))
    t = c[i + 3] - c[i - 3]
    t = t / max(np.hypot(*t), 1e-9)
    return c[i], t, np.array([-t[1], t[0]])


def _outward(p, n, centre):
    return n if (p - centre) @ n > 0 else -n


def _leaf(P, w, e=0.3):
    st = OL.Stroke(np.asarray(P, float), w=w, taper0=False, taper1=False, belly=0.5, e0w=e, e1w=e)
    st.no_contrast = True
    return st


def eyes(ch, S, W, centre):
    extra, cuts = [], []
    st, _ = _main(S)
    for s, us in zip(st[:2], ((0.34, 0.72), (0.5,))):
        for u in us:
            L = OL.arclen(s.c)[-1]
            if L < 3.2 * W:
                continue
            f = 1.75 * W / L
            pts = [_at(s, min(max(u + k * f, 0.02), 0.98))[0] for k in np.linspace(-1, 1, 7)]
            p, _, n = _at(s, u)
            o = _outward(p, n, centre)
            bow = (np.sin(np.linspace(0, np.pi, 7)) * 0.34 * W)[:, None] * o
            P = np.array(pts) + bow
            extra.append(_leaf(P, 1.9, 0.3))
            h = OL.arclen(P)[-1] * 0.31
            q = P[3]
            tn = (P[4] - P[2]) / max(np.hypot(*(P[4] - P[2])), 1e-9)
            nn = np.array([-tn[1], tn[0]])
            a = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            cuts.append(q + np.outer(np.cos(a) * h, tn) + np.outer(np.sin(a) * 0.33 * W, nn))
    return extra, cuts


def _teeth(ch, S, W, centre, ref, which, scale, w, n, tip):
    extra = []
    st, _ = _main(S)
    keep = CU.TOOTH_TIP
    CU.TOOTH_TIP = tip
    for s in st[:2]:
        L = OL.arclen(s.c)[-1]
        if L < 3 * W:
            continue
        for k, u in enumerate(np.linspace(0.25, 0.8, n)):
            p, _, nr = _at(s, u)
            flip = False
            t = CU.teeth_on(s, ref, which=[which[k % len(which)]], scale=scale, w=w, W=W, on=(u, min(u + 0.04, 1)))
            if not t:
                continue
            if (t[0].c[-1] - p) @ _outward(p, nr, centre) < 0:
                t = CU.teeth_on(s, ref, which=[which[k % len(which)]], scale=scale, w=w, W=W,
                                on=(u, min(u + 0.04, 1)), flip=True)
            extra += t
    CU.TOOTH_TIP = keep
    return extra


def fruit(ch, S, W, centre):
    t = _teeth(ch, S, W, centre, "wavy-feather/15", [2, 3, 8, 10], 2.0, 0.42, 2, 0.8)
    return t + [OL.Dot(float(x.c[-1, 0]), float(x.c[-1, 1]), 1.05) for x in t], []


def fingers(ch, S, W, centre):
    return _teeth(ch, S, W, centre, "wavy-feather/1", [3, 5, 6], 2.0, 0.85, 2, 0.8), []


def knots(ch, S, W, centre):
    extra = []
    st, _ = _main(S)
    for s in st:
        s.w *= 0.68
    OL.outline(S, W)
    for s in st:
        if not s.closed and not s.taper0:
            s.b0 = True
        if not s.closed and not s.taper1:
            s.b1 = True
        s.taper0_set, s.taper1_set = s.taper0, s.taper1
    s = st[0]
    L = OL.arclen(s.c)[-1]
    if L > 4 * W:
        f = 0.8 * W / L
        extra.append(_leaf([_at(s, 0.5 + k * f)[0] for k in np.linspace(-1, 1, 5)], 1.55, 0.35))
    return extra, []


def _widths(s, W):
    w = OL.widths(s, W)
    return np.interp(np.arange(len(s.c)), np.arange(len(w)), w)


def _heavier(S, W, k):
    st, _ = _main(S)
    for s in st:
        s.w *= k
    OL.outline(S, W)
    return st


def inline(ch, S, W, centre):
    cuts = []
    for s in _heavier(S, W, 1.45):
        c = np.vstack([s.c, s.c[:1]]) if s.closed else s.c
        w = _widths(s, W)
        w = np.append(w, w[0]) if s.closed else w
        u = OL.arclen(c) / OL.arclen(c)[-1]
        h = 0.11 * w * (np.ones_like(u) if s.closed else np.clip(np.minimum(u, 1 - u) / 0.1, 0, 1) ** 0.6)
        t = OL.tangents(c, False)
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        cuts.append(np.vstack([c + n * h[:, None], (c - n * h[:, None])[::-1]]))
    return [], cuts


def hatched(ch, S, W, centre):
    cuts = []
    for s in _heavier(S, W, 1.4):
        c = s.c
        sa = OL.arclen(c)
        w = _widths(s, W)
        t = OL.tangents(c, s.closed)
        for d in np.arange(0.6 * W, sa[-1] - 0.5 * W, 0.46 * W):
            i = int(np.searchsorted(sa, d))
            n = np.array([-t[i][1], t[i][0]])
            h, b = 0.36 * w[i], 0.075 * W
            cuts.append(np.array([c[i] - n * h - t[i] * b, c[i] + n * h - t[i] * b,
                                  c[i] + n * h + t[i] * b, c[i] - n * h + t[i] * b]))
    return [], cuts


def lobed(ch, S, W, centre):
    st, _ = _main(S)
    for k, s in enumerate(st):
        s.w *= 1.5
        L = OL.arclen(s.c)[-1]
        x = np.linspace(0, L / W, 200)
        s.prof = 1 + 0.26 * np.sin(2 * np.pi * x / 2.1 + k) + 0.12 * np.sin(2 * np.pi * x / 1.27 + 2 * k)
    return [], []


def echo(ch, S, W, centre):
    extra = []
    st, _ = _main(S)
    for s in st[:2]:
        if OL.arclen(s.c)[-1] < 3.5 * W:
            continue
        p, _, n = _at(s, 0.5)
        sd = 1 if (p - centre) @ n > 0 else -1
        us = np.linspace(0.07, 0.93, 36)
        P = []
        for u in us:
            q, _, nn = _at(s, u)
            P.append(q + nn * sd * (0.5 * s.w + 0.28 + 0.31) * W * max(np.sin(np.pi * (u - 0.07) / 0.86), 0.0) ** 0.3)
        extra.append(_leaf(P, 0.62, 0.5))
    return extra, []


def _free_ends(st):
    for s in st:
        if s.closed:
            continue
        if not s.taper0:
            yield s.c[0], s.c[0] - s.c[6]
        if not s.taper1:
            yield s.c[-1], s.c[-1] - s.c[-7]


def roots(ch, S, W, centre):
    extra = []
    st, _ = _main(S)
    for e, h in _free_ends(st):
        h = h / max(np.hypot(*h), 1e-9)
        for a, L in ((-34, 1.5), (0, 1.25), (34, 1.5)):
            r = np.radians(a)
            d = np.array([h[0] * np.cos(r) - h[1] * np.sin(r), h[0] * np.sin(r) + h[1] * np.cos(r)])
            b = e - h * 0.8 * W
            extra.append(_leaf([b, b + (h + d) * 0.5 * W, e + d * L * W * 0.6, e + (d + 0.15 * h) * L * W], 0.52, 0.9))
    return extra, []


def curls(ch, S, W, centre):
    extra = []
    st, _ = _main(S)
    sp = CU.skel("curved-sprig/13")["spine"]
    z0 = (sp[:, 0] + 1j * sp[:, 1])[int(0.58 * len(sp)):]
    for e, h in _free_ends(st):
        cross = h[0] * (centre - e)[1] - h[1] * (centre - e)[0]
        turn = -1 if cross > 0 else 1
        a = np.unwrap(np.angle(np.diff(z0)))
        z = np.conj(z0) if np.sign(a[-1] - a[0]) != turn else z0
        m = complex(*h) / abs(complex(*h)) / ((z[3] - z[0]) / abs(z[3] - z[0])) * 3.6 * W / np.sum(np.abs(np.diff(z)))
        P = complex(*e) + m * (z - z[0])
        extra.append(_leaf(np.stack([P.real, P.imag], 1)[::2], 0.62, 0.9))
    return extra, []


def letter(ch, W, fn):
    S = CU._fitted(ch, W)
    OL.outline(S, W)
    _, centre = _main(S)
    extra, cuts = fn(ch, S, W, centre)
    return OL.outline(list(S) + extra, W, pinhole=0.2), cuts


def strip(text, fn, H=1500):
    rows = []
    for ch in text:
        if ch == " ":
            rows.append(np.zeros((int(H * SCALE), int(SPACE * SCALE)), bool))
            continue
        cs, cuts = letter(ch, CU.weight(ch, BD.main_stroke), fn)
        P = np.vstack(cs)
        x0, x1 = P[:, 0].min() - GAP / 2, P[:, 0].max() + GAP / 2
        h, w = int(H * SCALE), int((x1 - x0) * SCALE) + 2
        m = np.zeros((h, w), bool)
        to = lambda C: [((x - x0) * SCALE, h - (y + 420) * SCALE) for x, y in C]
        for C in cs:
            t = Image.new("1", (w, h), 0)
            ImageDraw.Draw(t).polygon(to(C), fill=1)
            m ^= np.asarray(t, bool)
        for C in cuts:
            t = Image.new("1", (w, h), 0)
            ImageDraw.Draw(t).polygon(to(C), fill=1)
            m &= ~np.asarray(t, bool)
        rows.append(m)
    return np.hstack(rows)


GRAFT = ["curved-branching-line/9", "curved-branching-line/7", "curved-branching-line/0", "curved-branching-line/3",
         "wavy-sprig/8", "curved-sprig/15", "curved-sprig/8", "curved-sprig/12", "fork/0", "fork/1"]
STEMS = ["wavy-feather/1", "curved-feather/1", "curved-feather/5", "straight-feather/4", "straight-feather/5",
         "straight-feather/7", "straight-comb/1", "curved-comb/7", "wavy-comb/3"]
MESHES = ["mesh/0", "mesh/1", "mesh/3", "mesh/7", "mesh/6", "mesh/2", "curved-comb/5"]
COILS = ["loop/14", "curved-branching-line/10", "curved-sprig/13", "fork/10", "fork/12", "fork/13"]


def _pick(pool, ch, k=0):
    return pool[(ord(ch) * 7 + k * 3) % len(pool)]


def _grow(ref, e, h, W, centre, length):
    """The piece's outline rooted at e: its narrower end first, its spine leaving along h, `length` weights
    long, turned to whichever side lies away from the letter's centre."""
    sl = SI.silhouette(ref)
    sp = sl["spine"]
    root_first = sl["profile"][0] <= sl["profile"][-1]
    z0, z1 = (sp[0], sp[min(8, len(sp) - 1)]) if root_first else (sp[-1], sp[max(-9, -len(sp))])
    root, t = complex(*z0), complex(*(z1 - z0))
    out = []
    for mirror in (False, True):
        m = complex(*h) / abs(complex(*h))
        cs = []
        for C in sl["contours"]:
            z = C[:, 0] + 1j * C[:, 1]
            r, tt = root, t
            if mirror:
                z, r, tt = np.conj(z), np.conj(r), np.conj(t)
            q = complex(*e) + (z - r) * m / (tt / abs(tt)) * length * W / OL.arclen(sp)[-1]
            cs.append(np.stack([q.real, q.imag], 1))
        out.append(cs)
    far = lambda cs: float(np.mean(np.hypot(*(np.vstack(cs) - centre).T)))
    return max(out, key=far)


def _ends(st, W, n=2):
    ends = [(e, h / max(np.hypot(*h), 1e-9)) for e, h in _free_ends(st)]
    return sorted(ends, key=lambda p: -p[0][1])[:n]             # the highest first


def _stem_of(st):
    for s in st:
        d = s.c[-1] - s.c[0]
        if s.kind == "crescent" and abs(d[0]) < 0.15 * abs(d[1]) and abs(d[1]) > 400:
            return s
    return None


def _standing(ref, s, W, centre):
    """The piece's outline standing along stroke s, end to end, its body about the stroke's width."""
    sl = SI.silhouette(ref)
    a, b = s.c[0], s.c[-1]
    L = np.hypot(*(b - a))
    u, n = (b - a) / L, np.array([-(b - a)[1], (b - a)[0]]) / L
    P = np.vstack(sl["contours"])
    lo, hi = P[:, 0].min(), P[:, 0].max()
    across = np.clip(1.25 * W / (sl["body"] * L), 0.5, 1.5) * L
    opts = []
    for sd in (1, -1):
        cs = [a + np.outer((C[:, 0] - lo) / (hi - lo) * L, u) + np.outer(sd * C[:, 1] * across, n) for C in sl["contours"]]
        opts.append(cs)
    return max(opts, key=lambda cs: float(np.mean(np.hypot(*(np.vstack(cs) - centre).T))))


def graft(ch, st, W, centre):
    return st, [_grow(_pick(GRAFT, ch, k), e - h * 0.3 * W, h, W, centre, 4.2)
                for k, (e, h) in enumerate(_ends(st, W))]


def coils(ch, st, W, centre):
    return st, [_grow(_pick(COILS, ch, k), e - h * 0.3 * W, h, W, centre, 9.0)
                for k, (e, h) in enumerate(_ends(st, W, 1))]


def _stood(pool, ch, st, W, centre):
    s = _stem_of(st)
    if s is None:
        return graft(ch, st, W, centre)
    return [x for x in st if x is not s], [_standing(_pick(pool, ch), s, W, centre)]


def stems(ch, st, W, centre):
    return _stood(STEMS, ch, st, W, centre)


def meshes(ch, st, W, centre):
    return _stood(MESHES, ch, st, W, centre)


# chosen by hand: letter → [(piece, which free end, counting from the highest)]; sized so the piece's body
# is the stroke's weight, at its own proportions
HAND = {"G": [("loop/14", 0)], "r": [("curved-branching-line/9", 0)], "a": [("curved-branching-line/7", 0)],
        "m": [("curved-sprig/12", 0)], "n": [("curved-sprig/12", 0)], "f": [("curved-branching-line/10", 0)],
        "t": [("wavy-sprig/8", 0)], "e": [("curved-sprig/13", 0)], "h": [("curved-branching-line/7", 0)],
        "b": [("curved-branching-line/9", 0)], "u": [("curved-sprig/15", 0)], "g": [("loop/14", 0)],
        "s": [("curved-branching-line/10", 0)], "i": [("curved-sprig/12", 0)], "v": [("curved-branching-line/9", 0)]}
BODY = 0.9


def hand(ch, st, W, centre):
    ends = _ends(st, W, 9)
    out = []
    for ref, k in HAND.get(ch, []):
        if k < len(ends):
            e, h = ends[k]
            sl = SI.silhouette(ref)
            out.append(_grow(ref, e - h * 0.3 * W, h, W, centre, BODY * OL.arclen(sl["spine"])[-1] / sl["body"]))
    return st, out


def strip_real(text, fn, H=1500):
    rows = []
    for ch in text:
        if ch == " ":
            rows.append(np.zeros((int(H * SCALE), int(SPACE * SCALE)), bool))
            continue
        W = CU.weight(ch, BD.main_stroke)
        S = CU._fitted(ch, W)
        OL.outline(S, W)
        st, centre = _main(S)
        keep, pieces = fn(ch, st, W, centre)
        rest = [x for x in S if x not in st or x in keep]
        cs = OL.outline(rest, W) if rest else []
        P = np.vstack(cs + [C for g in pieces for C in g])
        x0, x1 = P[:, 0].min() - GAP / 2, P[:, 0].max() + GAP / 2
        h, w = int(H * SCALE), int((x1 - x0) * SCALE) + 2
        to = lambda C: [((x - x0) * SCALE, h - (y + 420) * SCALE) for x, y in C]

        def fill(group):
            m = np.zeros((h, w), bool)
            for C in group:
                t = Image.new("1", (w, h), 0)
                ImageDraw.Draw(t).polygon(to(C), fill=1)
                m ^= np.asarray(t, bool)
            return m
        m = fill(cs)
        for g in pieces:
            m |= fill(g)
        rows.append(m)
    return np.hstack(rows)


def main():
    BD.configure("Curated")
    out = HERE / "proofs" / "options"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("graft", graft), ("stems", stems), ("meshes", meshes), ("coils", coils), ("hand", hand)):
        strips = [strip_real(t, fn) for t in SAMPLE]
        w = max(s.shape[1] for s in strips)
        im = np.vstack([np.pad(s, ((0, 0), (0, w - s.shape[1]))) for s in strips])
        Image.fromarray(~im).save(out / f"{name}.png")
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()


# ── parts of real compounds, placed by hand (parts.py) ───────────────────────────────────────────────
import parts as PT          # noqa: E402

FORK_A, FORK_B, FORK_C = ("curved-sprig/15", 0.30, 0.43), ("mesh/3", 0.41, 0.50), ("fork/0", 0.26, 0.60)
FINGERS = ("wavy-sprig/3", 0.62, 0.83)
HOOK = ("curved-branching-line/10", 0.35, 0.46)
SEED = ("curved-comb/0", 0.0, 0.15)
PARTS = {"G": [(HOOK, 0)], "r": [(FINGERS, 0)], "a": [(FORK_A, 0)], "m": [(FORK_B, 0)], "o": [(SEED, "left")],
         "f": [(HOOK, 0)], "O": [(SEED, "left")], "n": [(FORK_B, 0)], "e": [(FINGERS, 0)], "t": [(FORK_A, 0)],
         "h": [(FORK_A, 0)], "b": [(FORK_A, 0)], "u": [(FORK_B, 0)], "g": [(HOOK, -1)], "s": [(HOOK, 0)],
         "i": [(FORK_B, -1)], "v": [(FORK_A, 0)], "T": [(FINGERS, 0), (FINGERS, 1)], "H": [(FORK_A, 0), (FORK_A, -1)],
         "E": [(FINGERS, 0)], "Q": [(SEED, "left")], "U": [(FORK_A, 0)], "I": [(FORK_A, 0), (FORK_A, -1)],
         "C": [(HOOK, 0)], "K": [(FORK_A, 0)], "2": [(HOOK, 0)], "9": [(SEED, "right")], "4": [(FORK_A, 0)],
         "6": [(SEED, "left")]}
PART_BODY = 0.85


def _put(spec, e, h, W, centre):
    """The part rooted at e, leaving along h, its body PART_BODY weights, on the side away from the centre."""
    cs = PT.part(*spec)["contours"]
    P = np.vstack(cs)
    d = complex(*(P.mean(0)))
    opts = []
    for mirror in (False, True):
        m = (complex(*h) / abs(complex(*h))) / ((d.conjugate() if mirror else d) / abs(d)) * PART_BODY * W
        g = []
        for C in cs:
            z = C[:, 0] + 1j * C[:, 1]
            q = complex(*e) + (np.conj(z) if mirror else z) * m
            g.append(np.stack([q.real, q.imag], 1))
        opts.append(g)
    return max(opts, key=lambda g: float(np.mean(np.hypot(*(np.vstack(g) - centre).T))))


def parted(ch, st, W, centre):
    ends = _ends(st, W, 9)
    B = np.vstack([s.c for s in st])
    out = []
    for spec, k in PARTS.get(ch, []):
        if isinstance(k, str):
            s = st[0]
            i = int(np.argmin(s.c[:, 0])) if k == "left" else int(np.argmax(s.c[:, 0]))
            out.append(_put(spec, s.c[i], np.array([-1.0 if k == "left" else 1.0, 0.25]), W, centre))
        elif ends and (k < len(ends) or k == -1):
            e, h = ends[k]
            out.append(_put(spec, e - h * 0.25 * W, h, W, centre))
    return st, out


# ── larger and finer parts, placed by hand ───────────────────────────────────────────────────────────
from functools import lru_cache          # noqa: E402


@lru_cache(maxsize=None)
def _frag(ref, a, b, large=False):
    p = PT.part(ref, a, b, 30, 0.06) if large else PT.part(ref, a, b, 14, 0.1)
    return p["contours"]


def _ink_centre(cs):
    P = np.vstack(cs)
    lo, hi = P.min(0), P.max(0)
    k = 200 / max(hi - lo)
    W, H = ((hi - lo) * k).astype(int) + 3
    m = np.zeros((H, W), bool)
    for C in cs:
        im = Image.new("1", (W, H), 0)
        ImageDraw.Draw(im).polygon([tuple(q) for q in (C - lo) * k], fill=1)
        m ^= np.asarray(im, bool)
    ys, xs = np.nonzero(m)
    return lo + np.array([xs.mean(), ys.mean()]) / k


def _axes(cs):
    P = np.vstack(cs)
    m = P.mean(0)
    _, V = np.linalg.eigh(np.cov((P - m).T))
    return m, V[:, 1], V[:, 0]


def along(spec, s, u0, u1, W, centre, rev=False, sink=0.2, height=1.0, toward=None):
    """The fragment laid along a stretch of stroke s: its long axis follows the stroke's curve, its denser
    edge sits in the stroke, the rest stands away from the letter's centre."""
    cs = _frag(*spec)
    m, ax, ac = _axes(cs)
    ic = _ink_centre(cs)
    P = np.vstack(cs)
    x, y = (P - m) @ ax, (P - m) @ ac
    sign = 1 if (ic - m) @ ac < (y.min() + y.max()) / 2 else -1       # +y points from the dense edge outward
    y0 = y.min() if sign > 0 else y.max()
    c = np.vstack([s.c, s.c[:1]]) if s.closed else s.c
    sa = OL.arclen(c)
    s0, s1 = u0 * sa[-1], u1 * sa[-1]
    k = (s1 - s0) / (x.max() - x.min())
    tg = OL.tangents(c, False)
    nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
    mid = (s0 + s1) / 2
    i = int(np.searchsorted(sa, mid))
    sd = 1 if ((c[i] - centre) if toward is None else np.asarray(toward, float)) @ nr[i] > 0 else -1
    out = []
    for C in cs:
        X = ((C - m) @ ax - x.min()) / (x.max() - x.min())
        Y = ((C - m) @ ac - y0) * sign * k * height
        d = s1 - X * (s1 - s0) if rev else s0 + X * (s1 - s0)
        q = np.stack([np.interp(d, sa, c[:, j]) for j in (0, 1)], 1)
        n = np.stack([np.interp(d, sa, nr[:, j]) for j in (0, 1)], 1)
        out.append(q + n * (sd * (Y - sink * W))[:, None])
    return out


def at_end(spec, e, h, W, centre, length=4.5, flip=False):
    """The fragment growing out of a free end, its long axis along the stroke's heading."""
    cs = _frag(*spec)
    m, ax, ac = _axes(cs)
    P = np.vstack(cs)
    x, y = (P - m) @ ax, (P - m) @ ac
    k = length * W / (x.max() - x.min())
    h = h / np.hypot(*h)
    n = np.array([-h[1], h[0]])
    opts = []
    for sx in (1, -1):
        for sy in (1, -1):
            o = []
            for C in cs:
                X = ((C - m) @ ax) * sx
                Y = ((C - m) @ ac) * sy
                x0 = (x * sx).min()
                o.append(e - h * 0.3 * W + np.outer((X - x0) * k, h) + np.outer(Y * k, n))
            opts.append(o)
    opts.sort(key=lambda o: -float(np.mean(np.hypot(*(np.vstack(o) - centre).T))))
    return opts[1 if flip else 0]


FINGER_ROW = ("wavy-feather/6", 0.0, 0.30)
FINGER_ROW2 = ("wavy-feather/6", 0.18, 0.48)
COMB = ("straight-comb/12", 0.36, 0.66)
RIB = ("curved-feather/14", 0.36, 0.66)
ANTLER_A, ANTLER_B = ("curved-comb/14", 0.0, 0.30), ("curved-comb/14", 0.36, 0.66)
ANTLER_C, ANTLER_D = ("curved-comb/11", 0.36, 0.66), ("curved-comb/11", 0.18, 0.48)
ANTLER_E = ("curved-comb/15", 0.54, 0.84)
BARBS, BARBS2 = ("straight-feather/10", 0.0, 0.30), ("straight-feather/8", 0.54, 0.84)
# letter → [("along", fragment, stroke, u0, u1) or ("end", fragment, which end, length in weights)]
FINE = {"G": [("along", FINGER_ROW, 0, 0.22, 0.62)], "r": [("end", ANTLER_C, 0, 4.0)], "a": [("along", BARBS, 0, 0.2, 0.6)],
        "m": [("along", COMB, 0, 0.3, 0.72)], "o": [("along", RIB, 0, 0.15, 0.55)], "f": [("end", ANTLER_B, 0, 4.5)],
        "O": [("along", FINGER_ROW2, 0, 0.12, 0.52)], "n": [("along", COMB, 0, 0.3, 0.72)], "e": [("end", ANTLER_D, 0, 4.0)],
        "t": [("end", ANTLER_E, -1, 3.6)], "h": [("along", RIB, 0, 0.5, 0.97)], "b": [("along", BARBS2, 0, 0.5, 0.97)],
        "u": [("along", BARBS, 0, 0.1, 0.45)], "g": [("end", ANTLER_A, -1, 4.5)], "s": [("end", ANTLER_D, 0, 3.2)],
        "i": [("along", BARBS2, 0, 0.12, 0.88)], "v": [("along", FINGER_ROW, 0, 0.1, 0.55)], "T": [("end", ANTLER_C, 0, 4.0), ("end", ANTLER_C, 1, 4.0)],
        "H": [("along", FINGER_ROW2, 0, 0.15, 0.85)], "E": [("end", ANTLER_E, 0, 4.2)], "Q": [("along", RIB, 0, 0.12, 0.5)],
        "U": [("along", BARBS, 0, 0.1, 0.45)], "I": [("along", BARBS2, 0, 0.12, 0.88)], "C": [("along", FINGER_ROW, 0, 0.15, 0.55)],
        "K": [("along", RIB, 0, 0.15, 0.85)]}


def fine(ch, st, W, centre):
    ends = _ends(st, W, 9)
    out = []
    for item in FINE.get(ch, []):
        if item[0] == "along":
            _, spec, i, u0, u1 = item
            out.append(along(spec, st[i], u0, u1, W, centre))
        elif ends and (item[2] < len(ends) or item[2] == -1):
            _, spec, k, L = item
            e, h = ends[k]
            out.append(at_end(spec, e, h, W, centre, L))
    return st, out


# one family, one place: comb fragments along the top of each letter
CRESTS = [COMB, RIB, BARBS, FINGER_ROW2]
CREST_MAX = 5.2             # longest crest, in weights


def _top_run(s, y_cut):
    c = s.c
    ok = c[:, 1] > y_cut
    best, i, n = (0, 0), 0, len(c)
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j + 1
        else:
            i += 1
    sa = OL.arclen(c)
    return sa[best[0]] / sa[-1], sa[best[1]] / sa[-1], sa[best[1]] - sa[best[0]], sa[-1]


def crest(ch, st, W, centre):
    B = np.vstack([s.c for s in st])
    top, bot = B[:, 1].max(), B[:, 1].min()
    cut = top - 0.34 * (top - min(bot, 0) if ch not in "gjpqy" else 500)
    cand = sorted(st, key=lambda s: -_top_run(s, cut)[2])
    s = cand[0]
    u0, u1, run, L = _top_run(s, cut)
    if run < 2.2 * W:                        # a stem: the crest runs down its upper left instead
        u0, u1 = (0.5, 0.96) if s.c[-1, 1] > s.c[0, 1] else (0.04, 0.5)
        run = (u1 - u0) * L
    if run > CREST_MAX * W:
        mid, half = (u0 + u1) / 2, CREST_MAX * W / L / 2
        u0, u1 = mid - half, mid + half
    return st, [along(CRESTS[ord(ch) % len(CRESTS)], s, u0, u1, W, centre, sink=0.12)]


# ── variants: every glyph, one hand-chosen fragment in one hand-chosen place ─────────────────────────
# letter → (fragment, stroke, where). stroke "L" is the longest; where is a stretch (u0, u1) or a region of
# the letter ("top", "left", "right", "upper"), and the fragment stands out toward that region.
LOWER = {"a": (BARBS, "L", (0.2, 0.6)), "b": (BARBS2, "L", (0.5, 0.97)), "c": (RIB, 0, "left"), "d": (BARBS2, 1, "upper"),
         "e": (RIB, 0, "left"), "f": (BARBS, 0, "top"), "g": (RIB, 0, "left"), "h": (BARBS2, 0, "upper"),
         "i": (BARBS2, 0, "left"), "j": (BARBS, 0, "left"), "k": (BARBS2, 0, "upper"), "l": (BARBS2, 0, "left"),
         "m": (COMB, "L", (0.3, 0.72)), "n": (COMB, "L", (0.3, 0.72)), "o": (RIB, "L", (0.15, 0.55)),
         "p": (BARBS2, 1, "right"), "q": (BARBS2, 0, "left"), "r": (COMB, 0, "top"), "s": (RIB, 0, "top"),
         "t": (BARBS2, 0, "upper"), "u": (BARBS, 0, "left"), "v": (FINGER_ROW, "L", (0.1, 0.55)),
         "w": (FINGER_ROW, 0, "left"), "x": (BARBS, 0, "left"), "y": (BARBS, 0, "left"), "z": (RIB, 0, "top")}
UPPER = {"A": (FINGER_ROW, 0, "left"), "B": (BARBS, 1, "right"), "C": (COMB, 0, "top"), "D": (COMB, 1, "top"),
         "E": (FINGER_ROW, 0, "left"), "F": (RIB, 0, "left"), "G": (COMB, 0, "top"), "H": (FINGER_ROW2, 0, "left"),
         "I": (BARBS2, 0, "left"), "J": (RIB, 0, "left"), "K": (RIB, 0, "left"), "L": (RIB, 0, "left"),
         "M": (COMB, 0, "top"), "N": (COMB, 0, "top"), "O": (COMB, 0, "top"), "P": (COMB, 1, "top"), "Q": (COMB, 0, "top"),
         "R": (COMB, 1, "top"), "S": (RIB, 0, "top"), "T": (COMB, 1, "top"), "U": (FINGER_ROW, 0, "left"),
         "V": (FINGER_ROW, 0, "left"), "W": (FINGER_ROW, 0, "left"), "X": (BARBS, 0, "left"), "Y": (FINGER_ROW, 0, "left"),
         "Z": (RIB, 0, "top")}
FIGS = {"0": (RIB, 0, "left"), "1": (BARBS2, 0, "left"), "2": (RIB, 0, "top"), "3": (RIB, 0, "top"), "4": (BARBS, 0, "left"),
        "5": (RIB, 0, "right"), "6": (RIB, 0, "left"), "7": (RIB, 0, "top"), "8": (RIB, 1, "left"), "9": (RIB, 0, "right")}
TABLE = {**LOWER, **UPPER, **FIGS}
TOWARD = {"top": (0, 1), "left": (-1, 0), "right": (1, 0), "upper": (-1, 0)}


def region(s, where, box, W, longest):
    (x0, y0), (x1, y1) = box
    c = s.c
    Hh, Ww = y1 - y0, x1 - x0
    ok = {"top": c[:, 1] > y1 - 0.3 * Hh,
          "left": (c[:, 0] < x0 + 0.34 * Ww) & (c[:, 1] > y0 + 0.14 * Hh) & (c[:, 1] < y1 - 0.14 * Hh),
          "right": (c[:, 0] > x1 - 0.34 * Ww) & (c[:, 1] > y0 + 0.14 * Hh) & (c[:, 1] < y1 - 0.14 * Hh),
          "upper": c[:, 1] > y0 + 0.5 * Hh}[where]
    best, i, n = (0, 1), 0, len(c)
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j + 1
        else:
            i += 1
    sa = OL.arclen(c)
    u0, u1 = sa[best[0]] / sa[-1], sa[best[1]] / sa[-1]
    if (u1 - u0) * sa[-1] > longest * W:
        mid, half = (u0 + u1) / 2, longest * W / sa[-1] / 2
        u0, u1 = mid - half, mid + half
    return u0, u1


def variant(low_h, cap_h, swap=None):
    def fn(ch, st, W, centre):
        ranked = sorted(st, key=lambda s: -OL.arclen(s.c)[-1])
        B = np.vstack([s.c for s in st])
        box = (B.min(0), B.max(0))
        spec, i, where = TABLE[ch]
        if swap:
            spec = swap.get(spec, spec)
        s = ranked[0] if i == "L" else st[min(i, len(st) - 1)]
        h = low_h if ch.islower() else cap_h
        if isinstance(where, tuple):
            return st, [along(spec, s, where[0], where[1], W, centre, height=h, sink=0.14)]
        wide = 0.8 * (box[1][0] - box[0][0]) / W            # a crest stays within its letter's width
        u0, u1 = region(s, where, box, W, 4.6 if ch.islower() else min(4.8, max(wide, 2.6)))
        return st, [along(spec, s, u0, u1, W, centre, height=h, sink=0.14, toward=TOWARD[where])]
    fn.ordered = True
    return fn


def strip_set(text, fn, H=1700, gap=130):
    """Letters at an even spacing of their bodies, ornaments overlapping into the gaps as they would in a font."""
    h = int(H * SCALE)
    cv = np.zeros((h, int(SCALE * 16000)), bool)
    x = 40.0
    for ch in text:
        if ch == " ":
            x += SPACE
            continue
        W = CU.weight(ch, BD.main_stroke)
        S = CU._fitted(ch, W)
        OL.outline(S, W)
        ordered = [q for q in S if isinstance(q, OL.Stroke) and not getattr(q, "free", False)]
        _, centre = _main(S)
        _, pieces = fn(ch, ordered, W, centre)
        cs = OL.outline(S, W)
        B = np.vstack(cs)
        ox = x - B[:, 0].min()

        def fill(group):
            t = np.zeros_like(cv)
            for C in group:
                im = Image.new("1", (cv.shape[1], h), 0)
                ImageDraw.Draw(im).polygon([((p[0] + ox) * SCALE, h - (p[1] + 420) * SCALE) for p in C], fill=1)
                t ^= np.asarray(im, bool)
            return t
        cv |= fill(cs)
        for g in pieces:
            cv |= fill(g)
        x += B[:, 0].max() - B[:, 0].min() + gap
    return cv[:, :int(x * SCALE) + 30]


def variants():
    BD.configure("Curated")
    out = HERE / "proofs" / "parts"
    lines = ["abcdefghijklm", "nopqrstuvwxyz", "ABCDEFGHIJKLM", "NOPQRSTUVWXYZ", "0123456789", "Grammar of Ornament"]
    antlers = {COMB: ("curved-comb/15", 0.36, 0.66), RIB: ANTLER_C, BARBS: ANTLER_A, BARBS2: ANTLER_D,
               FINGER_ROW: ANTLER_B, FINGER_ROW2: ("curved-comb/8", 0.0, 0.30)}
    large = {COMB: ("wavy-comb/1", 0.13, 0.34, True), RIB: ("curved-branching-line/10", 0.28, 0.51, True),
             BARBS: ("curved-sprig/14", 0.15, 0.40, True), BARBS2: ("curved-comb/6", 0.50, 0.77, True),
             FINGER_ROW: ("straight-comb/1", 0.26, 0.70, True), FINGER_ROW2: ("curved-feather/3", 0.20, 0.54, True)}
    feathers = {COMB: ("curved-comb/12", 0.36, 0.66), RIB: ("wavy-feather/14", 0.36, 0.66),
                BARBS: ("straight-feather/8", 0.36, 0.66), BARBS2: ("curved-feather/9", 0.54, 0.84),
                FINGER_ROW: ("curved-feather/14", 0.0, 0.30), FINGER_ROW2: ("wavy-feather/8", 0.18, 0.48)}
    hands = {COMB: FINGER_ROW2, RIB: ("curved-feather/12", 0.0, 0.30), BARBS: ("curved-comb/12", 0.18, 0.48),
             BARBS2: ("straight-comb/12", 0.54, 0.84), FINGER_ROW: ("curved-comb/12", 0.54, 0.84)}
    for name, fn in (("variant-d-base", variant(1.0, 0.9)), ("variant-e-antlers", variant(1.0, 0.9, antlers)),
                     ("variant-f-large", variant(0.8, 0.8, large)), ("variant-g-feathers", variant(1.0, 0.9, feathers)),
                     ("variant-h-hands", variant(1.0, 0.9, hands))):
        st = [strip_set(t, fn) for t in lines]
        w = max(q.shape[1] for q in st)
        Image.fromarray(~np.vstack([np.pad(q, ((0, 0), (0, w - q.shape[1]))) for q in st])).save(out / f"{name}.png")
        print(f"wrote {name}", flush=True)


# ── the combined alphabet: each letter takes the kind of ornament that worked on it ──────────────────
ARC = ("wavy-comb/1", 0.13, 0.34, True)            # the doubled arc, from the large fragments
ANTLER = ANTLER_C
HAND = FINGER_ROW2
HAND_F = ("curved-comb/12", 0.18, 0.48)
STEM_COMB = ("straight-comb/12", 0.54, 0.84)
# letter → (fragment, stroke, where, height)
COMBO = {**{k: (*v, 1.0) for k, v in LOWER.items()}, **{k: (*v, 0.9) for k, v in UPPER.items()},
         **{k: (*v, 0.75) for k, v in FIGS.items()}}
COMBO.update({
    "m": (HAND, "L", (0.3, 0.72), 1.0), "n": (HAND, "L", (0.3, 0.72), 1.0), "r": (HAND, 0, "top", 1.0), "f": (HAND_F, 0, "top", 1.0),
    "s": (ANTLER, 0, "top", 0.9), "d": (STEM_COMB, 1, "upper", 0.8), "h": (STEM_COMB, 0, "upper", 0.8),
    "k": (STEM_COMB, 0, "upper", 0.8), "l": (STEM_COMB, 0, "left", 0.8), "i": (STEM_COMB, 0, "left", 0.7),
    "j": (STEM_COMB, 0, "left", 0.7), "t": (STEM_COMB, 0, "upper", 0.7), "x": (ANTLER_D, 0, "left", 0.8),
    "O": (ARC, 0, "top", 0.8), "P": (ARC, 1, "top", 0.8), "Q": (ARC, 0, "top", 0.8), "R": (ARC, 1, "top", 0.8),
    "T": (ARC, 1, "top", 0.8), "S": (ANTLER, 0, "top", 0.9), "H": (RIB, 0, "left", 0.9), "I": (STEM_COMB, 0, "left", 0.8),
    "X": (ANTLER_D, 0, "left", 0.8), "4": (STEM_COMB, 0, "left", 0.7)})


def combo(ch, st, W, centre):
    ranked = sorted(st, key=lambda s: -OL.arclen(s.c)[-1])
    B = np.vstack([s.c for s in st])
    box = (B.min(0), B.max(0))
    spec, i, where, h = COMBO[ch]
    s = ranked[0] if i == "L" else st[min(i, len(st) - 1)]
    if isinstance(where, tuple):
        return st, [along(spec, s, where[0], where[1], W, centre, height=h, sink=0.14)]
    wide = 0.8 * (box[1][0] - box[0][0]) / W
    u0, u1 = region(s, where, box, W, 4.6 if ch.islower() else min(4.8, max(wide, 2.6)))
    return st, [along(spec, s, u0, u1, W, centre, height=h, sink=0.14, toward=TOWARD[where])]


def combined(name="combined"):
    BD.configure("Curated")
    lines = ["abcdefghijklm", "nopqrstuvwxyz", "ABCDEFGHIJKLM", "NOPQRSTUVWXYZ", "0123456789",
             "Grammar of Ornament", "The Quick Brown Fox"]
    st = [strip_set(t, combo) for t in lines]
    w = max(q.shape[1] for q in st)
    Image.fromarray(~np.vstack([np.pad(q, ((0, 0), (0, w - q.shape[1]))) for q in st])).save(
        HERE / "proofs" / "parts" / f"{name}.png")
    print(f"wrote {name}", flush=True)
