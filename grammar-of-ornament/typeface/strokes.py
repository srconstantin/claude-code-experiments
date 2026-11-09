"""
Neuron — trials for an Uncial that isn't medieval: its skeletons drawn with strokelib's strokes rather than a pen.
Sample strips only.

  leaf     every stroke swells in the middle and runs to a point at each free end
  drop     the reverse of a tapered stroke: thin where it joins, heavy at its free end
  wave     strokes follow real wave strokes' small jogs, with the tapered profile
  hook     curves gather their turning into elbows (strokelib's hooks), with the tapered profile
  eye      leaf strokes whose free ends roll into a small loop (strokelib's loops, curved-sprig/13's spiral)

  .venv/bin/python typeface/strokes.py     → typeface/proofs/variants/<trial>.png
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as BD          # noqa: E402
import curated as CU        # noqa: E402
import formal as FM         # noqa: E402
import outline as OL        # noqa: E402
import silhouettes as SI    # noqa: E402

LINES = ["Grammar of Ornament", "hamburgefonstiv", "The Quick Brown Fox", "SAPPHIRE MOON 1984"]
SCALE, SPACE, GAP = 0.34, 270, 110
TAPERED = [f"strokelib:family-08-short-tapered-straight-branch-fine/{i}" for i in (2, 4, 7, 8)]
FLARED = [f"strokelib:family-10-short-flared-straight-branch-fine/{i}" for i in (1, 2, 5, 9)]
WAVES = [f"strokelib:family-07-short-wave-branch-fine/{i}" for i in (2, 4, 6, 8)]
FLOOR = 0.3                 # no line thinner than this, in weights
SWELL = 2.0                 # exponent on the measured width profile
TIP = 0.14                  # tip width, as a share of the stroke there
GATHER = 2.2                # exponent sharpening a curve's turning into an elbow


@lru_cache(maxsize=None)
def profile(ref):
    """The stroke's width along its length, thickest = 1, heaviest end last (for a one-ended stroke)."""
    p = SI.silhouette(ref)["profile"]
    p = np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, len(p)), p)
    p = ndi.gaussian_filter1d(p, 3, mode="nearest") / p.max()
    return p if p[-12:].mean() >= p[:12].mean() else p[::-1]


@lru_cache(maxsize=None)
def jog(ref):
    """The stroke's spine as sideways offset from its chord, in shares of its length, ends at zero."""
    sp = SI.silhouette(ref)["spine"]
    sa = OL.arclen(sp)
    y = np.interp(np.linspace(0, sa[-1], 60), sa, sp[:, 1])
    return ndi.gaussian_filter1d(y - np.linspace(y[0], y[-1], 60), 3, mode="nearest") / sa[-1]


def gather(q):
    """The curve with its turning concentrated, its ends left where they were."""
    d = np.diff(q, axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = ndi.gaussian_filter1d(np.diff(th), 4, mode="nearest")
    if np.abs(k).sum() < 0.9:
        return q
    k2 = np.sign(k) * np.abs(k) ** GATHER
    k2 = ndi.gaussian_filter1d(k2, 2, mode="nearest")
    for sign in (1, -1):                         # each way of turning keeps its own total
        on = np.sign(k2) == sign
        if on.any():
            k2[on] *= k[np.sign(k) == sign].sum() / k2[on].sum() if (np.sign(k) == sign).any() else 0
    t = th[0] + np.concatenate([[0], np.cumsum(k2)])
    z = np.concatenate([[0], np.cumsum(np.hypot(*d.T) * np.exp(1j * t))])
    a, b = complex(*q[0]), complex(*q[-1])
    z = a + (z - z[0]) * (b - a) / (z[-1] - z[0])
    return np.stack([z.real, z.imag], 1)


def _ends(s, q, W, others):
    near = lambda e: bool(others) and min(float(np.min(np.hypot(*(o - e).T))) for o in others) < 0.6 * W
    d = q[-1] - q[0]
    upright = s.kind == "crescent" and abs(d[1]) > 0.8 * np.hypot(*d)
    return (False, False) if upright else (s.taper0 or near(q[0]), s.taper1 or near(q[-1]))


def _curl(e, h, W, size=2.6):
    """A small loop off a free end: the inner turn of curved-sprig/13, rolling left."""
    sp = CU.skel("curved-sprig/13")["spine"]
    z = (sp[:, 0] + 1j * sp[:, 1])[int(0.62 * len(sp)):]
    a = np.unwrap(np.angle(np.diff(z)))
    if a[-1] < a[0]:
        z = np.conj(z)
    m = complex(*h) / abs(complex(*h)) / ((z[3] - z[0]) / abs(z[3] - z[0])) * size * W / np.sum(np.abs(np.diff(z)))
    r = complex(*e) + m * (z - z[0])
    return np.stack([r.real, r.imag], 1)


def stroke(s, W, others, mode, i):
    q = FM._centre(s, W)
    if s.closed:
        tg = OL.tangents(q, True)
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        a = np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean())
        w = W * (0.66 + 0.56 * np.cos(a + (2.4 if mode != "drop" else -0.7)))
        return [[q + nr * (w / 2)[:, None], (q - nr * (w / 2)[:, None])[::-1]]]
    j0, j1 = _ends(s, q, W, others)
    if mode == "hook" and s.kind != "crescent":
        q = gather(q)
    L = OL.arclen(q)
    u = L / L[-1]
    if mode == "wave":
        tg = OL.tangents(q, False)
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        y = np.interp(u, np.linspace(0, 1, 60), jog(WAVES[i % len(WAVES)]))
        y = y / max(np.abs(y).max(), 1e-6) * min(0.32 * W, 0.06 * L[-1])
        q = q + nr * y[:, None]
    extra = []
    if mode in ("leaf", "eye"):
        p = 0.25 + 0.75 * np.sin(np.pi * u) ** 0.8
    else:
        p = np.interp(u, np.linspace(0, 1, 60), profile(TAPERED[2])) ** SWELL
        heavy_free = mode == "drop"
        free_end_last = (j0 and not j1) or (j0 == j1 and q[0, 1] > q[-1, 1])
        if free_end_last != heavy_free:          # tapered: heavy at the join; drop: heavy at the free end
            p = p[::-1]
        if heavy_free:
            p = p ** 0.6
    w = W * 1.3 * s.w * np.maximum(p, 0.3)
    for x, joined, end in ((L, j0, 0), (L[-1] - L, j1, 1)):
        if joined:
            w = w * (0.5 + 0.5 * FM._ease(x / (1.4 * W)))
        elif mode != "drop":
            w = w * (TIP + (1 - TIP) * FM._ease(x / (1.6 * W)))
        if mode == "eye" and not joined and L[-1] > 3 * W:
            h = (q[-1] - q[-5]) if end else (q[0] - q[4])
            c = _curl(q[-1] if end else q[0], h, W)
            ct = OL.tangents(c, False)
            cn = np.stack([-ct[:, 1], ct[:, 0]], 1)
            cw = W * 0.42 * np.linspace(1, 0.5, len(c))
            extra.append(np.vstack([c + cn * (cw / 2)[:, None], OL._cap(c[-1], ct[-1], cw[-1] / 2),
                                    (c - cn * (cw / 2)[:, None])[::-1]]))
    tg = OL.tangents(q, False)
    nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
    return [np.vstack([q + nr * (w / 2)[:, None], OL._cap(q[-1], tg[-1], w[-1] / 2),
                       (q - nr * (w / 2)[:, None])[::-1], OL._cap(q[0], -tg[0], w[0] / 2)])] + extra


def letter(ch, W, mode):
    S = FM._strokes(ch, W, False)
    if ch in FM.NO_SPUR:
        S = [s for s in S if not getattr(s, "free", False)]
    OL.outline(S, W)
    lines = [s.c for s in S if isinstance(s, OL.Stroke)]
    polys = []
    for i, s in enumerate(S):
        if isinstance(s, OL.Dot):
            t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            polys.append(np.stack([s.x + 0.6 * W * np.cos(t), s.y - 40 + 0.6 * W * np.sin(t)], 1))
        else:
            polys += stroke(s, W, [c for c in lines if c is not s.c], mode, ord(ch) + i)
    return OL.union(polys)


def strip(text, mode, H=1700):
    h = int(H * SCALE)
    cv = np.zeros((h, int(SCALE * 18000)), bool)
    x = 40.0
    for ch in text:
        if ch == " ":
            x += SPACE
            continue
        cs = letter(ch, CU.weight(ch, BD.main_stroke), mode)
        B = np.vstack(cs)
        ox = x - B[:, 0].min()
        m = np.zeros_like(cv)
        for C in cs:
            im = Image.new("1", (cv.shape[1], h), 0)
            ImageDraw.Draw(im).polygon([((q[0] + ox) * SCALE, h - (q[1] + 420) * SCALE) for q in C], fill=1)
            m ^= np.asarray(im, bool)
        cv |= m
        x += B[:, 0].max() - B[:, 0].min() + GAP
    return cv[:, :int(x * SCALE) + 30]


def main():
    BD.configure("Uncial")
    out = HERE / "proofs" / "variants"
    out.mkdir(parents=True, exist_ok=True)
    for mode in ("leaf", "drop", "wave", "hook", "eye"):
        st = [strip(t, mode) for t in LINES]
        w = max(q.shape[1] for q in st)
        Image.fromarray(~np.vstack([np.pad(q, ((0, 0), (0, w - q.shape[1]))) for q in st])).save(out / f"{mode}.png")
        print(f"wrote {mode}", flush=True)


if __name__ == "__main__":
    main()
