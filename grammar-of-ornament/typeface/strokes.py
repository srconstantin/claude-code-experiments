"""
Neuron — trials taking thickness and curvature from strokelib instead of from a pen. Sample strips only.

A broad pen makes thickness depend on direction. The library's strokes vary along their length instead: heavy at
one end and tapering (tapered families), pinched at the waist (flared), with curvature gathered at an elbow (hook)
or small jogs along the way (wave). Each trial redraws Uncial's skeletons that way, using real strokes' measured
width profiles and spines.

  club     each stroke takes a tapered stroke's width profile, heavy end away from its joins
  flared   the same with flared strokes' profiles: swollen ends, pinched waist
  elbow    club, with each curve's turning gathered into an elbow
  wave     club, with stems following a real wave stroke's spine

  .venv/bin/python typeface/strokes.py     → typeface/proofs/strokes/<trial>.png
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

LINES = ["Grammar of Ornament", "hamburgefonstiv", "The Quick Brown Fox"]
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


def stroke(s, W, pool, i, elbow=False, wave=False):
    q = FM._centre(s, W)
    if s.closed:                                 # the library's loops: heavier on one side
        n = len(q)
        tg = OL.tangents(q, True)
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        a = np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean())
        w = W * 1.05 * (0.62 + 0.6 * np.cos(a + 2.4))
        return [[q + nr * (w / 2)[:, None], (q - nr * (w / 2)[:, None])[::-1]]]
    if elbow:
        q = gather(q)
    upright = s.kind == "crescent" and abs((q[-1] - q[0])[1]) > 0.8 * np.hypot(*(q[-1] - q[0]))
    L = OL.arclen(q)
    if wave and upright:
        tg = OL.tangents(q, False)
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        y = np.interp(L / L[-1], np.linspace(0, 1, 60), jog(WAVES[i % len(WAVES)]))
        q = q + nr * (y * L[-1] * 0.8)[:, None]
    p = np.interp(L / L[-1], np.linspace(0, 1, 60), profile(pool[i % len(pool)]))
    # the heavy end goes to the join, the fine end runs out free; a stroke free or joined at both is heavy low down
    if s.taper0 and not s.taper1 or (s.taper0 == s.taper1 and q[0, 1] < q[-1, 1]):
        p = p[::-1]
    p = p ** SWELL                               # the measured swelling, exaggerated for display sizes
    w = W * 1.45 * np.maximum(p, FLOOR / 1.45)
    w = w * (TIP + (1 - TIP) * FM._ease(np.minimum(L, L[-1] - L) / (1.6 * W)))    # sharp tips, as in Uncial
    tg = OL.tangents(q, False)
    nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
    return [np.vstack([q + nr * (w / 2)[:, None], (q - nr * (w / 2)[:, None])[::-1]])]


def letter(ch, W, **kw):
    S = FM._strokes(ch, W, False)
    if ch in FM.NO_SPUR:
        S = [s for s in S if not getattr(s, "free", False)]
    OL.outline(S, W)
    polys = []
    for i, s in enumerate(S):
        polys += [FM._lozenge(s, W)] if isinstance(s, OL.Dot) else stroke(s, W, i=ord(ch) + i, **kw)
    return OL.union(polys)


def strip(text, H=1600, **kw):
    h = int(H * SCALE)
    cv = np.zeros((h, int(SCALE * 18000)), bool)
    x = 40.0
    for ch in text:
        if ch == " ":
            x += SPACE
            continue
        cs = letter(ch, CU.weight(ch, BD.main_stroke), **kw)
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
    out = HERE / "proofs" / "strokes"
    out.mkdir(parents=True, exist_ok=True)
    for name, kw in (("club", dict(pool=TAPERED)), ("flared", dict(pool=FLARED)),
                     ("elbow", dict(pool=TAPERED, elbow=True)), ("wave", dict(pool=TAPERED, wave=True))):
        st = [strip(t, **kw) for t in LINES]
        w = max(q.shape[1] for q in st)
        Image.fromarray(~np.vstack([np.pad(q, ((0, 0), (0, w - q.shape[1]))) for q in st])).save(out / f"{name}.png")
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
