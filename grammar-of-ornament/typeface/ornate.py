"""
Neuron Ornate — Curated's letters carrying real fragments of the compounds.

A whole compound at stroke weight is as big as a letter, so each letter takes a
part of one (parts.py): a stretch of a comb, feather or branch with its own outline.
The fragment is laid along a stretch of one stroke, its dense edge sunk into the
stroke, its long axis bending to follow the stroke's curve, its teeth standing away
from the letter. Fragments are chosen letter by letter (ORN): different kinds suit
different letters.

  ribs, barbs   low fine combs on bowls and curves
  hands         fingers, on arches
  arc           a doubled arc, over round capitals
  antlers       branching, on S
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curated as CU        # noqa: E402
import outline as OL        # noqa: E402
import parts as PT          # noqa: E402

SINK = 0.14                 # the fragment's dense edge sits this far (in weights) inside the stroke's centre side
COMB = ("straight-comb/12", 0.36, 0.66)
RIB = ("curved-feather/14", 0.36, 0.66)
BARBS, BARBS2 = ("straight-feather/10", 0.0, 0.30), ("straight-feather/8", 0.54, 0.84)
ROW = ("wavy-feather/6", 0.0, 0.30)
HAND = ("wavy-feather/6", 0.18, 0.48)
HAND_F = ("curved-comb/12", 0.18, 0.48)
STEM = ("straight-comb/12", 0.54, 0.84)
ANTLER, ANTLER_S = ("curved-comb/11", 0.36, 0.66), ("curved-comb/11", 0.18, 0.48)
ARC = ("wavy-comb/1", 0.13, 0.34, True)
TOWARD = {"top": (0, 1), "left": (-1, 0), "right": (1, 0), "upper": (-1, 0), "lower": (1, 0)}

# letter → (fragment, stroke, where, height, longest): stroke "L" is the longest; where is a stretch (u0, u1) or a
# region of the letter, and the fragment stands out toward that region; height scales how far it stands out;
# longest (in weights, optional) limits its length
ORN = {
    "a": (BARBS, "L", (0.2, 0.6), 1.0), "b": (BARBS2, "L", (0.5, 0.97), 1.0), "c": (RIB, 0, "left", 0.8),
    "d": (STEM, 1, "upper", 0.8), "e": (RIB, 0, "left", 0.65), "f": (HAND_F, 0, "top", 0.5, 3.0),
    "g": (RIB, 0, "left", 0.65), "h": (STEM, 0, "upper", 0.8), "i": (STEM, 0, "left", 0.7), "j": (STEM, 0, "left", 0.7),
    "k": (STEM, 0, "upper", 0.8), "l": (STEM, 0, "left", 0.8), "m": (HAND, "L", (0.3, 0.72), 1.0),
    "n": (HAND, "L", (0.3, 0.72), 1.0), "o": (RIB, 0, "left", 0.8), "p": (BARBS2, 1, "right", 0.9),
    "q": (BARBS2, 0, "left", 0.9), "r": (HAND, 0, "top", 1.0), "s": (ANTLER, 0, "top", 0.7, 3.2),
    "t": (STEM, 0, "upper", 0.7), "u": (BARBS, 0, "left", 0.75), "v": (ROW, "L", (0.12, 0.5), 0.7),
    "w": (ROW, 0, "left", 0.7), "x": (ANTLER_S, 0, (0.1, 0.5), 0.7), "y": (BARBS, 0, "left", 0.75), "z": (RIB, 0, "top", 0.9),
    "A": (ROW, 0, "left", 0.65), "B": (BARBS, 1, "right", 0.9), "C": (COMB, 0, "top", 0.9), "D": (COMB, 1, "top", 0.9),
    "E": (ROW, 0, "left", 0.8), "F": (RIB, 0, "left", 0.9), "G": (COMB, 0, "top", 0.9), "H": (RIB, 0, "left", 0.9),
    "I": (STEM, 0, "left", 0.8), "J": (RIB, 0, "left", 0.9), "K": (RIB, 0, "left", 0.9), "L": (RIB, 0, "left", 0.9),
    "M": (COMB, 0, "top", 0.9), "N": (COMB, 0, "top", 0.9), "O": (ARC, 0, "top", 0.8), "P": (ARC, 1, "top", 0.8),
    "Q": (ARC, 0, "top", 0.8), "R": (ARC, 1, "top", 0.8), "S": (ANTLER, 0, "top", 0.9), "T": (ARC, 1, "top", 0.8),
    "U": (ROW, 0, "left", 0.7), "V": (ROW, 0, "left", 0.7), "W": (ROW, 0, "left", 0.7), "X": (ANTLER_S, 0, (0.1, 0.5), 0.7),
    "Y": (ROW, 0, "left", 0.7), "Z": (RIB, 0, "top", 0.9),
    "0": (RIB, 0, "left", 0.75), "1": (BARBS2, 0, (0.3, 0.9), 0.8), "2": (RIB, 0, "top", 0.75), "3": (RIB, 0, "top", 0.75),
    "4": (STEM, 0, (0.35, 0.92), 0.7), "5": (RIB, 0, "right", 0.7), "6": (RIB, 0, "left", 0.7), "7": (RIB, 0, "top", 0.75),
    "8": (RIB, 1, "left", 0.75), "9": (RIB, 0, "right", 0.7),
}


@lru_cache(maxsize=None)
def _fragment(ref, a, b, large=False, fat=0.0):
    p = PT.part(ref, a, b, 30, 0.06, fat) if large else PT.part(ref, a, b, 14, 0.1, fat)
    return p["contours"]


def _ink_centre(cs):
    P = np.vstack(cs)
    lo, hi = P.min(0), P.max(0)
    k = 200 / max(hi - lo)
    Wd, Hd = ((hi - lo) * k).astype(int) + 3
    m = np.zeros((Hd, Wd), bool)
    for C in cs:
        im = Image.new("1", (Wd, Hd), 0)
        ImageDraw.Draw(im).polygon([tuple(q) for q in (C - lo) * k], fill=1)
        m ^= np.asarray(im, bool)
    ys, xs = np.nonzero(m)
    return lo + np.array([xs.mean(), ys.mean()]) / k


def along(spec, s, u0, u1, W, centre, height=1.0, toward=None, seat=0.0):
    """The fragment laid along stretch u0..u1 of stroke s, its long axis following the stroke's curve."""
    cs = _fragment(*spec)
    P = np.vstack(cs)
    m = P.mean(0)
    _, V = np.linalg.eigh(np.cov((P - m).T))
    ax, ac = V[:, 1], V[:, 0]
    ic = _ink_centre(cs)
    x, y = (P - m) @ ax, (P - m) @ ac
    sign = 1 if (ic - m) @ ac < (y.min() + y.max()) / 2 else -1      # +y runs from the dense edge outward
    y0 = y.min() if sign > 0 else y.max()
    c = np.vstack([s.c, s.c[:1]]) if s.closed else s.c
    sa = OL.arclen(c)
    s0, s1 = u0 * sa[-1], u1 * sa[-1]
    k = (s1 - s0) / (x.max() - x.min())
    tg = OL.tangents(c, False)
    nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
    i = int(np.searchsorted(sa, (s0 + s1) / 2))
    sd = 1 if ((c[i] - centre) if toward is None else np.asarray(toward, float)) @ nr[i] > 0 else -1
    out = []
    for C in cs:
        X = ((C - m) @ ax - x.min()) / (x.max() - x.min())
        Y = ((C - m) @ ac - y0) * sign * k * height
        if seat:                                # the ends fold down into the stroke, leaving no horns
            Y = Y * np.sin(0.5 * np.pi * np.clip(np.minimum(X, 1 - X) / seat, 0, 1)) ** 1.2
        d = s0 + X * (s1 - s0)
        q = np.stack([np.interp(d, sa, c[:, j]) for j in (0, 1)], 1)
        n = np.stack([np.interp(d, sa, nr[:, j]) for j in (0, 1)], 1)
        out.append(q + n * (sd * (Y - SINK * W))[:, None])
    return out


def region(s, where, box, W, longest):
    """The longest run of the stroke inside a region of the letter, as a stretch, at most `longest` weights."""
    (x0, y0), (x1, y1) = box
    c = s.c
    Hh, Ww = y1 - y0, x1 - x0
    ok = {"top": c[:, 1] > y1 - 0.3 * Hh,
          "left": (c[:, 0] < x0 + 0.34 * Ww) & (c[:, 1] > y0 + 0.14 * Hh) & (c[:, 1] < y1 - 0.14 * Hh),
          "right": (c[:, 0] > x1 - 0.34 * Ww) & (c[:, 1] > y0 + 0.14 * Hh) & (c[:, 1] < y1 - 0.14 * Hh),
          "upper": c[:, 1] > y0 + 0.5 * Hh, "lower": c[:, 1] < y0 + 0.5 * Hh}[where]
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


def fragment(ch: str, S: list, W: float, table: dict = ORN, seat: float = 0.0) -> list:
    """The letter's fragment, placed on its strokes."""
    st = [s for s in S if isinstance(s, OL.Stroke) and not getattr(s, "free", False)]
    B = np.vstack([s.c for s in st])
    box = (B.min(0), B.max(0))
    centre = B.mean(0)
    if table[ch] is None:
        return None, None
    spec, i, where, h, *rest = table[ch]
    s = max(st, key=lambda q: OL.arclen(q.c)[-1]) if i == "L" else st[min(i, len(st) - 1)]
    if isinstance(where, tuple):
        return s, along(spec, s, where[0], where[1], W, centre, h, seat=seat if spec[0] == ARC[0] else 0.0)
    wide = 0.8 * (box[1][0] - box[0][0]) / W            # a crest stays within its letter's width
    longest = rest[0] if rest else (4.6 if ch.islower() else min(4.8, max(wide, 2.6)))
    u0, u1 = region(s, where, box, W, longest)
    return s, along(spec, s, u0, u1, W, centre, h, TOWARD[where], seat if spec[0] == ARC[0] else 0.0)


def glyph(ch: str, W: float) -> list[np.ndarray]:
    S = CU._fitted(ch, W)
    OL.outline(S, W)                    # (sets the joins)
    s, piece = fragment(ch, S, W)
    s.grafts = [piece]
    area = lambda C: 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    return [C for C in OL.outline(S, W, pinhole=0.2) if abs(area(C)) > 200]
