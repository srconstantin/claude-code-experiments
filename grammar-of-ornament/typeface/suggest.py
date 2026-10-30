"""
Neuron — candidate pieces for each letter (a picking aid for curated.py, not a font).

Every piece's smoothed skeleton (curated.skel: spine and branches) is drawn as
lines in 24 rotations, plain and mirrored, stretched to the letter's
proportions (within STRETCH), and compared with the letter's Bare skeleton
drawn the same way (blurred, normalized correlation). The best few per letter
go on a sheet, labelled with what curated.placed_box needs (ref, mirror, rot).

  .venv/bin/python typeface/suggest.py "abc…"   → typeface/proofs/curated/suggest-<letters>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curated as CU         # noqa: E402
import dress as DR           # noqa: E402
import glyphs as GL          # noqa: E402
import outline as OL         # noqa: E402
import pieces as PC          # noqa: E402

FAMILIES = ("curved-branching-line", "wavy-branching-line", "straight-branching-line", "curved-sprig",
            "wavy-sprig", "straight-sprig", "fork", "loop", "curved-band", "wavy-band", "curved-comb",
            "curved-feather", "wavy-comb", "wavy-feather")
N = 56
LINE = 0.13            # line width, relative to the box
STRETCH = (0.6, 1.6)   # a piece may be stretched sideways this much to take the letter's proportions
ROTS = range(0, 360, 15)
TOP = 8


def draw(lines, lo, hi, aspect):
    """Lines (list of N×2) fitted into a box of the given width/height aspect, drawn and blurred."""
    h = N - 8
    w = h * aspect
    if w > N - 8:
        w, h = N - 8, (N - 8) / aspect
    s = np.array([w, h]) / np.maximum(hi - lo, 1e-9)
    off = (np.array([N, N]) - [w, h]) / 2
    im = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(im)
    for P in lines:
        q = (P - lo) * s + off
        d.line([(x, N - y) for x, y in q], fill=255, width=max(2, int(LINE * N)), joint="curve")
    a = ndi.gaussian_filter(np.asarray(im, float) / 255, 1.2)
    a -= a.mean()
    return a / max(np.linalg.norm(a), 1e-9)


def letter_lines(ch):
    S = [s for s in GL.GLYPHS[ch] if isinstance(s, OL.Stroke)]
    DR.dress(ch, S, 100, None, None)
    return [np.vstack([s.c, s.c[:1]]) if s.closed else s.c for s in S]


def piece_lines(ref):
    sk = CU.skel(ref)
    return [sk["spine"]] + [t["pts"] for t in sk["teeth"]]


def main(letters: str) -> None:
    refs = [f"{fam}/{int(f.stem)}" for fam in FAMILIES for f in sorted((PC.ROOT / fam).glob("[0-9][0-9].svg"))]
    pieces = []
    for r in refs:
        try:
            pieces.append((r, piece_lines(r)))
        except Exception:
            continue
    print(f"  {len(pieces)} pieces", flush=True)
    targets = {}
    for ch in letters:
        L = letter_lines(ch)
        P = np.vstack(L)
        lo, hi = P.min(0), P.max(0)
        targets[ch] = (draw(L, lo, hi, (hi - lo)[0] / (hi - lo)[1]), (hi - lo)[0] / (hi - lo)[1])
    best = {ch: [] for ch in letters}
    for k, (ref, lines) in enumerate(pieces):
        for mir in (False, True):
            for rot in ROTS:
                z = [(l[:, 0] + 1j * l[:, 1]) for l in lines]
                z = [(np.conj(q) if mir else q) * np.exp(1j * np.radians(rot)) for q in z]
                ls = [np.stack([q.real, q.imag], 1) for q in z]
                P = np.vstack(ls)
                lo, hi = P.min(0), P.max(0)
                own = (hi - lo)[0] / max((hi - lo)[1], 1e-9)
                for ch in letters:
                    tgt, asp = targets[ch]
                    if not (STRETCH[0] <= asp / max(own, 1e-9) <= STRETCH[1]):
                        continue
                    sc = float(np.sum(draw(ls, lo, hi, asp) * tgt))
                    best[ch].append((sc, ref, mir, rot, asp / own))
        if k % 40 == 0:
            print(f"  {k}/{len(pieces)}", flush=True)
    # sheet
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    cell = 150
    sheet = Image.new("L", (cell * (TOP + 1), cell * len(letters)), 255)
    d = ImageDraw.Draw(sheet)
    for r, ch in enumerate(letters):
        d.text((50, r * cell + 50), ch, font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60), fill=0)
        seen, picks = set(), []
        for c in sorted(best[ch], key=lambda c: -c[0]):
            if c[1] in seen:
                continue
            seen.add(c[1])
            picks.append(c)
            if len(picks) == TOP:
                break
        for j, (sc, ref, mir, rot, st) in enumerate(picks):
            box = (0, 0, None, 500)
            S = CU.placed_box(ref, box, rot=rot, mirror=mir, stretch=st)
            cs = OL.outline(S, 70)
            P = np.vstack(cs)
            lo, hi = P.min(0), P.max(0)
            k = (cell - 30) / max(hi - lo)
            im = Image.new("1", (cell, cell), 0)
            m = np.zeros((cell, cell), bool)
            for C in cs:
                t = Image.new("1", (cell, cell), 0)
                ImageDraw.Draw(t).polygon([((x - lo[0]) * k + 5, cell - 22 - (y - lo[1]) * k) for x, y in C], fill=1)
                m ^= np.asarray(t, bool)
            sheet.paste(Image.fromarray(~m).convert("L"), ((j + 1) * cell, r * cell))
            d.text(((j + 1) * cell + 3, r * cell + cell - 16), f"{ref} {'M' if mir else ''}{rot} x{st:.2f}", font=f, fill=90)
    out = HERE / "proofs" / "curated" / f"suggest-{''.join(c if c.isalnum() else '_' for c in letters)}.png"
    sheet.save(out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
