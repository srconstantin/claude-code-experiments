"""
Neuron — letters already present in the compound shapes.

Every compound piece (compounds/vgg16features/*/NN.svg, smoothed by
silhouettes.py) is compared with every glyph of the Bare font, in all eight
orientations (four rotations, mirrored or not). Both are scaled to fit the
same box with their proportions kept, blurred a little (so near-misses still
count), and scored by normalized correlation. For each letter the sheet shows
the best-matching pieces, turned the way they match.

  .venv/bin/python typeface/found.py     → typeface/proofs/found-letters.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glyphs as GL          # noqa: E402
import pieces as PC          # noqa: E402
import silhouettes as SI     # noqa: E402

N = 64
BLUR = 1.6
TOP = 6


def norm(mask: np.ndarray) -> np.ndarray:
    """A mask cropped to its ink, scaled (proportions kept) into an N×N box, centred, blurred, unit-normalized."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((N, N))
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    s = (N - 6) / max(h, w)
    im = Image.fromarray((m * 255).astype(np.uint8)).resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    out = np.zeros((N, N))
    a = np.asarray(im, float) / 255
    y0, x0 = (N - a.shape[0]) // 2, (N - a.shape[1]) // 2
    out[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    out = ndi.gaussian_filter(out, BLUR)
    out -= out.mean()
    return out / max(np.linalg.norm(out), 1e-9)


def piece_mask(ref: str, px: int = 240) -> np.ndarray:
    s = SI.silhouette(ref)
    P = np.vstack(s["contours"])
    lo, hi = P.min(0), P.max(0)
    k = (px - 4) / max(hi - lo)
    W, H = int((hi - lo)[0] * k) + 4, int((hi - lo)[1] * k) + 4
    m = np.zeros((H, W), bool)
    for C in s["contours"]:
        im = Image.new("1", (W, H), 0)
        ImageDraw.Draw(im).polygon([((x - lo[0]) * k + 2, H - ((y - lo[1]) * k + 2)) for x, y in C], fill=1)
        m ^= np.asarray(im, bool)
    return m


def orientations(m: np.ndarray):
    for flip in (False, True):
        a = m[:, ::-1] if flip else m
        for r in range(4):
            yield (r, flip), np.rot90(a, r)


def main() -> None:
    font = ImageFont.truetype(str(HERE / "fonts" / "Neuron-Bare.otf"), 400)
    letters = {}
    for ch in GL.GLYPHS:
        im = Image.new("L", (700, 700), 0)
        ImageDraw.Draw(im).text((150, 560), ch, font=font, fill=255, anchor="ls")
        letters[ch] = np.asarray(im) > 127
    L = {ch: norm(m) for ch, m in letters.items()}
    refs = [f"{d.name}/{int(f.stem)}" for d in sorted(PC.ROOT.iterdir()) if d.is_dir()
            for f in sorted(d.glob("[0-9][0-9].svg"))]
    print(f"  {len(refs)} pieces × 8 orientations × {len(L)} letters", flush=True)
    cands = []                                             # (ref, orientation, normalized mask, raw mask)
    for i, ref in enumerate(refs):
        try:
            m = piece_mask(ref)
        except Exception:
            continue
        for o, mm in orientations(m):
            cands.append((ref, o, norm(mm), mm))
        if i % 50 == 0:
            print(f"  {i}/{len(refs)} pieces", flush=True)
    A = np.stack([c[2].ravel() for c in cands])          # (n, N*N)
    best = {}
    for ch, v in L.items():
        sc = A @ v.ravel()
        order = np.argsort(-sc)
        seen, picks = set(), []
        for j in order:
            if cands[j][0] in seen:
                continue
            seen.add(cands[j][0])
            picks.append((float(sc[j]), j))
            if len(picks) == TOP:
                break
        best[ch] = picks
    # sheet: each letter, then its best pieces
    cell = 130
    lab = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    sheet = Image.new("L", (cell * (TOP + 1), cell * len(best)), 255)
    d = ImageDraw.Draw(sheet)
    for r, (ch, picks) in enumerate(best.items()):
        g = Image.fromarray(~letters[ch]).convert("L")
        g = g.crop(g.point(lambda p: 255 - p).getbbox())
        g.thumbnail((cell - 20, cell - 20))
        sheet.paste(g, (10, r * cell + 10))
        for c, (score, j) in enumerate(picks):
            ref, o, _, mm = cands[j]
            im = Image.fromarray(~mm).convert("L")
            im.thumbnail((cell - 24, cell - 24))
            x, y = (c + 1) * cell + 8, r * cell + 4
            sheet.paste(im, (x, y))
            d.text((x, r * cell + cell - 16), f"{ref} {score:.2f}", font=lab, fill=120)
    out = HERE / "proofs" / "found-letters.png"
    sheet.save(out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
