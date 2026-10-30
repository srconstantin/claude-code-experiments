"""
Neuron Found — letters that are real compound pieces, nudged into legibility.

For each letter, the compound pieces of the branched families (BRANCHED)
that already resemble it are found (found.py: every piece in eight
orientations, scored by correlation with the Bare glyph). Each candidate is
then registered to the letter: a smooth deformation field (TV-L1 optical
flow between the blurred letter and the blurred piece) says where each part
of the piece would have to move to become the letter, and the piece is moved
only part of the way (ALPHA). It keeps its own outline, branches and lumps,
but its overall shape is pulled toward the letter's.

  .venv/bin/python typeface/warp.py [TEXT]   → typeface/proofs/found/warped.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import found as FO          # noqa: E402
import pieces as PC         # noqa: E402

BRANCHED = ["fork", "straight-sprig", "curved-sprig", "wavy-sprig", "straight-comb", "curved-comb", "wavy-comb",
            "straight-feather", "curved-feather", "wavy-feather", "straight-branching-line",
            "curved-branching-line", "wavy-branching-line"]
N = 200            # working box, pixels
ALPHAS = (0.6, 0.85)
TOP = 3


def boxed(mask: np.ndarray, n: int = N, pad: int = 12) -> np.ndarray:
    """A mask cropped to its ink and scaled (proportions kept) into an n×n box, centred."""
    ys, xs = np.nonzero(mask)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    s = (n - 2 * pad) / max(h, w)
    im = Image.fromarray(m.astype(np.uint8) * 255).resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    a = np.asarray(im) > 127
    out = np.zeros((n, n), bool)
    y0, x0 = (n - a.shape[0]) // 2, (n - a.shape[1]) // 2
    out[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    return out


def register(piece: np.ndarray, letter: np.ndarray, alpha: float) -> np.ndarray:
    """The piece, moved `alpha` of the way toward the letter along a smooth deformation."""
    a = ndi.gaussian_filter(letter.astype(float), 3)
    b = ndi.gaussian_filter(piece.astype(float), 3)
    v, u = optical_flow_tvl1(a, b, attachment=8, tightness=0.4, num_warp=8, num_iter=30)
    v = ndi.gaussian_filter(v, 4) * alpha
    u = ndi.gaussian_filter(u, 4) * alpha
    rr, cc = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    moved = warp(piece.astype(float), np.array([rr + v, cc + u]), mode="constant", order=1)
    moved = ndi.gaussian_filter(moved, 1.5) > 0.5
    lab, n = ndi.label(moved)
    if n > 1:                                                    # keep it one piece
        sz = ndi.sum(moved, lab, range(1, n + 1))
        moved = np.isin(lab, 1 + np.flatnonzero(sz > 0.04 * sz.max()))
    return moved


def main(text: str) -> None:
    font = ImageFont.truetype(str(HERE / "fonts" / "Neuron-Bare.otf"), 400)
    refs = [f"{k}/{int(f.stem)}" for k in BRANCHED for f in sorted((PC.ROOT / k).glob("[0-9][0-9].svg"))]
    cands = []
    for ref in refs:
        try:
            m = FO.piece_mask(ref, px=240)
        except Exception:
            continue
        for o, mm in FO.orientations(m):
            cands.append((ref, o, mm, FO.norm(mm)))
    A = np.stack([c[3].ravel() for c in cands])
    print(f"  {len(cands)} candidates", flush=True)
    rows = []
    for ch in text:
        im = Image.new("L", (700, 700), 0)
        ImageDraw.Draw(im).text((150, 560), ch, font=font, fill=255, anchor="ls")
        L = np.asarray(im) > 127
        Lb = boxed(L)
        sc = A @ FO.norm(L).ravel()
        picks, seen = [], set()
        for j in np.argsort(-sc):
            if cands[j][0] in seen:
                continue
            seen.add(cands[j][0])
            picks.append(j)
            if len(picks) == TOP:
                break
        row = [Lb]
        labels = [ch]
        for j in picks:
            ref, o, mm, _ = cands[j]
            Pb = boxed(mm)
            row.append(Pb)
            labels.append(ref)
            for al in ALPHAS:
                row.append(register(Pb, Lb, al))
                labels.append(f"{al:.2f}")
        rows.append((row, labels))
        print(f"  {ch}: {[cands[j][0] for j in picks]}", flush=True)
    cell = N + 10
    ncol = 1 + TOP * (1 + len(ALPHAS))
    sheet = Image.new("L", (cell * ncol, (cell + 16) * len(rows)), 255)
    d = ImageDraw.Draw(sheet)
    lab = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    for r, (row, labels) in enumerate(rows):
        for c, (m, t) in enumerate(zip(row, labels)):
            sheet.paste(Image.fromarray(~m).convert("L"), (c * cell, r * (cell + 16)))
            d.text((c * cell + 4, r * (cell + 16) + N + 2), t, font=lab, fill=110)
    out = HERE / "proofs" / "found" / "warped.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AEFGHKMRSY abefghknrsy")
