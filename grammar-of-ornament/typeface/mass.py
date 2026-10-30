"""
Neuron Mass — letters made of a thin frame and one compound piece as their body.

After the reference Sarah showed (letters drawn as a hairline frame plus one
large organic black mass, whose curved edge is part of the letter's outline):

  frame   the Bare skeleton drawn thin (FRAME × the weight), so the letter is
          always readable
  body    the letter's body region: its ink, closed with a large disc and
          filled, i.e. the area the letter occupies including its counters
  mass    one real compound piece (silhouettes.py), in one of its eight
          orientations, at the scale and position where it best fills the
          body: mostly inside it (INSIDE), covering about COVER of it, and
          leaving at least part of each counter open. The piece is cut by the
          body region, so its outer edge is flush with the letter and its own
          edge (teeth, lobes, branches) faces into the letter
  letter  frame ∪ mass

  .venv/bin/python typeface/mass.py "ABDEFGHKMNRS abdeghnopqs"   → typeface/proofs/mass/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dress as DR          # noqa: E402
import found as FO          # noqa: E402
import glyphs as GL         # noqa: E402
import outline as OL        # noqa: E402
import pieces as PC         # noqa: E402

KINDS = ["fork", "straight-sprig", "curved-sprig", "wavy-sprig", "straight-comb", "curved-comb", "wavy-comb",
         "straight-feather", "curved-feather", "wavy-feather", "straight-branching-line", "curved-branching-line",
         "wavy-branching-line", "loop", "curved-band", "wavy-band", "lobed-wavy-band", "hatched-blob"]
SEARCH = 0.12      # pixels per font unit while searching
FINAL = 1.0        # pixels per font unit for the result
FRAME = 0.22       # frame weight, relative to the letter's weight
COVER = 0.42       # share of the body the mass should fill
COUNTER = 0.15     # most of a counter the mass may fill before it is penalized
SCALES = (0.6, 0.8, 1.0, 1.2)


def fill(contours, s, lo, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    for C in contours:
        im = Image.new("1", (shape[1], shape[0]), 0)
        ImageDraw.Draw(im).polygon([((x - lo[0]) * s, shape[0] - (y - lo[1]) * s) for x, y in C], fill=1)
        m ^= np.asarray(im, bool)
    return m


def letter(ch: str, W: float):
    st = GL.GLYPHS[ch]
    DR.dress(ch, st, W, None, None)
    ink = OL.outline(st, W)
    DR.dress(ch, st, W, None, None)
    frame = OL.outline(st, W * FRAME)
    return ink, frame


def body_mask(ink_mask: np.ndarray, s: float, W: float) -> np.ndarray:
    r = 1.6 * W * s
    dil = ndi.distance_transform_edt(~ink_mask) <= r
    closed = ndi.distance_transform_edt(dil) > r
    return ndi.binary_fill_holes(closed | ink_mask)


_CANDS = None


def candidates():
    global _CANDS
    if _CANDS is None:
        _CANDS = []
        for kind in KINDS:
            for f in sorted((PC.ROOT / kind).glob("[0-9][0-9].svg")):
                ref = f"{kind}/{int(f.stem)}"
                try:
                    m = FO.piece_mask(ref, px=160)
                except Exception:
                    continue
                for o, mm in FO.orientations(m):
                    _CANDS.append((ref, o, mm))
        print(f"  {len(_CANDS)} candidate piece orientations", flush=True)
    return _CANDS


def _resize(m: np.ndarray, h: int) -> np.ndarray:
    w = max(1, int(round(m.shape[1] * h / m.shape[0])))
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) > 127


USED: dict = {}     # pieces already given to a letter (so letters get different pieces)
REUSE = 0.25


def best_mass(H: np.ndarray, Cn: np.ndarray, I: np.ndarray):
    """The piece orientation, height (px) and offset whose mass best becomes the letter's heavy strokes:
    mostly inside the body H, over as much of the letter's own ink I as it can, filling about COVER of
    the body, and leaving the counters Cn mostly open."""
    best = (-9, None)
    aH, aC, aI = H.sum(), max(Cn.sum(), 1), max(I.sum(), 1)
    hh = int(np.ptp(np.nonzero(H.any(1))[0]) + 1)
    for ref, o, m in candidates():
        for sc in SCALES:
            M = _resize(m, max(4, int(hh * sc)))
            aM = M.sum()
            if aM < 0.15 * aH or M.shape[0] > H.shape[0] or M.shape[1] > H.shape[1]:
                continue
            k = M[::-1, ::-1].astype(float)
            inside = fftconvolve(H.astype(float), k, mode="valid")
            cc = fftconvolve(Cn.astype(float), k, mode="valid")
            on_ink = fftconvolve(I.astype(float), k, mode="valid")
            score = (inside / aM + 1.5 * on_ink / aI - 1.0 * np.abs(inside / aH - COVER)
                     - 3.0 * np.maximum(0, cc / aC - COUNTER) - REUSE * USED.get(ref, 0))
            j = np.unravel_index(np.argmax(score), score.shape)
            if score[j] > best[0]:
                best = (float(score[j]), (ref, o, sc, j, M.shape))
    return best


def build(ch: str):
    W = GL.W_LOW if ch.islower() else GL.W_CAP
    ink, frame = letter(ch, W)
    P = np.vstack(ink)
    lo = P.min(0) - 2 * W
    hi = P.max(0) + 2 * W

    def raster(s):
        shape = tuple((np.ceil((hi - lo) * s)).astype(int)[::-1])
        I = fill(ink, s, lo, shape)
        F = fill(frame, s, lo, shape)
        H = body_mask(I, s, W)
        Cn = H & ~I
        return I, F, H, Cn

    I, F, H, Cn = raster(SEARCH)
    score, (ref, o, sc, (y, x), mshape) = best_mass(H, Cn, I)
    USED[ref] = USED.get(ref, 0) + 1
    # redo at full resolution
    I2, F2, H2, _ = raster(FINAL)
    k = FINAL / SEARCH
    hh = int(np.ptp(np.nonzero(H2.any(1))[0]) + 1)
    r, flip = o
    m = FO.piece_mask(ref, px=int(max(mshape) * k * 1.05))
    m = m[:, ::-1] if flip else m
    m = np.rot90(m, r)
    M = _resize(m, max(4, int(hh * sc)))
    Y, X = int(round(y * k)), int(round(x * k))
    mass = np.zeros_like(H2)
    ys, xs = slice(Y, min(Y + M.shape[0], H2.shape[0])), slice(X, min(X + M.shape[1], H2.shape[1]))
    mass[ys, xs] = M[:ys.stop - Y, :xs.stop - X]
    out = F2 | (mass & H2)
    return out, I2, ref, o, score


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "ABDEFGHKMNRS abdeghnopqs"
    outdir = HERE / "proofs" / "mass"
    outdir.mkdir(parents=True, exist_ok=True)
    cells = []
    for ch in text.replace(" ", ""):
        g, bare, ref, o, sc = build(ch)
        print(f"  {ch}: {ref} {o} score {sc:.2f}", flush=True)
        cells.append((ch, g, bare, ref))
    cw = 300
    sheet = Image.new("L", (cw * 2 * 6, 380 * ((len(cells) + 5) // 6)), 255)
    d = ImageDraw.Draw(sheet)
    for i, (ch, g, bare, ref) in enumerate(cells):
        x, y = (i % 6) * 2 * cw, (i // 6) * 380
        for j, m in enumerate((bare, g)):
            im = Image.fromarray(~m).convert("L")
            im = im.crop(Image.fromarray(m.astype(np.uint8) * 255).getbbox())
            im.thumbnail((cw - 30, 330))
            sheet.paste(im, (x + j * cw + 10, y + 10))
        d.text((x + 10, y + 350), f"{ch}: {ref}", fill=120)
    sheet.save(outdir / "concept.png")
    print(f"wrote {outdir / 'concept.png'}")
