#!/usr/bin/env python3
"""
Grammar of Ornament — recrop: clean silhouettes for the hand-cut motifs.

A hand-drawn lasso is only a hint. Where it cuts straight across a form the
motif gets a ragged, ruled edge. Here every saved motif in vgg16motifs/ is cut
again from the same print (same settings), so that its edge follows the dark
marks of the print instead of the lasso:

  1. ink = the black of the whole, uncropped print
  2. a band REACH wide on each side of the lasso is left undecided; inside it
     the cut is settled by a marker watershed, flooding from the inside and
     the outside of the lasso over the (blurred) ink, so the two meet on the
     darkest ridge, i.e. along a dark line of the print near the hand line
  3. the body = the inside region plus the dark line it stopped at, holes
     filled, smoothed, crumbs dropped, and always one piece: anything
     the cut split apart is bridged back together (see one_piece)
  4. the motif = the print inside the body, ground outside, plus a dark rim
     along the body's edge, as wide as the print's typical black stroke, so
     the silhouette always ends in a dark outline.

The originals are never touched. Output, in vgg16motifs-clean/<feature>/<id>/:
motif.png (1 bit, tight box), motif.svg, mask.png (the body), motif.json; and
vgg16motifs-clean/compare.png, original beside clean.

Run:  python3 recrop.py              # every motif
      python3 recrop.py --only c117  # motifs whose feature id contains c117
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.draw import polygon2mask
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

import motifs as M

HERE = Path(__file__).resolve().parent
SRC = HERE / "vgg16features"
OUT_DIR = HERE / "vgg16motifs-clean"
PX = M.OUT          # 1792: the print size
REACH = 0.025       # the undecided band on each side of the lasso, fraction of the print
CLOSE = 0.012       # closing radius, fraction of the print
MARGIN = 0.03       # ground around the tight box, fraction of the motif's long side


def grow(mask: np.ndarray, r: float) -> np.ndarray:
    return ndi.distance_transform_edt(~mask) <= r


def shrink(mask: np.ndarray, r: float) -> np.ndarray:
    return ndi.distance_transform_edt(mask) > r


def stroke_width(ink: np.ndarray) -> float:
    """The typical width of the black marks: twice the median distance to the ground along their skeleton."""
    sk = skeletonize(ink)
    dt = ndi.distance_transform_edt(ink)
    v = dt[sk]
    return float(2 * np.median(v)) if v.size else 4.0


def body(ink: np.ndarray, lasso: np.ndarray, w: float) -> np.ndarray:
    n = ink.shape[0]
    R = REACH * n
    inner, outer = shrink(lasso, R), ~grow(lasso, R)
    markers = inner.astype(np.int32) + 2 * outer.astype(np.int32)
    ridge = ndi.gaussian_filter(ink.astype(np.float32), max(1.0, w / 2))   # dark lines are ridges
    region = watershed(ridge, markers) == 1         # the cut settles on the darkest ridge in the band
    b = region | (ink & grow(region, 1.5 * w) & ~outer)   # take the dark line itself along with it
    r = CLOSE * n
    b = ndi.binary_fill_holes(b)
    b = ndi.gaussian_filter(b.astype(np.float32), r * 0.35) > 0.5    # smooth the edge
    b = grow(shrink(b, r * 0.5), r * 0.5)  # opening: drop hairline spurs
    return ndi.binary_fill_holes(one_piece(b, r))


def one_piece(b: np.ndarray, r: float) -> np.ndarray:
    """A motif is always one connected shape. Crumbs (< 4% of the largest
    piece) are dropped; the rest are bridged by closings of growing radius
    until they join; failing that, only the largest piece is kept."""
    lab, k = ndi.label(b)
    if k <= 1:
        return b
    sizes = ndi.sum(b, lab, index=np.arange(1, k + 1))
    b = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.04 * sizes.max()))
    for f in (1, 2, 3, 4, 6, 8):
        if ndi.label(b)[1] <= 1:
            return b
        b = shrink(grow(b, f * r), f * r) | b
    lab, k = ndi.label(b)
    if k > 1:
        b = lab == 1 + int(np.argmax(ndi.sum(b, lab, index=np.arange(1, k + 1))))
    return b


def print_ink(m: dict) -> np.ndarray:
    """The whole print's black, at PX, cached in cache/prints/ (it takes ~10 s to make)."""
    f = HERE / "cache" / "prints" / f"{m['feature']}__{hashlib.md5(json.dumps(m['params'], sort_keys=True).encode()).hexdigest()[:8]}.png"
    if f.exists():
        return np.asarray(Image.open(f)) > 0
    ink = ~M.render(SRC, m["feature"], m["params"], None, PX)
    f.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ink).convert("1").save(f)
    return ink


def cut(ink: np.ndarray, lasso: np.ndarray, w: float) -> dict:
    """The motif inside `lasso`, its edge snapped to the print's dark lines, with a dark rim, in a tight box."""
    b = body(ink, lasso, w)
    rim = b & ~shrink(b, w * 0.45)
    out_ink = (ink & b) | rim
    ys, xs = np.nonzero(b)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = int(MARGIN * max(y1 - y0, x1 - x0))
    Y0, X0 = max(0, y0 - pad), max(0, x0 - pad)
    Y1, X1 = min(PX, y1 + pad), min(PX, x1 + pad)
    return {"white": ~out_ink[Y0:Y1, X0:X1], "mask": b[Y0:Y1, X0:X1], "lasso": lasso[Y0:Y1, X0:X1],
            "box": [int(X0), int(Y0), int(X1), int(Y1)], "rim": round(w * 0.45, 1)}


def centred(m: dict):
    """The print and the lasso, both rolled so the lasso sits in the middle: the
    print tiles, so a motif the hand line ran off the edge comes out whole."""
    ink = print_ink(m)
    k = PX / m.get("work_px", M.WORK)
    lasso = polygon2mask(ink.shape, [(y * k, x * k) for x, y in m["crop"]])
    ys, xs = np.nonzero(lasso)
    dy, dx = PX // 2 - int((ys.min() + ys.max()) / 2), PX // 2 - int((xs.min() + xs.max()) / 2)
    return np.roll(ink, (dy, dx), (0, 1)), np.roll(lasso, (dy, dx), (0, 1))


def clean(m: dict) -> dict:
    ink, lasso = centred(m)
    w = max(2.0, stroke_width(ink & lasso))
    return cut(ink, lasso, w)


SIB_SCALE = 4       # matching runs at 1/4 size
SIB_MIN = 0.2       # normalized correlation a sibling must reach (of the blurred ink)
SIB_MAX = 6


def siblings(m: dict) -> list[dict]:
    """Other occurrences of the motif in its print (the print repeats its forms
    irregularly), found by normalized cross-correlation of the print's ink
    against the ink inside the lasso, wrapping round the edges; each is cut
    with the lasso moved onto it and snapped to its own dark lines."""
    ink, lasso = centred(m)
    w = max(2.0, stroke_width(ink & lasso))
    s = SIB_SCALE
    P = ndi.gaussian_filter(ink.astype(np.float32), 2 * s)[::s, ::s]
    L = lasso[::s, ::s].astype(np.float32)
    n = L.sum()
    T = P * L
    T = (T - L * T.sum() / n)
    T /= np.sqrt((T * T).sum()) + 1e-9
    F = np.fft.rfft2
    I = lambda a: np.fft.irfft2(a, s=P.shape)
    num = I(F(P) * np.conj(F(T)))                       # Σ P(x+u)·T(u)
    ps, ps2 = I(F(P) * np.conj(F(L))), I(F(P * P) * np.conj(F(L)))
    var = np.maximum(ps2 - ps * ps / n, 1e-9)
    ncc = num / np.sqrt(var)
    ys, xs = np.nonzero(L)
    size = max(np.ptp(ys), np.ptp(xs))
    taken = [(0, 0)]                                    # the original, at zero shift
    out = []
    order = np.argsort(ncc, axis=None)[::-1]
    H = P.shape[0]
    for idx in order[:20000]:
        v = float(ncc.flat[idx])
        if v < SIB_MIN or len(out) >= SIB_MAX:
            break
        dy, dx = np.unravel_index(idx, ncc.shape)
        dy, dx = (dy + H // 2) % H - H // 2, (dx + H // 2) % H - H // 2
        if any(np.hypot(dy - a, dx - b) < 0.7 * size for a, b in taken):
            continue
        taken.append((dy, dx))
        c = cut(np.roll(ink, (-dy * s, -dx * s), (0, 1)), lasso, w)
        c["score"], c["shift"] = round(v, 3), [int(dx * s), int(dy * s)]
        out.append(c)
    return out


def sheet(pairs, path: Path, T: int = 220, cols: int = 6) -> None:
    rows = (len(pairs) + cols - 1) // cols
    per = max(len(p) for p in pairs)
    img = Image.new("L", (cols * per * T, rows * T), 150)
    d = ImageDraw.Draw(img)
    for i, row in enumerate(pairs):
        x, y = (i % cols) * per * T, (i // cols) * T
        for j, f in enumerate(row):
            im = Image.open(f).convert("L")
            im.thumbnail((T - 8, T - 8))
            img.paste(im, (x + j * T + 4, y + 4))
        d.line([(x + per * T - 1, y), (x + per * T - 1, y + T)], fill=60, width=3)
    img.save(path)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="")
    ap.add_argument("--siblings", action="store_true", help="also cut the motif's other occurrences in its print")
    a = ap.parse_args()
    items = [json.loads(f.read_text()) for f in sorted((HERE / "vgg16motifs").glob("*/*/motif.json"))]
    items = [m for m in items if m.get("crop") and a.only in m["feature"]]
    print(f"recropping {len(items)} motifs → {OUT_DIR.name}/")
    pairs, fams = [], []
    for i, m in enumerate(items):
        t0 = time.time()
        c = clean(m)
        out = OUT_DIR / m["id"]
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(c["white"]).convert("1").save(out / "motif.png", optimize=True)
        Image.fromarray(c["mask"]).convert("1").save(out / "mask.png", optimize=True)
        (out / "motif.svg").write_text(M.svg(c["white"]))
        h, w = c["white"].shape
        rec = {"id": m["id"], "feature": m["feature"], "source": f"vgg16motifs/{m['id']}", "params": m["params"],
               "box": c["box"], "print_px": PX, "rim": c["rim"], "width": w, "height": h,
               "lasso_overlap": round(float((c["mask"] & c["lasso"]).sum() / max(1, c["mask"].sum())), 3)}
        msg = f"  [{i + 1}/{len(items)}] {m['id'][:60]}  {w}×{h}  rim {c['rim']}px"
        if a.siblings:
            sd = out / "siblings"
            sd.mkdir(exist_ok=True)
            for old in sd.glob("s*.png"):       # only this script's own files
                old.unlink()
            sibs = siblings(m)
            rec["siblings"] = []
            for j, sc in enumerate(sibs):
                Image.fromarray(sc["white"]).convert("1").save(sd / f"s{j + 1}.png", optimize=True)
                Image.fromarray(sc["mask"]).convert("1").save(sd / f"s{j + 1}.mask.png", optimize=True)
                rec["siblings"].append({"file": f"siblings/s{j + 1}.png", "score": sc["score"], "shift": sc["shift"],
                                        "width": int(sc["white"].shape[1]), "height": int(sc["white"].shape[0])})
            fams.append([out / "motif.png"] + [sd / f"s{j + 1}.png" for j in range(len(sibs))])
            msg += f"  {len(sibs)} siblings " + " ".join(str(x["score"]) for x in sibs)
        (out / "motif.json").write_text(json.dumps(rec, indent=1))
        pairs.append((HERE / "vgg16motifs" / m["id"] / "motif.png", out / "motif.png"))
        print(msg + f"  {time.time() - t0:.1f}s", flush=True)
    sheet(pairs, OUT_DIR / "compare.png")
    if fams:
        sheet(fams, OUT_DIR / "siblings.png", T=160, cols=2)


if __name__ == "__main__":
    main()
