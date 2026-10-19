#!/usr/bin/env python3
"""
Grammar of Ornament — patterns: tessellation and multiscale structure in the
features and their linework prints.

Tessellation (each feature image, as its luminance, high-passed so the slow
vignette does not count):
  lattice      the circular autocorrelation's strongest off-centre peak v1 and
               the strongest peak not parallel to it v2; their angle and length
               ratio classify the repeat: square, hexagonal, rectangular,
               centered (equal lengths, other angles), oblique, or stripes (v2
               much weaker than v1). Strength is the autocorrelation at v1
               (1 = a perfect repeat). The renders are exactly periodic over the
               224px frame, so circular correlation is the right tool.
  symmetry     the power spectrum's angular energy in the 4–80 cycles/frame
               annulus, as Fourier coefficients of order 2 (stripes, one
               direction), 4 (square) and 6 (hexagonal); a real image's
               spectrum is point-symmetric, so odd orders vanish.
  stride lock  the repeat period |v1| against the layer's stride in input
               pixels: how often the period is within 8% of a whole multiple
               of the stride, against the rate expected by chance for periods
               spread over the same range.

Multiscale structure:
  spectrum     the slope β of the radially averaged power spectrum, P ∝ f^-β,
               over 4–90 cycles/frame, beside the renderer's own prior (its
               Fourier parameterization scales amplitudes by f^-1.25, β = 2.5,
               at the start of optimization); and how peaked the spectrum is
               after the power law is removed: the share of residual energy
               within half an octave of its maximum (near 1 = one
               characteristic scale; near the uniform share = scale-free).
  box count    on the linework prints (ink = masses, strokes and dots), the
               box-counting dimension of the ink's edges over boxes of 2–128px
               at 896px, and how straight the log–log line is (R²) — a straight
               line over many octaves is the signature of self-similarity.
  mark sizes   from the per-stroke tables: the sizes of all marks (stroke
               widths and lengths, dot and mass sizes); the spread of their
               logs, and the slope of the upper tail of the size distribution
               (a straight power-law tail suggests self-similarity; a bump at
               one size, a characteristic scale).
  nesting      how often a mass has relief marks cut into it (marks inside
               marks: the depth of the hierarchy).

Written into patterns/: features.csv (one row per feature), prints.csv (one
row per linework print), examples.png (the clearest lattices with their
vectors drawn), and a summary printed at the end.

Run:  python3 patterns.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
OUT = HERE / "patterns"
FOLDERS = ["vgg16features", "resnet50features", "convnext_basefeatures"]


# ─────────────────────────────────────────────────────────────────────────────
#  Layer depth and stride
# ─────────────────────────────────────────────────────────────────────────────

def stride_of(model: str, layer: str) -> int:
    """The layer's stride in input pixels (total downsampling before it)."""
    if model == "vgg16":                     # max-pools after features.4, 9, 16, 23
        i = int(layer.split(".")[1])
        return 1 if i < 5 else 2 if i < 10 else 4 if i < 17 else 8 if i < 24 else 16
    if model == "resnet50":                  # stem /4; layer2 /8, layer3 /16, layer4 /32
        return {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}[layer.split(".")[0]]
    if model == "convnext_base":             # stem /4, then /2 before stages 3, 5, 7
        st = int(layer.split(".")[1])
        return 4 if st < 3 else 8 if st < 5 else 16 if st < 7 else 32
    return 1


def stage_of(model: str, layer: str) -> str:
    s = stride_of(model, layer)
    return f"stride {s}"


# ─────────────────────────────────────────────────────────────────────────────
#  Tessellation
# ─────────────────────────────────────────────────────────────────────────────

def luminance(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255
    L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    L = L - ndi.gaussian_filter(L, 20, mode="wrap")       # drop the vignette and slow shading
    return (L - L.mean()) / (L.std() + 1e-9)


def autocorr(x: np.ndarray) -> np.ndarray:
    F = np.fft.fft2(x)
    a = np.real(np.fft.ifft2(np.abs(F) ** 2))
    return np.fft.fftshift(a / a.flat[0])


def lattice(a: np.ndarray, rmin: float = 3.0, rmax: float = 80.0) -> dict:
    n = a.shape[0]
    c = n // 2
    mx = ndi.maximum_filter(a, size=5, mode="wrap")
    yy, xx = np.mgrid[0:n, 0:n] - c
    r = np.hypot(yy, xx)
    cand = (a == mx) & (r >= rmin) & (r <= rmax) & (xx >= 0) & ~((xx == 0) & (yy < 0))   # one of each ± pair
    ys, xs = np.nonzero(cand)
    if len(ys) == 0:
        return {"strength": 0.0}
    h = a[ys, xs]
    o = np.argsort(-h)
    vecs = np.c_[xs - c, ys - c][o].astype(float)
    hs = h[o]
    v1, h1 = vecs[0], hs[0]
    # the peak's radius may be a harmonic: prefer the shortest peak along v1's direction with most of its height
    for v, hv in zip(vecs, hs):
        if hv > 0.8 * h1 and np.linalg.norm(v) < np.linalg.norm(v1) - 1:
            u, u1 = v / np.linalg.norm(v), v1 / np.linalg.norm(v1)
            if abs(u[0] * u1[1] - u[1] * u1[0]) < 0.15:
                v1, h1 = v, hv
    v2, h2 = None, 0.0
    for v, hv in zip(vecs, hs):
        ang = np.degrees(np.arccos(np.clip(abs(v @ v1) / (np.linalg.norm(v) * np.linalg.norm(v1)), 0, 1)))
        if ang > 25:
            v2, h2 = v, hv
            break
    out = {"strength": float(h1), "period": float(np.linalg.norm(v1)), "angle1": float(np.degrees(np.arctan2(-v1[1], v1[0])) % 180),
           "v1": v1.tolist(), "strength2": float(h2)}
    if v2 is None or h2 < 0.5 * h1:
        out["type"] = "stripes"
        out["v2"] = v2.tolist() if v2 is not None else None
        return out
    # reduce to the shortest basis and classify
    a1, a2 = v1, v2
    for _ in range(10):
        if np.linalg.norm(a2) < np.linalg.norm(a1):
            a1, a2 = a2, a1
        m = round(float(a1 @ a2) / float(a1 @ a1))
        if m == 0:
            break
        a2 = a2 - m * a1
    th = np.degrees(np.arccos(np.clip(abs(a1 @ a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)), 0, 1)))
    ratio = np.linalg.norm(a2) / max(np.linalg.norm(a1), 1e-9)
    if abs(th - 90) < 9:
        typ = "square" if ratio < 1.15 else "rectangular"
    elif abs(th - 60) < 9 and ratio < 1.15:
        typ = "hexagonal"
    elif ratio < 1.15:
        typ = "centered"
    else:
        typ = "oblique"
    out.update({"type": typ, "v2": v2.tolist(), "basis_angle": float(th), "basis_ratio": float(ratio), "period2": float(np.linalg.norm(a2))})
    return out


def spectrum_measures(x: np.ndarray) -> dict:
    n = x.shape[0]
    P = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    yy, xx = np.mgrid[0:n, 0:n] - n // 2
    r = np.hypot(yy, xx)
    phi = np.arctan2(yy, xx)
    ann = (r >= 4) & (r <= 80)
    w = P[ann]
    sym = {f"sym{m}": float(abs(np.sum(w * np.exp(1j * m * phi[ann]))) / (np.sum(w) + 1e-12)) for m in (2, 4, 6)}
    # radial profile and slope
    rb = np.round(r).astype(int)
    prof = np.bincount(rb.ravel(), P.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
    f = np.arange(len(prof))
    sel = (f >= 4) & (f <= 90)
    lf, lp = np.log(f[sel]), np.log(prof[sel] + 1e-12)
    beta, b0 = np.polyfit(lf, lp, 1)
    resid = np.exp(lp - (beta * lf + b0))          # power relative to the fitted law
    resid = np.clip(resid - np.median(resid), 0, None)
    k = np.argmax(resid)
    near = np.abs(lf - lf[k]) <= np.log(2) / 2
    peak_share = float(resid[near].sum() / (resid.sum() + 1e-12))
    uniform_share = float(near.mean())
    return {"beta": float(-beta), "peak_share": peak_share, "uniform_share": uniform_share,
            "peak_scale_px": float(n / f[sel][k]), **sym}


# ─────────────────────────────────────────────────────────────────────────────
#  Prints
# ─────────────────────────────────────────────────────────────────────────────

def ink_of_classes(path: Path) -> np.ndarray:
    """The print's black, from the linework classes image: masses (grey),
    strokes (black) and dots (blue) are ink; white, relief strokes (orange)
    and relief dots (green) are not."""
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(int)
    white = (rgb[..., 0] > 240) & (rgb[..., 1] > 240) & (rgb[..., 2] > 240)
    orange = (abs(rgb[..., 0] - 235) < 25) & (abs(rgb[..., 1] - 120) < 25) & (abs(rgb[..., 2] - 30) < 25)
    green = (abs(rgb[..., 0] - 90) < 25) & (abs(rgb[..., 1] - 190) < 25) & (abs(rgb[..., 2] - 70) < 25)
    return ~(white | orange | green)


def box_dimension(edge: np.ndarray, sizes=(2, 4, 8, 16, 32, 64, 128)) -> tuple[float, float]:
    counts = []
    n = edge.shape[0]
    for s in sizes:
        m = n // s
        e = edge[:m * s, :m * s].reshape(m, s, m, s).any(axis=(1, 3))
        counts.append(e.sum())
    x, y = np.log(1 / np.array(sizes, float)), np.log(np.array(counts, float) + 1e-9)
    D, b = np.polyfit(x, y, 1)
    pred = D * x + b
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(D), float(r2)


def mark_sizes(strokes_json: Path) -> dict:
    d = json.loads(strokes_json.read_text())
    sizes = []
    for s in d["strokes"]:
        if s["pol"] in ("ink", "relief"):
            sizes += [s["width"], s["length"]]
    for m in d["masses"]:
        sizes.append(2 * np.sqrt(m["area"] / np.pi))
    sizes = np.array([v for v in sizes if v > 0.3])
    ls = np.log(sizes)
    # upper tail: CCDF slope over the top decade of sizes
    srt = np.sort(sizes)
    ccdf = 1 - np.arange(len(srt)) / len(srt)
    top = srt >= np.quantile(srt, 0.5)
    slope = np.polyfit(np.log(srt[top]), np.log(ccdf[top]), 1)[0] if top.sum() > 10 else np.nan
    nested = sum(1 for s in d["strokes"] if s["pol"] == "relief" and s.get("mass", -1) >= 0)
    masses_with = len({s["mass"] for s in d["strokes"] if s["pol"] == "relief" and s.get("mass", -1) >= 0})
    return {"n_marks": int(len(sizes)), "log_size_sd": float(ls.std()), "tail_slope": float(-slope),
            "size_median": float(np.median(sizes)), "masses": len(d["masses"]), "masses_with_relief": masses_with,
            "relief_in_masses": nested}


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    rows = []
    for folder in FOLDERS:
        files = sorted((HERE / folder).glob("*.png"))
        for i, p in enumerate(files):
            parts = p.stem.split("__")
            model, layer = parts[0], parts[1]
            x = luminance(p)
            lat = lattice(autocorr(x))
            sp = spectrum_measures(x)
            stride = stride_of(model, layer)
            row = {"name": p.stem, "folder": folder, "model": model, "layer": layer, "stride": stride, **sp,
                   **{k: v for k, v in lat.items() if k not in ("v1", "v2")}, "v1": json.dumps(lat.get("v1")), "v2": json.dumps(lat.get("v2"))}
            if "period" in lat and stride > 1:
                q = lat["period"] / stride
                row["period_over_stride"] = q
                row["stride_locked"] = bool(q >= 0.92 and abs(q - round(q)) / max(round(q), 1) <= 0.08)
            rows.append(row)
        print(f"  {folder}: {len(files)} features ({time.time() - t0:.0f}s)", flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(OUT / "features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # prints
    prows = []
    lw = HERE / "linework" / "vgg16features"
    for p in sorted(lw.glob("*.classes.png")):
        name = p.name[: -len(".classes.png")]
        ink = ink_of_classes(p)
        edge = ink ^ ndi.binary_erosion(ink)
        D_edge, r2_edge = box_dimension(edge)
        D_ink, r2_ink = box_dimension(ink)
        ms = mark_sizes(lw / f"{name}.strokes.json")
        layer = name.split("__")[1]
        prows.append({"name": name, "layer": layer, "stride": stride_of("vgg16", layer), "D_edge": D_edge, "r2_edge": r2_edge,
                      "D_ink": D_ink, "r2_ink": r2_ink, **ms})
    with open(OUT / "prints.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prows[0]))
        w.writeheader()
        w.writerows(prows)
    print(f"  prints: {len(prows)} ({time.time() - t0:.0f}s)", flush=True)

    # ── summary ──
    def frac(xs):
        return f"{100 * np.mean(xs):.0f}%" if len(xs) else "–"

    print("\n  TESSELLATION — lattice types (autocorrelation strength ≥ 0.3 counts as clearly periodic)")
    types = ["square", "hexagonal", "rectangular", "centered", "oblique", "stripes"]
    print("  " + "group".ljust(26) + "n".rjust(4) + "  periodic" + "".join(t[:11].rjust(12) for t in types))
    groups = [(f"{m} all", [r for r in rows if r["model"] == m]) for m in ("vgg16", "resnet50", "convnext_base")]
    groups += [(f"stride {s}", [r for r in rows if r["stride"] == s]) for s in sorted({r["stride"] for r in rows})]
    for g, rs in groups:
        per = [r for r in rs if r.get("strength", 0) >= 0.3]
        line = "  " + g.ljust(26) + f"{len(rs):4d}" + f"{frac([r.get('strength', 0) >= 0.3 for r in rs]):>10}"
        line += "".join(f"{(sum(1 for r in per if r.get('type') == t) / max(len(per), 1)) * 100:11.0f}%" for t in types)
        print(line)
    st = np.array([r.get("strength", 0) for r in rows])
    print(f"\n  periodicity strength: median {np.median(st):.2f}; ≥0.2 {frac(st >= 0.2)}, ≥0.3 {frac(st >= 0.3)}, ≥0.5 {frac(st >= 0.5)}")

    print("\n  SYMMETRY of the power spectrum (mean coefficient; 0 = none, 1 = perfect)")
    for g, rs in groups:
        print("  " + g.ljust(26) + "".join(f"  order {m}: {np.mean([r[f'sym{m}'] for r in rs]):.3f}" for m in (2, 4, 6)))

    print("\n  STRIDE LOCK — periodic features whose repeat is a whole multiple of the layer stride (±8%)")
    rng = np.random.default_rng(0)
    for g, rs in groups:
        per = [r for r in rs if r.get("strength", 0) >= 0.3 and "stride_locked" in r]
        if not per:
            continue
        obs = np.mean([r["stride_locked"] for r in per])
        # chance: same periods, strides reassigned at random from the same group
        sims = []
        strides = np.array([r["stride"] for r in per])
        periods = np.array([r["period"] for r in per])
        for _ in range(2000):
            q = periods / rng.permutation(strides) * rng.uniform(0.85, 1.15, len(per))
            sims.append(np.mean((q >= 0.92) & (np.abs(q - np.round(q)) / np.maximum(np.round(q), 1) <= 0.08)))
        print("  " + g.ljust(26) + f"{len(per):4d} periodic   locked {100 * obs:.0f}%   by chance {100 * np.mean(sims):.0f}% "
              f"(95% {100 * np.quantile(sims, 0.025):.0f}–{100 * np.quantile(sims, 0.975):.0f}%)")
    print("  periods (px), periodic features: " + "  ".join(f"stride {s}: median {np.median([r['period'] for r in rows if r['stride'] == s and r.get('strength', 0) >= 0.3] or [np.nan]):.1f}"
                                                      for s in sorted({r['stride'] for r in rows})))

    print("\n  MULTISCALE — spectral slope β (P ∝ f^-β; the renderer's prior is β = 2.5) and peakiness")
    for g, rs in groups:
        b = np.array([r["beta"] for r in rs])
        ps = np.array([r["peak_share"] for r in rs])
        us = np.mean([r["uniform_share"] for r in rs])
        print("  " + g.ljust(26) + f"β median {np.median(b):.2f} (IQR {np.quantile(b, .25):.2f}–{np.quantile(b, .75):.2f})   "
              f"residual energy within ½ octave of its peak: median {np.median(ps):.2f} (scale-free ≈ {us:.2f})   "
              f"peak scale median {np.median([r['peak_scale_px'] for r in rs]):.1f}px")

    print("\n  PRINTS (vgg16 linework) — box-counting over 2–128px boxes at 896px")
    for s in sorted({r["stride"] for r in prows}):
        rs = [r for r in prows if r["stride"] == s]
        print(f"  stride {s:<3d} n={len(rs):3d}  edge D {np.median([r['D_edge'] for r in rs]):.2f} (R² {np.median([r['r2_edge'] for r in rs]):.3f})   "
              f"ink D {np.median([r['D_ink'] for r in rs]):.2f} (R² {np.median([r['r2_ink'] for r in rs]):.3f})   "
              f"mark size: median {np.median([r['size_median'] for r in rs]):.2f}px, log-sd {np.median([r['log_size_sd'] for r in rs]):.2f}, "
              f"tail slope {np.median([r['tail_slope'] for r in rs]):.2f}   masses with relief marks {np.median([r['masses_with_relief'] / max(r['masses'], 1) for r in rs]) * 100:.0f}%")

    # examples: the clearest lattice of each type
    font = ImageFont.load_default(size=13)
    picks = []
    for t in types:
        cand = sorted([r for r in rows if r.get("type") == t and r.get("strength", 0) >= 0.3], key=lambda r: -r["strength"])[:4]
        picks += [(t, r) for r in cand]
    cell = 230
    sheet = Image.new("RGB", (cell * 4 + 140, cell * len(types)), "white")
    dr = ImageDraw.Draw(sheet)
    for row_i, t in enumerate(types):
        dr.text((8, row_i * cell + 10), t, fill=(0, 0, 0), font=font)
        for j, (tt, r) in enumerate([pr for pr in picks if pr[0] == t]):
            im = Image.open(HERE / r["folder"] / f"{r['name']}.png").convert("RGB")
            d2 = ImageDraw.Draw(im)
            c = 112
            for v, col in ((json.loads(r["v1"]), (255, 255, 255)), (json.loads(r["v2"]) if r["v2"] != "null" else None, (255, 230, 0))):
                if v:
                    d2.line([(c, c), (c + v[0], c + v[1])], fill=col, width=3)
            sheet.paste(im, (140 + j * cell, row_i * cell + 3))
            dr.text((140 + j * cell, row_i * cell + 212), f"{r['model']} {r['layer']} · {r['strength']:.2f}", fill=(60, 60, 60), font=font)
    sheet.save(OUT / "examples.png", optimize=True)
    print(f"\n  {time.time() - t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
