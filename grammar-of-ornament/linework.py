#!/usr/bin/env python3
"""
Grammar of Ornament — linework: the marks of a feature, as strokes.

For every feature image in a folder, the black-and-white print is made with
the motifs.py pipeline (upscale, restore detail, threshold; default settings),
and then taken apart by scale:

  masses   black that survives a morphological opening with a disc of radius
           --max-stroke-width / 2: the solid areas
  strokes  the thin black that is left (and, separately, the thin white marks
           that the print cuts into the masses: "relief" strokes)
  dots     thin pieces no longer than about twice their width

Each stroke is skeletonized, cut into branches at its junctions, and each
branch becomes a centerline with a width at every point (twice the distance
to the nearest edge of the mark). From those come the measurements:

  width, length, straightness (chord / arc length), curvature, bend (width /
  radius of curvature, scale-free), inflections per length, taper at free
  ends, closed loops, junctions per length, orientation anisotropy and the
  dominant angle, and parallelism (how often a stroke has a neighbour within
  three widths running in the same direction).

Written per feature into <dst>/:
  <name>.strokes.svg    centerlines (masses filled, each stroke as an SVG
                        stroked path at its own mean width, dots as circles)
                        — reusable as strokes
  <name>.outlines.svg   variable-width stroke outlines (faithful fills)
  <name>.strokes.json   the per-stroke table: every stroke's centerline and
                        width profile (print px), its measurements (source
                        px), which junction each end touches and which mass it
                        sits in; the masses' spines traced the same way (pol
                        "mass"); the junctions; and every mass's outline — what
                        strokelib.py and compounds.py work from
  <name>.classes.png    the print with masses grey, strokes black, dots blue,
                        relief strokes orange, relief dots green — for checking
  <name>.json           the measurements, in source-image pixels (224px scale)
and over the folder: summary.json, summary.csv, and a per-layer table printed
at the end.

Run:  python3 linework.py                           # vgg16features/ → linework/vgg16features/
      python3 linework.py --src resnet50features --only layer2
The first pass over a feature upscales it (about 15 s on the CPU); the upscale
is cached in cache/sr/ by motifs.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure, morphology

import motifs

HERE = Path(__file__).resolve().parent
N8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]   # 4-neighbours first
RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # clockwise


# ─────────────────────────────────────────────────────────────────────────────
#  Scale separation: masses / strokes / dots
# ─────────────────────────────────────────────────────────────────────────────

def edt(m: np.ndarray) -> np.ndarray:
    return ndi.distance_transform_edt(m)


def split_scale(mask: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
    """(thick, thin): thick is the opening of `mask` with a disc of radius r (done
    with two distance transforms, which is fast for a large disc); thin is what
    is left of the mask more than r/2 away from the thick part."""
    if not mask.any():
        return mask.copy(), mask.copy()
    eroded = edt(mask) > r
    thick = (edt(~eroded) <= r) if eroded.any() else np.zeros_like(mask)
    near = (edt(~thick) <= r / 2) if thick.any() else np.zeros_like(mask)
    return thick, mask & ~near


def split_dots(thin: np.ndarray, skel: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thin components whose skeleton is shorter than about twice their width are
    dots. Returns (dots, strokes) masks."""
    lab, n = ndi.label(thin, structure=np.ones((3, 3)))
    if n == 0:
        return thin.copy(), thin.copy()
    sk_lab = np.where(skel, lab, 0)
    length = np.bincount(sk_lab.ravel(), minlength=n + 1).astype(float)
    idx = np.arange(1, n + 1)
    width = 2 * np.asarray(ndi.median(dt, sk_lab, idx), float)
    width = np.nan_to_num(width, nan=0.0)
    is_dot = np.zeros(n + 1, bool)
    is_dot[1:] = length[1:] < np.maximum(4.0, 2.0 * width)
    dots = is_dot[lab]
    return dots, thin & ~dots


# ─────────────────────────────────────────────────────────────────────────────
#  Skeleton → branches → centerlines
# ─────────────────────────────────────────────────────────────────────────────

def crossing_number(sk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: the number of skeleton neighbours, and the number of 0→1
    transitions going round the 8-neighbourhood (1 at an end, 2 along a line,
    3 or more at a junction; a staircase step counts as a line)."""
    H, W = sk.shape
    p = np.pad(sk, 1).astype(np.uint8)
    nb = [p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in RING]
    deg = sum(nb)
    cn = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
    return deg, cn


def walk(pts: set) -> list[list[tuple[int, int]]]:
    """Order a set of 8-connected skeleton pixels into chains, starting at ends
    and preferring 4-connected steps so staircase steps are not skipped."""
    deg = {}
    for p in pts:
        deg[p] = sum((p[0] + dy, p[1] + dx) in pts for dy, dx in N8)
    unvisited = set(pts)
    chains = []
    while unvisited:
        ends = [p for p in unvisited if deg[p] <= 1]
        cur = min(ends) if ends else min(unvisited)
        chain = [cur]
        unvisited.discard(cur)
        while True:
            nxt = None
            for dy, dx in N8:
                q = (cur[0] + dy, cur[1] + dx)
                if q in unvisited:
                    nxt = q
                    break
            if nxt is None:
                break
            chain.append(nxt)
            unvisited.discard(nxt)
            cur = nxt
        chains.append(chain)
    return chains


def branches(skel: np.ndarray, dt: np.ndarray) -> tuple[list[dict], list[list[float]]]:
    """Cut the skeleton at its junctions. Each branch: ordered pixel coordinates,
    whether it is a closed loop, and which junction (index into the returned
    junction list, -1 for none) each end touches. Junctions are [x, y, r] in
    print px, r the half-width of the mark there."""
    deg, cn = crossing_number(skel)
    junction = skel & (cn >= 3)
    # cut a 3×3 block around each junction pixel: with only the pixel removed,
    # two arms of a T still touch diagonally and would read as one branch
    cut = ndi.binary_dilation(junction, structure=np.ones((3, 3)))
    body = skel & ~cut
    jlab, nj = ndi.label(cut, structure=np.ones((3, 3)))   # blocks that touch are one junction
    jnear = ndi.grey_dilation(jlab, size=(3, 3))          # the junction label touching each pixel, 0 for none
    lab, n = ndi.label(body, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == i)
        pts = set(zip((ys + sl[0].start).tolist(), (xs + sl[1].start).tolist()))
        for chain in walk(pts):
            a, b = chain[0], chain[-1]
            closed = len(chain) > 8 and max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
            out.append({"pts": np.array(chain, float), "closed": closed,
                        "j0": bool(jnear[a]), "j1": bool(jnear[b]), "jid0": int(jnear[a]) - 1, "jid1": int(jnear[b]) - 1})
    junctions = []
    if nj:
        for (cy, cx) in ndi.center_of_mass(cut, jlab, np.arange(1, nj + 1)):
            junctions.append([round(float(cx), 1), round(float(cy), 1), round(float(dt[int(round(cy)), int(round(cx))]), 1)])
    return out, junctions


def trace(br: dict, dt: np.ndarray, step: float = 3.0, sigma: float = 2.0) -> dict | None:
    """Smooth a branch into a centerline sampled every `step` px, with width,
    tangent angle and curvature at every sample."""
    P = br["pts"]
    w = 2 * dt[P[:, 0].astype(int), P[:, 1].astype(int)]
    mode = "wrap" if br["closed"] else "nearest"
    Ps = ndi.gaussian_filter1d(P, sigma, axis=0, mode=mode)
    ws = ndi.gaussian_filter1d(w, sigma, mode=mode)
    if br["closed"]:
        Ps = np.vstack([Ps, Ps[:1]])
        ws = np.append(ws, ws[0])
    seg = np.hypot(*np.diff(Ps, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    if L < step or len(Ps) < 3:
        return None
    n = max(int(round(L / step)), 2) + 1
    t = np.linspace(0, L, n)
    y, x, wt = np.interp(t, s, Ps[:, 0]), np.interp(t, s, Ps[:, 1]), np.interp(t, s, ws)
    # where a branch ends at a junction, the distance transform measures the
    # junction, not the stroke, and the width bulges; hold it to the stroke's own
    w_ref = float(np.median(wt))
    if not br["closed"]:
        near = np.zeros(n, bool)
        if br["j0"]:
            near |= t <= 1.5 * w_ref
        if br["j1"]:
            near |= (L - t) <= 1.5 * w_ref
        wt = np.where(near, np.minimum(wt, w_ref), wt)
    h = L / (n - 1)
    dy, dx = np.gradient(y, h), np.gradient(x, h)
    theta = np.unwrap(np.arctan2(dy, dx))
    kappa = np.gradient(theta, h)
    return {"y": y, "x": x, "w": wt, "theta": theta, "kappa": kappa, "L": L, "h": h,
            "closed": br["closed"], "j0": br["j0"], "j1": br["j1"], "jid0": br["jid0"], "jid1": br["jid1"]}


# ─────────────────────────────────────────────────────────────────────────────
#  Measurements
# ─────────────────────────────────────────────────────────────────────────────

def stroke_stats(tr: dict) -> dict:
    n = len(tr["y"])
    wt, k = tr["w"], tr["kappa"]
    inner = slice(1, -1) if n > 4 else slice(None)
    mid = wt[n // 4: n - n // 4] if n >= 8 else wt
    tapers, taper_ends = [], [None, None]
    if not tr["closed"]:
        m = max(2, n // 7)
        for i, (free, seg) in enumerate(((not tr["j0"], wt[:m]), (not tr["j1"], wt[-m:]))):
            if free:
                taper_ends[i] = float(seg.mean() / (mid.mean() + 1e-9))
                tapers.append(taper_ends[i])
    chord = float(np.hypot(tr["y"][-1] - tr["y"][0], tr["x"][-1] - tr["x"][0]))
    ks = ndi.gaussian_filter1d(k, 2.0) if n > 6 else k
    sign = np.sign(ks[inner])
    sign = sign[np.abs(ks[inner]) > 0.2 * np.abs(ks[inner]).mean() + 1e-12]
    inflections = int(np.count_nonzero(np.diff(sign))) if len(sign) > 1 else 0
    turn = float(tr["theta"][-1] - tr["theta"][0]) if not tr["closed"] else 2 * np.pi
    # width along the stroke at five equal stations, relative to its mean
    stations = np.interp(np.linspace(0, n - 1, 5), np.arange(n), wt) / (wt.mean() + 1e-9)
    return {
        "length": tr["L"], "width": float(wt.mean()), "width_min": float(wt.min()), "width_max": float(wt.max()),
        "width_cv": float(wt.std() / (wt.mean() + 1e-9)),
        "straightness": 0.0 if tr["closed"] else chord / tr["L"],
        "curvature": float(np.abs(k[inner]).mean()), "bend": float((np.abs(k[inner]) * wt[inner]).mean()),
        "turn": turn, "inflections": inflections, "tapers": tapers, "taper_ends": taper_ends,
        "profile": [round(float(v), 3) for v in stations], "closed": tr["closed"],
        "angle": float(np.degrees(np.angle(np.sum(np.exp(2j * tr["theta"])))) / 2 % 180),
        "orient": complex(np.sum(np.exp(2j * tr["theta"])) * tr["h"]),
    }


def stroke_record(i: int, pol: str, tr: dict, s: dict, k: float) -> dict:
    """One row of the per-stroke table: the centerline and widths in print
    pixels (every other sample), the measurements in source pixels."""
    return {
        "id": i, "pol": pol, "closed": s["closed"], "j0": tr["j0"], "j1": tr["j1"], "j0id": tr["jid0"], "j1id": tr["jid1"],
        "pts": [[round(float(x), 1), round(float(y), 1)] for x, y in zip(tr["x"][::2], tr["y"][::2])],
        "w": [round(float(w), 1) for w in tr["w"][::2]],
        "length": round(s["length"] / k, 3), "width": round(s["width"] / k, 3),
        "width_min": round(s["width_min"] / k, 3), "width_max": round(s["width_max"] / k, 3),
        "width_cv": round(s["width_cv"], 3), "straightness": round(s["straightness"], 3),
        "curvature": round(s["curvature"] * k, 4), "bend": round(s["bend"], 4), "turn": round(s["turn"], 3),
        "inflections": s["inflections"], "taper_start": s["taper_ends"][0], "taper_end": s["taper_ends"][1],
        "profile": s["profile"], "angle": round(s["angle"], 1),
    }


def parallelism(traces: list[dict], w_med: float, rng: np.random.Generator, max_pts: int = 4000) -> tuple[float, float]:
    """Over sampled centerline points: the fraction that have a point of another
    stroke within three widths, and among those, the fraction of such neighbours
    running within 15° of the same direction."""
    ys = np.concatenate([t["y"] for t in traces])
    xs = np.concatenate([t["x"] for t in traces])
    th = np.concatenate([t["theta"] for t in traces])
    ids = np.concatenate([np.full(len(t["y"]), i) for i, t in enumerate(traces)])
    if len(ys) < 2:
        return float("nan"), float("nan")
    R = min(3.0 * w_med, 80.0)
    tree = cKDTree(np.c_[ys, xs])
    pick = rng.choice(len(ys), min(max_pts, len(ys)), replace=False)
    have, par = 0, []
    for i in pick:
        nb = [j for j in tree.query_ball_point((ys[i], xs[i]), R) if ids[j] != ids[i]]
        if not nb:
            continue
        have += 1
        d = np.abs(((th[i] - th[nb] + np.pi / 2) % np.pi) - np.pi / 2)
        par.append(float(np.mean(d < np.radians(15))))
    return (have / len(pick), float(np.mean(par)) if par else float("nan"))


def wmedian(v, w) -> float:
    v, w = np.asarray(v, float), np.asarray(w, float)
    if len(v) == 0:
        return float("nan")
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1] / 2)])


def mass_list(masses: np.ndarray, k: float, tolerance: float = 1.5) -> tuple[list[dict], np.ndarray]:
    """Every mass: area (source px²), centroid and bbox (print px), outline rings
    (print px, approximated). Also the label image."""
    lab, n = ndi.label(masses, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        sub = lab[sl] == i
        y0, x0 = sl[0].start, sl[1].start
        rings = []
        for c in measure.find_contours(np.pad(sub, 1).astype(np.float32), 0.5):
            c = measure.approximate_polygon(c - 1, tolerance)
            if len(c) >= 3:
                rings.append([[round(float(x + x0), 1), round(float(y + y0), 1)] for y, x in c])
        cy, cx = ndi.center_of_mass(sub)
        out.append({"id": i - 1, "area": round(float(sub.sum()) / k / k, 2), "cx": round(float(cx + x0), 1), "cy": round(float(cy + y0), 1),
                    "bbox": [int(x0), int(y0), int(sl[1].stop), int(sl[0].stop)], "rings": rings})
    return out, lab


def summarize(stats: list[dict], traces: list[dict], n_junctions: int, k: float, rng) -> dict:
    """Aggregate one polarity's strokes. Lengths in source pixels (print px / k)."""
    if not stats:
        return {"n_strokes": 0}
    L = np.array([s["length"] for s in stats])
    W = np.array([s["width"] for s in stats])
    tot = float(L.sum())
    orient = sum(s["orient"] for s in stats)
    tapers = [t for s in stats for t in s["tapers"]]
    w_med = wmedian(W, L)
    nbr, par = parallelism(traces, w_med, rng)
    return {
        "n_strokes": len(stats), "total_length": tot / k,
        "width_median": w_med / k, "width_iqr": (wmedian(W, L * (W > w_med)) - wmedian(W, L * (W <= w_med))) / k,
        "length_median": float(np.median(L)) / k, "length_max": float(L.max()) / k,
        "straightness": float(np.average([s["straightness"] for s in stats], weights=L)),
        "curvature": float(np.average([s["curvature"] for s in stats], weights=L)) * k,
        "bend": float(np.average([s["bend"] for s in stats], weights=L)),
        "inflections_per_len": sum(s["inflections"] for s in stats) / tot * k,
        "taper": float(np.mean(tapers)) if tapers else float("nan"),
        "loop_fraction": float(np.mean([s["closed"] for s in stats])),
        "junctions_per_len": n_junctions / tot * k,
        "anisotropy": float(abs(orient) / tot), "dominant_angle_deg": float(np.degrees(np.angle(orient)) / 2 % 180),
        "neighbour_fraction": nbr, "parallelism": par,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SVG
# ─────────────────────────────────────────────────────────────────────────────

def f1(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def bezier(y: np.ndarray, x: np.ndarray, closed: bool) -> str:
    """Catmull-Rom spline through the points as SVG cubic segments."""
    P = np.c_[x, y]
    if closed:
        P = np.vstack([P, P[:1]])
    n = len(P)
    if n < 2:
        return ""
    T = np.zeros_like(P)
    T[1:-1] = (P[2:] - P[:-2]) / 2
    if closed and n > 2:
        T[0] = T[-1] = (P[1] - P[-2]) / 2
    else:
        T[0], T[-1] = P[1] - P[0], P[-1] - P[-2]
    d = [f"M{f1(P[0, 0])},{f1(P[0, 1])}"]
    for i in range(n - 1):
        c1, c2 = P[i] + T[i] / 3, P[i + 1] - T[i + 1] / 3
        d.append(f"C{f1(c1[0])},{f1(c1[1])} {f1(c2[0])},{f1(c2[1])} {f1(P[i + 1, 0])},{f1(P[i + 1, 1])}")
    if closed:
        d.append("Z")
    return "".join(d)


def offset_rings(x: np.ndarray, y: np.ndarray, w: np.ndarray, closed: bool, theta: np.ndarray | None = None) -> list[np.ndarray]:
    """The stroke as filled shape(s): the centerline offset by half its width on
    each side, with round caps. Returns rings as (N, 2) arrays of x, y; a closed
    stroke gives two rings to be filled even-odd."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    if len(x) < 2:
        return []
    if theta is None:
        if closed:
            xc, yc = np.concatenate([x[-1:], x, x[:1]]), np.concatenate([y[-1:], y, y[:1]])
            theta = np.arctan2(yc[2:] - yc[:-2], xc[2:] - xc[:-2])
        else:
            theta = np.arctan2(np.gradient(y), np.gradient(x))
    tx, ty = np.cos(theta), np.sin(theta)
    nx, ny = ty, -tx                      # left normal
    h = w / 2
    left = np.c_[x + nx * h, y + ny * h]
    right = np.c_[x - nx * h, y - ny * h]
    if closed:
        return [left, right[::-1]]
    phis = np.linspace(0, np.pi, 7)[1:-1]
    cap_end = np.array([[x[-1] + h[-1] * (nx[-1] * np.cos(p) + tx[-1] * np.sin(p)),
                         y[-1] + h[-1] * (ny[-1] * np.cos(p) + ty[-1] * np.sin(p))] for p in phis])
    cap_start = np.array([[x[0] + h[0] * (-nx[0] * np.cos(p) - tx[0] * np.sin(p)),
                           y[0] + h[0] * (-ny[0] * np.cos(p) - ty[0] * np.sin(p))] for p in phis])
    return [np.vstack([left, cap_end, right[::-1], cap_start])]


def rings_path(rings: list[np.ndarray]) -> str:
    return " ".join("M" + " L".join(f"{f1(px)},{f1(py)}" for px, py in r) + " Z" for r in rings)


def outline(tr: dict, every: int) -> str:
    return rings_path(offset_rings(tr["x"][::every], tr["y"][::every], tr["w"][::every], tr["closed"], tr["theta"][::every]))


def region_path(mask: np.ndarray, tolerance: float = 1.0) -> str:
    rings = []
    for c in measure.find_contours(np.pad(mask, 1).astype(np.float32), 0.5):
        c = measure.approximate_polygon(c - 1, tolerance)
        if len(c) >= 3:
            rings.append("M" + " L".join(f"{f1(x)},{f1(y)}" for y, x in c) + " Z")
    return " ".join(rings)


def circles(mask: np.ndarray, fill: str) -> str:
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return ""
    idx = np.arange(1, n + 1)
    cy, cx = zip(*ndi.center_of_mass(mask, lab, idx)) if n else ([], [])
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    r = np.sqrt(area / np.pi)
    return "".join(f'<circle cx="{f1(x)}" cy="{f1(y)}" r="{f1(rr)}" fill="{fill}"/>' for y, x, rr in zip(cy, cx, r))


def write_svgs(dst: Path, name: str, shape, masses, dots, rdots, ink, relief, every: int = 2) -> None:
    H, W = shape
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
            f'<rect width="100%" height="100%" fill="#fff"/>')
    mass = f'<path id="masses" fill="#000" fill-rule="evenodd" d="{region_path(masses)}"/>'
    dots_svg = f'<g id="dots">{circles(dots, "#000")}</g><g id="relief-dots">{circles(rdots, "#fff")}</g>'

    def stroked(trs, color):   # each stroke at its own mean width
        return (f'<g fill="none" stroke="{color}" stroke-linecap="round" stroke-linejoin="round">'
                + "".join(f'<path stroke-width="{f1(t["w"].mean())}" d="{bezier(t["y"][::every], t["x"][::every], t["closed"])}"/>'
                          for t in trs if len(t["y"]) >= 2) + "</g>")

    (dst / f"{name}.strokes.svg").write_text(
        head + mass + f'<g id="ink-strokes">{stroked(ink, "#000")}</g>'
        + f'<g id="relief-strokes">{stroked(relief, "#fff")}</g>' + dots_svg + "</svg>")

    def filled(trs, color):
        return "".join(f'<path fill="{color}" fill-rule="nonzero" d="{outline(t, every)}"/>' for t in trs)

    (dst / f"{name}.outlines.svg").write_text(
        head + mass + f'<g id="ink-strokes">{filled(ink, "#000")}</g>'
        + f'<g id="relief-strokes">{filled(relief, "#fff")}</g>' + dots_svg + "</svg>")


def write_classes(dst: Path, name: str, masses, ink_m, dots, relief_m, rdots, px: int = 896) -> None:
    H, W = masses.shape
    img = np.full((H, W, 3), 255, np.uint8)
    img[masses] = (150, 150, 150)
    img[ink_m] = (0, 0, 0)
    img[dots] = (40, 90, 210)
    img[relief_m] = (235, 120, 30)
    img[rdots] = (90, 190, 70)
    Image.fromarray(img).resize((px, px * H // W), Image.NEAREST).save(dst / f"{name}.classes.png", optimize=True)


# ─────────────────────────────────────────────────────────────────────────────
#  One feature
# ─────────────────────────────────────────────────────────────────────────────

def analyse(src: Path, name: str, dst: Path, max_stroke_width: float, seed: int = 0) -> dict:
    white = motifs.render(src, name, motifs.params(None))
    orig_px = Image.open(src / f"{name}.png").size[0]
    k = white.shape[0] / orig_px                      # print px per source px
    r = max_stroke_width * k / 2
    rng = np.random.default_rng(seed)
    black = ~white
    masses, thin_black = split_scale(black, r)
    ground, thin_white = split_scale(white, r)
    dt_black, dt_white = edt(black), edt(white)
    result = {"name": name, "model": name.split("__")[0], "layer": name.split("__")[1],
              "source_px": orig_px, "print_px": int(white.shape[0]), "max_stroke_width": max_stroke_width,
              "ink_fraction": float(black.mean()), "mass_fraction": float(masses.mean()),
              "n_masses": int(ndi.label(masses, structure=np.ones((3, 3)))[1])}
    mass_items, mass_lab = mass_list(masses, k)
    # the mass nearest every pixel (for attaching relief marks to the mass they sit in)
    if masses.any():
        mass_dist, mass_idx = ndi.distance_transform_edt(~masses, return_indices=True)
        mass_near = mass_lab[mass_idx[0], mass_idx[1]] - 1
        mass_near[mass_dist > r] = -1
    else:
        mass_near = np.full(masses.shape, -1)
    traces_by = {}
    masks = {}
    table = []
    junctions = {}
    # the masses are traced too (their spines), as a third polarity
    for pol, thin, dt in (("ink", thin_black, dt_black), ("relief", thin_white, dt_white), ("mass", masses, dt_black)):
        skel = morphology.skeletonize(thin)
        dots, strokes = split_dots(thin, skel, dt)
        brs, jcts = branches(skel & strokes, dt)
        traced = [(b, trace(b, dt)) for b in brs]
        # drop spurs (short branches hanging off a junction) and untraceable
        # scraps; but when a scrap joined two junctions, merge those junctions so
        # the graph stays connected (a wide mark's skeleton is full of them)
        parent = list(range(len(jcts)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        trs = []
        for b, t in traced:
            spur = t is not None and (t["j0"] != t["j1"]) and t["L"] < 2.0 * t["w"].mean()
            if t is None or spur:
                if b["jid0"] >= 0 and b["jid1"] >= 0 and b["jid0"] != b["jid1"]:
                    parent[find(b["jid0"])] = find(b["jid1"])
            else:
                trs.append(t)
        roots = sorted({find(j) for j in range(len(jcts))})
        remap = {rt: n for n, rt in enumerate(roots)}
        for t in trs:
            t["jid0"] = remap[find(t["jid0"])] if t["jid0"] >= 0 else -1
            t["jid1"] = remap[find(t["jid1"])] if t["jid1"] >= 0 else -1
        merged = [[] for _ in roots]
        for j, jc in enumerate(jcts):
            merged[remap[find(j)]].append(jc)
        jcts = [[round(float(np.mean([c[0] for c in g])), 1), round(float(np.mean([c[1] for c in g])), 1), max(c[2] for c in g)] for g in merged]
        stats = [stroke_stats(t) for t in trs]
        result[pol] = summarize(stats, trs, len(jcts), k, rng)
        result[pol].update({"stroke_fraction": float(strokes.mean()), "dot_fraction": float(dots.mean()),
                            "n_dots": int(ndi.label(dots, structure=np.ones((3, 3)))[1])})
        traces_by[pol] = trs
        masks[pol] = (strokes, dots)
        junctions[pol] = jcts
        base = len(table)
        recs = [stroke_record(base + i, pol, t, s, k) for i, (t, s) in enumerate(zip(trs, stats))]
        for rec, t in zip(recs, trs):
            m = len(t["y"]) // 2
            rec["mass"] = int(mass_near[int(round(t["y"][m])), int(round(t["x"][m]))])
        table.extend(recs)
    write_svgs(dst, name, white.shape, masses, masks["ink"][1], masks["relief"][1], traces_by["ink"], traces_by["relief"])
    write_classes(dst, name, masses, masks["ink"][0], masks["ink"][1], masks["relief"][0], masks["relief"][1])
    (dst / f"{name}.json").write_text(json.dumps(result, indent=1))
    (dst / f"{name}.strokes.json").write_text(json.dumps(
        {"name": name, "model": result["model"], "layer": result["layer"], "print_px": int(white.shape[0]),
         "source_px": orig_px, "k": k, "strokes": table, "junctions": junctions, "masses": mass_items}, separators=(",", ":")))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Folder
# ─────────────────────────────────────────────────────────────────────────────

TABLE = [("ink", "n_strokes"), ("ink", "width_median"), ("ink", "length_median"), ("ink", "straightness"),
         ("ink", "bend"), ("ink", "taper"), ("ink", "anisotropy"), ("ink", "parallelism"),
         ("ink", "junctions_per_len"), ("relief", "n_strokes"), ("relief", "parallelism")]


def layer_table(results: list[dict]) -> str:
    def key(layer: str):
        return [int(p) if p.isdigit() else p for p in layer.split(".")]
    layers = sorted({r["layer"] for r in results}, key=key)
    cols = [f"{p[:3]}.{s}" for p, s in TABLE]
    lines = ["  " + "layer".ljust(14) + " n " + " ".join(c.rjust(14) for c in cols)]
    for ly in layers:
        rs = [r for r in results if r["layer"] == ly]
        cells = []
        for pol, s in TABLE:
            v = [r[pol][s] for r in rs if s in r[pol] and r[pol][s] == r[pol][s]]
            cells.append(f"{np.median(v):14.3f}" if v else " " * 14)
        lines.append("  " + ly.ljust(14) + f"{len(rs):2d} " + " ".join(cells))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "vgg16features"))
    ap.add_argument("--dst", default="", help="default: linework/<src folder name>")
    ap.add_argument("--only", default="", help="substring; process only files whose name contains it")
    ap.add_argument("--max-stroke-width", type=float, default=6.0,
                    help="in source pixels: black wider than this is a mass, not a stroke (default 6)")
    ap.add_argument("--force", action="store_true", help="redo outputs that are already up to date")
    a = ap.parse_args()
    src = Path(a.src)
    dst = Path(a.dst) if a.dst else HERE / "linework" / src.name
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.glob("*.png") if not a.only or a.only in f.name)
    if not files:
        print(f"no PNGs in {src}")
        sys.exit(1)
    results, done, skipped = [], 0, 0
    t0 = time.time()
    for i, f in enumerate(files):
        out = dst / f"{f.stem}.json"
        if not a.force and out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
            results.append(json.loads(out.read_text()))
            skipped += 1
            continue
        t1 = time.time()
        r = analyse(src, f.stem, dst, a.max_stroke_width)
        with motifs._LAYERS_LOCK:
            motifs._LAYERS.clear()
        results.append(r)
        done += 1
        ink = r["ink"]
        print(f"  {i + 1}/{len(files)}  {f.stem[:52]}  {ink['n_strokes']} strokes, w {ink.get('width_median', 0):.2f}px, "
              f"bend {ink.get('bend', 0):.3f}, par {ink.get('parallelism', 0):.2f}  ({time.time() - t1:.0f}s, total {time.time() - t0:.0f}s)",
              flush=True)
    (dst / "summary.json").write_text(json.dumps(results, indent=1))
    with open(dst / "summary.csv", "w", newline="") as fh:
        keys = ["name", "model", "layer", "ink_fraction", "mass_fraction", "n_masses"]
        pk = [k for k in results[0]["ink"] if k != "n_strokes"] + ["n_strokes"]
        w = csv.writer(fh)
        w.writerow(keys + [f"ink.{k}" for k in pk] + [f"relief.{k}" for k in pk])
        for r in results:
            w.writerow([r[k] for k in keys] + [r["ink"].get(k, "") for k in pk] + [r["relief"].get(k, "") for k in pk])
    print(f"\n  {done} done, {skipped} already up to date, {time.time() - t0:.0f}s → {dst}\n")
    print("  per-layer medians (lengths in source px; ink = black strokes, relief = white marks in the masses):")
    print(layer_table(results))


if __name__ == "__main__":
    main()
