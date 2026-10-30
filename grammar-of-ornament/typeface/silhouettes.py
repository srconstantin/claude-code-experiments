"""
Neuron — the compound pieces' own outlines, as vector shapes to graft onto letters.

A piece (compounds/vgg16features/<kind>/NN.svg) is rasterized, smoothed only
at its own scale (a blur of SMOOTH × its body width, so the traced jaggies go
but its branches, hooks and lobes stay), and its outline is traced back to
vector contours. The contours are expressed in the frame of the piece's spine
(pieces.tree): the spine's start at (0, 0), its end at (1, 0), y to the left.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import skeletonize

import pieces as PC

SMOOTH = 0.32


STROKELIB = PC.ROOT.parent.parent / "strokelib" / "vgg16features"


def source(ref: str):
    """'fork/3' (compounds) or 'strokelib:family-10-short-flared-gentle-arc-branch-fine/3' → the SVG file."""
    if ref.startswith("strokelib:"):
        fam, idx = ref[len("strokelib:"):].rsplit("/", 1)
        return STROKELIB / fam / f"stroke-{int(idx):02d}.svg"
    kind, idx = ref.rsplit("/", 1)
    return PC.ROOT / kind / f"{int(idx):02d}.svg"


@lru_cache(maxsize=None)
def silhouette(ref: str, idx: int | None = None) -> dict:
    if idx is not None:                          # silhouette(kind, idx) as before
        ref = f"{ref}/{idx}"
    ink = PC.raster(source(ref).read_text())
    lab, n = ndi.label(ink)
    if n > 1:
        sz = ndi.sum(ink, lab, range(1, n + 1))
        ink = lab == 1 + int(np.argmax(sz))
    ink = np.pad(ink, 40)
    dt = ndi.distance_transform_edt(ink)
    sk = skeletonize(ink)
    body = float(2 * np.median(dt[sk])) if sk.any() else 6.0
    soft = ndi.gaussian_filter(ink.astype(float), SMOOTH * body)
    m = soft > 0.5
    lab, n = ndi.label(m)
    if n > 1:
        sz = ndi.sum(m, lab, range(1, n + 1))
        m = lab == 1 + int(np.argmax(sz))
        soft = np.where(ndi.binary_dilation(m, iterations=3), soft, 0)
    # spine: the longest skeleton path of the smoothed shape
    sk = skeletonize(m)
    pts = list(zip(*np.nonzero(sk)))
    d0, _ = PC._bfs(sk, pts[0])
    a = max(d0, key=d0.get)
    da, par = PC._bfs(sk, a)
    b = max(da, key=da.get)
    spine = PC._path(par, b)
    A = np.array([spine[0][1], -spine[0][0]], float)        # (x, y up)
    B = np.array([spine[-1][1], -spine[-1][0]], float)
    d = B - A
    L = float(np.hypot(*d))
    u = d / L
    nrm = np.array([-u[1], u[0]])
    contours = []
    for c in measure.find_contours(soft, 0.5):
        xy = np.stack([c[:, 1], -c[:, 0]], 1)
        rel = xy - A
        contours.append(np.stack([rel @ u, rel @ nrm], 1) / L)
    sl = PC._smooth(spine, 4)
    sxy = np.stack([sl[:, 1], -sl[:, 0]], 1) - A
    dts = ndi.distance_transform_edt(m)
    # the spine's own thickness along it (relative to its median): the piece's stroke modulation
    prof = 2 * dts[spine[:, 0].astype(int), spine[:, 1].astype(int)]
    prof = ndi.gaussian_filter1d(prof, max(2.0, len(prof) / 5), mode="nearest")      # its overall swelling, not its lumps
    prof = prof / max(np.median(prof), 1e-6)
    # where side branches leave the spine: skeleton forks next to it, as fractions along the chord
    deg = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    forks = np.argwhere(sk & (deg >= 3))
    bx = []
    for fy, fx in forks:
        dd = np.hypot(spine[:, 0] - fy, spine[:, 1] - fx)
        if dd.min() <= 2.5:
            q = np.array([fx, -fy], float) - A
            bx.append(float(q @ u / L))
    # tips: skeleton end points, with the direction the branch is travelling as it ends
    tips = []
    for ty, tx in np.argwhere(sk & (deg == 1)):
        path = [(ty, tx)]
        prev, cur = None, (ty, tx)
        for _ in range(int(max(6, 1.2 * body))):
            nb = [(cur[0] + dy, cur[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (dy or dx) and 0 <= cur[0] + dy < sk.shape[0] and 0 <= cur[1] + dx < sk.shape[1]
                  and sk[cur[0] + dy, cur[1] + dx] and (cur[0] + dy, cur[1] + dx) != prev]
            if len(nb) != 1:
                break
            prev, cur = cur, nb[0]
            path.append(cur)
        if len(path) < 4:
            continue
        p_end = np.array([tx, -ty], float) - A
        p_back = np.array([path[-1][1], -path[-1][0]], float) - A
        dv = p_end - p_back
        dv /= max(np.hypot(*dv), 1e-9)
        tips.append([float(p_end @ u / L), float(p_end @ nrm / L), float(dv @ u), float(dv @ nrm)])
    return {"contours": contours, "body": body / L, "aspect": L / body, "tips": tips,
            "spine": np.stack([sxy @ u, sxy @ nrm], 1) / L, "profile": prof,
            "branch_x": sorted(set(np.round(bx, 3)))}
