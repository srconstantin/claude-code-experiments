"""
Neuron — the combs, feathers, sprigs and forks of the compounds folder, as
structures a letter stroke can carry.

Each compound piece (compounds/vgg16features/<kind>/NN.svg) is rasterized,
thinned to its skeleton, and read as a tree:

  spine   the longest path through the skeleton
  teeth   every other branch that leaves the spine: where it leaves (u, the
          fraction along the spine), its path in the spine's local frame at
          that point (x along the spine, y across it, in units of the spine's
          length; y > 0 on the left), and its width relative to the spine's.
          Spurs left by thinning (branches shorter than about the stroke's
          width) are dropped.

A letter stroke then carries a piece's teeth: the same positions, angles,
curves and proportions, redrawn as smooth tapering strokes (glyphs.py).
Results are cached in cache/typeface-pieces.json.
"""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent.parent / "compounds" / "vgg16features"
CACHE = Path(__file__).resolve().parent.parent / "cache" / "typeface-pieces.json"
KINDS = ["straight-comb", "curved-comb", "wavy-comb", "straight-feather", "curved-feather", "wavy-feather",
         "straight-sprig", "curved-sprig", "wavy-sprig", "fork"]
PX = 700          # rasterize so the piece's long side is this many pixels


def raster(svg: str) -> np.ndarray:
    vb = [float(x) for x in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    s = PX / max(vb[2], vb[3])
    W, H = int(vb[2] * s) + 4, int(vb[3] * s) + 4
    acc = np.zeros((H, W), bool)
    for d in re.findall(r' d="([^"]+)"', svg):
        m = np.zeros((H, W), bool)
        for part in re.split(r"(?=M)", d):
            n = re.findall(r"-?\d+(?:\.\d+)?", part)
            if len(n) >= 6:
                r = np.array(n, float).reshape(-1, 2)
                one = Image.new("1", (W, H), 0)
                ImageDraw.Draw(one).polygon([((x - vb[0]) * s + 2, (y - vb[1]) * s + 2) for x, y in r], fill=1)
                m ^= np.asarray(one, bool)
        acc |= m
    return acc


_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _bfs(sk: np.ndarray, start, allowed=None):
    """Distances and parents over skeleton pixels from start."""
    H, W = sk.shape
    dist = {start: 0}
    par = {start: None}
    q = deque([start])
    while q:
        p = q.popleft()
        for dy, dx in _N8:
            n = (p[0] + dy, p[1] + dx)
            if 0 <= n[0] < H and 0 <= n[1] < W and sk[n] and n not in dist and (allowed is None or allowed[n]):
                dist[n] = dist[p] + 1
                par[n] = p
                q.append(n)
    return dist, par


def _path(par, end):
    out = []
    while end is not None:
        out.append(end)
        end = par[end]
    return np.array(out[::-1], float)


def _smooth(P, sigma=3.0):
    if len(P) < 5:
        return P
    return ndi.gaussian_filter1d(P, sigma, axis=0, mode="nearest")


def tree(kind: str, idx: int) -> dict | None:
    ink = raster((ROOT / kind / f"{idx:02d}.svg").read_text())
    lab, n = ndi.label(ink)
    if n > 1:                                               # the largest piece only
        sz = ndi.sum(ink, lab, range(1, n + 1))
        ink = lab == 1 + int(np.argmax(sz))
    ink = ndi.binary_closing(ink, iterations=2)
    dt = ndi.distance_transform_edt(ink)
    sk = skeletonize(ink)
    pts = list(zip(*np.nonzero(sk)))
    if len(pts) < 20:
        return None
    d0, _ = _bfs(sk, pts[0])
    a = max(d0, key=d0.get)
    da, par = _bfs(sk, a)
    b = max(da, key=da.get)
    spine = _path(par, b)                                    # (row, col)
    if len(spine) < 30:
        return None
    sp = _smooth(spine, 4)
    seg = np.hypot(*np.diff(sp, axis=0).T)
    s_at = np.concatenate([[0], np.cumsum(seg)])
    L = s_at[-1]
    tan = np.gradient(sp, axis=0)
    tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)[:, None]
    sw = float(2 * np.median(dt[spine[:, 0].astype(int), spine[:, 1].astype(int)]))
    # teeth: skeleton left after removing the spine (and a band around it as wide as the spine)
    on = np.zeros_like(sk)
    on[spine[:, 0].astype(int), spine[:, 1].astype(int)] = True
    near = ndi.binary_dilation(on, iterations=max(2, int(0.6 * sw)))
    rest = sk & ~near
    lab, n = ndi.label(rest, structure=np.ones((3, 3)))
    teeth = []
    for k in range(1, n + 1):
        comp = lab == k
        # attach: the component pixel closest to the spine
        cy, cx = np.nonzero(comp)
        dd = np.min(np.hypot(cy[:, None] - spine[None, :, 0], cx[:, None] - spine[None, :, 1]), axis=1)
        j = int(np.argmin(dd))
        start = (int(cy[j]), int(cx[j]))
        dist, par2 = _bfs(sk, start, allowed=comp)
        far = max(dist, key=dist.get)
        path = _path(par2, far)
        # prepend the run back to the spine so the tooth starts on it
        at = int(np.argmin(np.hypot(*(spine - path[0]).T)))
        path = np.vstack([spine[at:at + 1], path])
        length = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
        if length < 1.4 * sw or length < 0.04 * L:
            continue                                         # a spur left by thinning
        P = _smooth(path, 2.5)
        # local frame at the attach point: x along the spine tangent, y to its left (in image coords, y down)
        t = tan[at]
        nrm = np.array([-t[1], t[0]])
        rel = P - sp[at]
        xy = np.stack([rel @ t, rel @ nrm], 1) / L
        # image rows grow downward, so flip the across axis to the usual (y up) handedness
        xy[:, 1] *= -1
        tw = float(2 * np.median(dt[path[:, 0].astype(int), path[:, 1].astype(int)]))
        teeth.append({"u": float(s_at[at] / L), "pts": np.round(xy[::max(1, len(xy) // 12)], 4).tolist() + [np.round(xy[-1], 4).tolist()],
                      "w": round(tw / max(sw, 1e-6), 3)})
    teeth.sort(key=lambda t: t["u"])
    curve = sp[:, ::-1] * [1, -1]
    return {"kind": kind, "idx": idx, "spine_w_rel": round(sw / L, 4), "teeth": teeth,
            "spine": np.round((curve[::max(1, len(curve) // 24)] - curve[0]) / L, 4).tolist()}


def library() -> dict[str, list[dict]]:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    for kind in KINDS:
        out[kind] = []
        for f in sorted((ROOT / kind).glob("[0-9][0-9].svg")):
            t = tree(kind, int(f.stem))
            if t and t["teeth"]:
                out[kind].append(t)
        print(f"  {kind}: {len(out[kind])} pieces with teeth", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    lib = library()
    for k, v in lib.items():
        nt = [len(p["teeth"]) for p in v]
        print(k, "teeth per piece", nt)
