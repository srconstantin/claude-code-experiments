"""The edge sequence of a feature: the colours met crossing its strong edges, from the dark side to the light side.

Features put their colours in a fixed order at every boundary (a dark outline, the body colour, a bright halo, its
opponent). This measures that order so the library can reproduce it round type and boxes.

  profile   the median colour at each distance from an edge, -16 … +16 px (a smooth strip)
  sequence  the most common quantized colour at each distance, consecutive repeats collapsed (real colours)
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import color


def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in np.clip(rgb, 0, 1))


def kmeans(X, k, it=20, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return C


def edge_sequence(im, n_off=8, step=2.0, k=20):
    """im: float RGB in 0–1. Returns (profile, sequence) as lists of hex colours, dark side first."""
    L = color.rgb2lab(im)[..., 0]
    gx, gy = ndi.gaussian_filter(L, 1.5, order=(0, 1)), ndi.gaussian_filter(L, 1.5, order=(1, 0))
    mag = np.hypot(gx, gy)
    ys, xs = np.nonzero(mag > np.percentile(mag, 93))
    nx, ny = gx[ys, xs] / mag[ys, xs], gy[ys, xs] / mag[ys, xs]          # toward the brighter side
    offs = np.arange(-n_off, n_off + 1) * step
    prof = np.zeros((len(offs), len(xs), 3))
    for i, d in enumerate(offs):
        px, py = xs + d * nx, ys + d * ny
        for ch in range(3):
            prof[i, :, ch] = ndi.map_coordinates(im[..., ch], [py, px], order=1, mode="nearest")
    lab = color.rgb2lab(prof)
    med = color.lab2rgb(np.median(lab, axis=1)[None])[0]
    rng = np.random.default_rng(0)
    pix = im.reshape(-1, 3)[rng.choice(im.shape[0] * im.shape[1], 6000, replace=False)]
    C = kmeans(color.rgb2lab(pix[None])[0], k)
    seq = []
    for i in range(len(offs)):
        d = ((lab[i][:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        j = int(np.bincount(d, minlength=k).argmax())
        if not seq or seq[-1] != j:
            seq.append(j)
    return [hexc(m) for m in med], [hexc(color.lab2rgb(C[j][None, None])[0, 0]) for j in seq]


def five(seq):
    """The sequence resampled to five steps, dark end to light end, for the --n-edge-1 … --n-edge-5 variables."""
    idx = np.round(np.linspace(0, len(seq) - 1, 5)).astype(int)
    return [seq[i] for i in idx]
