#!/usr/bin/env python3
"""
Grammar of Ornament — motifs: pure black-and-white renderings of the features,
and single motifs cut out of them by hand.

Pipeline (per feature, deterministic):
  1. upscale 4× (224 → 896px) with Real-ESRGAN (the photo model, x4plus): a
     learned super-resolution network that turns the pixel mush of a small
     feature into clean shapes while staying faithful to it
  2. put back the fine texture the upscaler smoothed away: the 2–6px band of
     the original minus the same band of the upscaled image shrunk back, so
     dots and hatching return exactly where the feature has them
  3. one threshold over the sum of
       tone       the shading, edge-preserving (bilateral) so it stops at forms
       texture    the upscaled image's own fine detail, locally normalized
       restored   the texture from step 2, locally normalized
       boundaries minus the colour-edge strength, which draws black lines
                  where forms meet even if they differ only in hue
     taken at 2× (1792px) so the curves are smooth. Mark width then follows the
     shading: thin white marks in the darks, thin black ones in the lights.
  4. optionally, a hand-drawn outline keeps one motif and floods the rest
     with the ground colour.

The upscaled images are cached in cache/sr/. The model weights (64 MB) are
downloaded once into models/.

Run:  python3 motifs.py                 # every whole feature → prints/<name>.png
      python3 motifs.py --only c141 --texture 0.3 --restore 0.6 --force
The Motifs tab in the web app previews the same pipeline and saves motifs cut
out of it into motifs/<feature>/<id>/ as motif.png, motif.svg and motif.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, measure
from skimage.draw import polygon2mask
from skimage.restoration import denoise_bilateral


HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
SR_CACHE = HERE / "cache" / "sr"
PRINTS = HERE / "prints"
MOTIFS = HERE / "vgg16motifs"
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS = MODELS / "RealESRGAN_x4plus.pth"
WORK = 896     # the upscaled feature; crop outlines are in these pixels
OUT = 1792     # the thresholded output

DEFAULTS = {"texture": 0.45, "restore": 0.45, "boundaries": 1.2, "threshold": 0.5, "invert": False, "ground": "white"}
RANGES = {"texture": (0.0, 1.5), "restore": (0.0, 1.5), "boundaries": (0.0, 3.0), "threshold": (0.2, 0.8)}


def say(msg: str) -> None:
    print(f"  {time.strftime('%H:%M:%S')}  motifs: {msg}", flush=True)


def params(given: dict | None) -> dict:
    """Defaults overlaid with `given`, numbers clamped to their ranges."""
    p = dict(DEFAULTS)
    for k, v in (given or {}).items():
        if k in RANGES and v not in (None, ""):
            lo, hi = RANGES[k]
            p[k] = float(min(hi, max(lo, float(v))))
        elif k == "invert":
            p[k] = v in (True, 1, "1", "true", "on")
        elif k == "ground" and v in ("white", "black"):
            p[k] = v
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Upscaling (Real-ESRGAN x4plus: RRDBNet with 23 blocks)
# ─────────────────────────────────────────────────────────────────────────────

_NET = None
_NET_LOCK = threading.Lock()


def _net():
    global _NET
    if _NET is not None:
        return _NET
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)

        def forward(self, x):
            a = lambda t: F.leaky_relu(t, 0.2)
            x1 = a(self.conv1(x))
            x2 = a(self.conv2(torch.cat((x, x1), 1)))
            x3 = a(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = a(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = RDB(), RDB(), RDB()

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, nb=23):
            super().__init__()
            self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB() for _ in range(nb)])
            self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        def forward(self, x):
            a = lambda t: F.leaky_relu(t, 0.2)
            f = self.conv_first(x)
            f = f + self.conv_body(self.body(f))
            f = a(self.conv_up1(F.interpolate(f, scale_factor=2, mode="nearest")))
            f = a(self.conv_up2(F.interpolate(f, scale_factor=2, mode="nearest")))
            return self.conv_last(a(self.conv_hr(f)))

    if not WEIGHTS.exists():
        MODELS.mkdir(exist_ok=True)
        say(f"downloading the upscaler's weights (64 MB) to {WEIGHTS} …")
        tmp = WEIGHTS.with_suffix(".part")
        with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as r, open(tmp, "wb") as f:
            total, got, last = int(r.headers.get("Content-Length") or 0), 0, 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total and got * 10 // total > last:
                    last = got * 10 // total
                    say(f"  {last * 10}%")
        tmp.rename(WEIGHTS)
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    net = RRDBNet()
    net.load_state_dict(sd.get("params_ema", sd.get("params", sd)), strict=True)
    _NET = net.eval()   # on the CPU: about 13 s per feature, and bit-for-bit repeatable
    return _NET


def upscaled(features_dir: Path, name: str) -> np.ndarray:
    """The feature at 4×, float RGB in 0–1; computed once and kept in cache/sr/."""
    out = SR_CACHE / f"{name}.png"
    src = features_dir / f"{name}.png"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return np.asarray(Image.open(out).convert("RGB")).astype(np.float32) / 255
    import torch
    import torch.nn.functional as F
    with _NET_LOCK:
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:   # another request made it while this one waited
            return np.asarray(Image.open(out).convert("RGB")).astype(np.float32) / 255
        net = _net()
        t0 = time.time()
        rgb = np.asarray(Image.open(src).convert("RGB")).astype(np.float32) / 255
        with torch.no_grad():
            x = F.pad(torch.from_numpy(rgb.transpose(2, 0, 1)[None]), (8, 8, 8, 8), mode="reflect")
            y = net(x)[0, :, 32:-32, 32:-32].clamp(0, 1).numpy().transpose(1, 2, 0)
        SR_CACHE.mkdir(parents=True, exist_ok=True)
        Image.fromarray((y * 255).round().astype(np.uint8)).save(out)
        say(f"upscaled {name[:48]} in {time.time() - t0:.1f}s")
    return np.asarray(Image.open(out).convert("RGB")).astype(np.float32) / 255


# ─────────────────────────────────────────────────────────────────────────────
#  Layers and the threshold
# ─────────────────────────────────────────────────────────────────────────────

_LAYERS: dict[str, dict] = {}
_LAYERS_LOCK = threading.Lock()


def _stretch(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def _norm(f: np.ndarray) -> np.ndarray:
    """Local contrast normalization into about ±1, so texture counts equally in light and dark areas."""
    return np.tanh(f / (1.5 * (np.sqrt(ndi.gaussian_filter(f * f, 20)) + 0.25 * f.std() + 1e-9)))


def layers(features_dir: Path, name: str) -> dict:
    """The four layers of a feature at the working size (see the module docstring), cached in memory."""
    with _LAYERS_LOCK:
        if name in _LAYERS:
            return _LAYERS[name]
    orig = np.asarray(Image.open(features_dir / f"{name}.png").convert("RGB")).astype(np.float32) / 255
    sr = upscaled(features_dir, name)
    L = color.rgb2gray(sr)
    smooth = denoise_bilateral(sr, sigma_color=0.1, sigma_spatial=6, channel_axis=-1)
    tone = _stretch(ndi.gaussian_filter(color.rgb2gray(smooth), 3))
    fine = _norm(ndi.gaussian_filter(L, 1.5) - ndi.gaussian_filter(L, 7))
    lab = color.rgb2lab(ndi.gaussian_filter(smooth, (1.5, 1.5, 0)))
    edge = np.sqrt(sum(filters.sobel(lab[..., i]) ** 2 for i in range(3)))
    edge = np.clip(edge / np.percentile(edge, 97), 0, 1.5)
    band = lambda a: ndi.gaussian_filter(a, 0.6) - ndi.gaussian_filter(a, 1.8)
    k = orig.shape[0] / L.shape[0]
    lost = band(color.rgb2gray(orig)) - band(ndi.zoom(ndi.gaussian_filter(L, 2), k, order=1))
    lost = _norm(ndi.zoom(lost, 1 / k, order=3))
    out = {"tone": tone.astype(np.float32), "fine": fine.astype(np.float32), "lost": lost.astype(np.float32),
           "edge": np.clip(edge - 0.5, 0, None).astype(np.float32)}
    with _LAYERS_LOCK:
        if len(_LAYERS) >= 6:
            _LAYERS.pop(next(iter(_LAYERS)))
        _LAYERS[name] = out
    return out


def render(features_dir: Path, name: str, p: dict, crop=None, px: int = OUT) -> np.ndarray:
    """The print as a boolean image (True = white), px wide. With a crop outline
    ([[x, y], …] in working pixels) everything outside it is the ground colour."""
    ly = layers(features_dir, name)
    v = ly["tone"] + p["texture"] * ly["fine"] + p["restore"] * ly["lost"] - p["boundaries"] * ly["edge"]
    white = ndi.zoom(v, px / v.shape[0], order=3) > p["threshold"]
    if p["invert"]:
        white = ~white
    if crop and len(crop) >= 3:
        k = px / WORK
        inside = polygon2mask(white.shape, [(y * k, x * k) for x, y in crop])
        white = np.where(inside, white, p["ground"] == "white")
    return white


def to_image(white: np.ndarray) -> Image.Image:
    return Image.fromarray(white).convert("1")


def trim(white: np.ndarray, crop, ground_white: bool, margin: float = 0.06) -> np.ndarray:
    """Cut a cropped print down to a square around its outline, with a margin of ground."""
    if not crop or len(crop) < 3:
        return white
    k = white.shape[0] / WORK
    xs, ys = [x * k for x, _ in crop], [y * k for _, y in crop]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * (1 + 2 * margin)
    n = int(round(2 * half))
    out = np.full((n, n), ground_white, bool)
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(white.shape[1], x0 + n), min(white.shape[0], y0 + n)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = white[sy0:sy1, sx0:sx1]
    return out


def svg(white: np.ndarray, tolerance: float = 0.6) -> str:
    """The black areas as one even-odd path on a white ground (holes stay open)."""
    h, w = white.shape
    rings = []
    for c in measure.find_contours(np.pad(~white, 1).astype(np.float32), 0.5):
        c = measure.approximate_polygon(c - 1, tolerance)
        if len(c) >= 3:
            rings.append("M" + " L".join(f"{x:.1f},{y:.1f}" for y, x in c) + " Z")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'<rect width="100%" height="100%" fill="#fff"/><path fill="#000" fill-rule="evenodd" d="{" ".join(rings)}"/></svg>')


# ─────────────────────────────────────────────────────────────────────────────
#  The hand-drawn outline
# ─────────────────────────────────────────────────────────────────────────────

def snap_outline(im: np.ndarray, pts: list, sigma: float = 3.0, radius: float = 14.0) -> list[list[float]]:
    """Magnetic lasso: reroute a closed outline, given as waypoints [[x, y], …]
    on the upscaled feature, along the strongest edges near it. Each waypoint is
    pulled to the strongest gradient within half the radius; then each leg
    between successive waypoints is replaced by the cheapest path through a
    corridor `radius` pixels wide around it, where cost falls with the gradient
    magnitude of the luminance (blurred by `sigma`). Returns a dense closed polyline."""
    from skimage.graph import route_through_array
    t = color.rgb2gray(im)
    H, W = t.shape
    gm = filters.sobel(ndi.gaussian_filter(t, max(0.3, float(sigma))))
    gm = np.clip(gm / (np.percentile(gm, 99) + 1e-9), 0, 1)
    cost = (1.0 - gm) ** 2 + 0.02
    r = max(3.0, float(radius))
    rr = int(round(r / 2))
    way = []
    for x, y in pts:
        cx, cy = int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, H - 1))
        win = gm[max(0, cy - rr):cy + rr + 1, max(0, cx - rr):cx + rr + 1]
        k = np.unravel_index(int(np.argmax(win)), win.shape)
        way.append((max(0, cy - rr) + int(k[0]), max(0, cx - rr) + int(k[1])))
    out = []
    for i in range(len(way)):
        (ay, ax), (by, bx) = way[i], way[(i + 1) % len(way)]
        Y0, Y1 = max(0, min(ay, by) - int(r) - 1), min(H, max(ay, by) + int(r) + 2)
        X0, X1 = max(0, min(ax, bx) - int(r) - 1), min(W, max(ax, bx) + int(r) + 2)
        sub = cost[Y0:Y1, X0:X1].copy()
        yy, xx = np.mgrid[Y0:Y1, X0:X1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        u = np.clip(((xx - ax) * dx + (yy - ay) * dy) / L2, 0, 1) if L2 else 0.0
        sub[np.hypot(xx - (ax + u * dx), yy - (ay + u * dy)) > r] = 1e6   # stay in the corridor around the drawn leg
        path, _ = route_through_array(sub, (ay - Y0, ax - X0), (by - Y0, bx - X0), fully_connected=True, geometric=True)
        out.extend([[x + X0 + 0.5, y + Y0 + 0.5] for y, x in path[:-1]])
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Saved motifs
# ─────────────────────────────────────────────────────────────────────────────

def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def save_motif(features_dir: Path, name: str, p: dict, crop, label: str = "") -> dict:
    """The print, cut to the outline if there is one, as a motif in
    motifs/<feature>/<id>[-label]/: motif.png (one bit per pixel), motif.svg
    and motif.json (the settings and the outline, so it can be reopened)."""
    mid = str(int(time.time())) + (f"-{slug(label)}" if slug(label) else "")
    out = MOTIFS / name / mid
    out.mkdir(parents=True, exist_ok=True)
    white = trim(render(features_dir, name, p, crop), crop, p["ground"] == "white")
    to_image(white).save(out / "motif.png", optimize=True)
    (out / "motif.svg").write_text(svg(white))
    item = {"id": f"{name}/{mid}", "feature": name, "label": label, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "params": p,
            "crop": [[round(float(x), 1), round(float(y), 1)] for x, y in crop] if crop else None, "work_px": WORK,
            "width": int(white.shape[1]), "height": int(white.shape[0]), "url": f"/motifs/{name}/{mid}/motif.png"}
    (out / "motif.json").write_text(json.dumps(item, indent=1))
    return item


def list_motifs() -> list[dict]:
    items = []
    if MOTIFS.exists():
        for p in sorted(MOTIFS.glob("*/*/motif.json")):
            try:
                items.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                continue
    items.sort(key=lambda it: it["saved"], reverse=True)
    return items


def delete_motif(motif_id: str) -> None:
    if ".." in motif_id or motif_id.count("/") != 1:
        raise ValueError("bad id")
    d = MOTIFS / motif_id
    if not (d / "motif.json").exists():
        raise FileNotFoundError(motif_id)
    shutil.rmtree(d)
    if d.parent.exists() and not any(d.parent.iterdir()):
        d.parent.rmdir()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "vgg16features"))
    ap.add_argument("--dst", default=str(PRINTS))
    ap.add_argument("--only", default="", help="substring; process only files whose name contains it")
    for k, (lo, hi) in RANGES.items():
        ap.add_argument(f"--{k}", type=float, default=DEFAULTS[k], help=f"{lo}–{hi} (default {DEFAULTS[k]})")
    ap.add_argument("--invert", action="store_true", help="swap black and white")
    ap.add_argument("--force", action="store_true", help="redo outputs that are already up to date")
    a = ap.parse_args()
    p = params({k: getattr(a, k) for k in RANGES} | {"invert": a.invert})
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.glob("*.png") if not a.only or a.only in f.name)
    if not files:
        print(f"no PNGs in {src}")
        sys.exit(1)
    done = skipped = 0
    t0 = time.time()
    for i, f in enumerate(files):
        out = dst / f.name
        if not a.force and out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
            skipped += 1
            continue
        to_image(render(src, f.stem, p)).save(out, optimize=True)
        with _LAYERS_LOCK:
            _LAYERS.clear()
        done += 1
        print(f"  {i + 1}/{len(files)}  {f.name}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"  {done} done, {skipped} already up to date, {time.time() - t0:.1f}s → {dst}")


if __name__ == "__main__":
    main()
