#!/usr/bin/env python3
"""
Grammar of Ornament — colorways: the colour effects of a feature set, as
gradients, duotones and mesh-gradient backgrounds.

Feature visualizations pair opponent colours: the network's colour channels
are decorrelated, so a feature that likes one hue dislikes its opposite, and
the image swings between the two. This script finds that pair for every
feature image in a folder and turns it into things a web or graphic designer
can use directly.

Per feature:
  pair       the two colours at the ends of the image's dominant chroma axis
             (first principal component of a, b in CIELAB, after a light blur;
             each end is the mean colour of the 5% of pixels farthest that way),
             with the hue angle between them (180° = exact opponents) and how
             much of the image's chroma variance the axis explains
  gradient   seven stops along the axis, each the mean colour of one septile of
             the pixels, so the gradient passes through the image's own
             intermediate colours; and the pure pair, interpolated in Lab
  duotone    the pair pushed to a dark shadow and a light highlight, as hex
             colours and as an SVG filter that maps any image's luminance onto
             them; applied to the feature itself as a demonstration
  mesh       the image blurred heavily, chroma lifted a little, and summarized
             as a CSS background: a base colour plus a few radial gradients at
             its colour blobs (the aurora / mesh-gradient look)

Written into <dst>/:
  <name>.json         everything above (hex, Lab, CSS strings, SVG filter)
  <name>.card.png     the feature, the pair, both gradients, the duotone, the mesh
  colorways.css       custom properties and classes for every feature
  filters.svg         the duotone filters, one per feature, id="duo-<short>"
  gallery.html        everything in one page (open it from this folder)
  summary.json / summary.csv, pairs.png (every pair, sorted by hue), and a
  table of hue pairs by layer printed at the end

Run:  python3 colorways.py --src convnext_basefeatures        # → colorways/convnext_basefeatures/
      python3 colorways.py --src vgg16features --mesh-sigma 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import color

HERE = Path(__file__).resolve().parent

HUE_NAMES = [(25, "red"), (70, "orange"), (110, "yellow"), (170, "green"), (230, "cyan"), (290, "blue"), (345, "purple"), (360, "red")]


# ─────────────────────────────────────────────────────────────────────────────
#  Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def hue_name(h: float) -> str:
    h %= 360
    for lim, name in HUE_NAMES:
        if h < lim:
            return name
    return "red"


def lab_to_hex(lab) -> str:
    rgb = np.clip(color.lab2rgb(np.asarray(lab, float).reshape(1, 1, 3))[0, 0], 0, 1)
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb)


def in_gamut(lab) -> bool:
    rgb = color.lab2rgb(np.asarray(lab, float).reshape(1, 1, 3))[0, 0]
    return bool(np.all(rgb > -0.002) and np.all(rgb < 1.002))


def fit_gamut(lab) -> np.ndarray:
    """Pull chroma in until the colour is displayable, keeping L and hue."""
    lab = np.asarray(lab, float).copy()
    for _ in range(40):
        if in_gamut(lab):
            break
        lab[1:] *= 0.95
    return lab


def hue_of(lab) -> float:
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360)


def chroma_of(lab) -> float:
    return float(np.hypot(lab[1], lab[2]))


# ─────────────────────────────────────────────────────────────────────────────
#  The opponent pair, gradients, duotone, mesh
# ─────────────────────────────────────────────────────────────────────────────

def opponent_pair(lab: np.ndarray, tail: float = 0.05) -> dict:
    ab = lab[..., 1:].reshape(-1, 2)
    mean = ab.mean(0)
    c = ab - mean
    vals, vecs = np.linalg.eigh(c.T @ c / len(c))
    axis = vecs[:, np.argmax(vals)]
    proj = c @ axis
    flat = lab.reshape(-1, 3)
    lo, hi = np.quantile(proj, [tail, 1 - tail])
    a, b = flat[proj <= lo].mean(0), flat[proj >= hi].mean(0)
    # order the pair warm first (hue nearer red/orange), purely for stable naming
    if hue_of(a) > hue_of(b) and not (hue_of(a) > 300 and hue_of(b) < 60):
        a, b = b, a
    ha, hb = hue_of(a), hue_of(b)
    diff = abs(ha - hb) % 360
    return {"a": a, "b": b, "hue_a": ha, "hue_b": hb, "name_a": hue_name(ha), "name_b": hue_name(hb),
            "opponency": min(diff, 360 - diff), "chroma_a": chroma_of(a), "chroma_b": chroma_of(b),
            "axis_share": float(vals.max() / (vals.sum() + 1e-9)), "proj": proj.reshape(lab.shape[:2]), "axis": axis}


def gradient_stops(lab: np.ndarray, proj: np.ndarray, n: int = 7) -> list[np.ndarray]:
    """The colour of each n-tile of the pixels along the axis: the chroma-weighted
    mean, so the stop keeps the hue of the bin's coloured pixels instead of
    sinking into the grey ones averaged with them."""
    q = np.quantile(proj, np.linspace(0, 1, n + 1))
    flat, p = lab.reshape(-1, 3), proj.ravel()
    chroma = np.hypot(flat[:, 1], flat[:, 2]) + 1.0
    stops = []
    for i in range(n):
        m = (p >= q[i]) & (p <= q[i + 1])
        stops.append(np.average(flat[m], axis=0, weights=chroma[m]) if m.any() else flat.mean(0))
    return stops


def pair_stops(a: np.ndarray, b: np.ndarray, n: int = 7) -> list[np.ndarray]:
    return [fit_gamut(a + (b - a) * t) for t in np.linspace(0, 1, n)]


def css_gradient(stops: list[np.ndarray], angle: int = 90) -> str:
    n = len(stops)
    return f"linear-gradient({angle}deg, " + ", ".join(f"{lab_to_hex(s)} {100 * i / (n - 1):.0f}%" for i, s in enumerate(stops)) + ")"


def duotone(a: np.ndarray, b: np.ndarray) -> dict:
    shadow, high = (a, b) if a[0] <= b[0] else (b, a)
    shadow = fit_gamut([min(shadow[0], 28.0), shadow[1], shadow[2]])
    high = fit_gamut([max(high[0], 88.0), high[1], high[2]])
    s_rgb = np.clip(color.lab2rgb(shadow.reshape(1, 1, 3))[0, 0], 0, 1)
    h_rgb = np.clip(color.lab2rgb(high.reshape(1, 1, 3))[0, 0], 0, 1)
    table = "".join(f'<feFuncR type="table" tableValues="{s_rgb[0]:.3f} {h_rgb[0]:.3f}"/>'
                    f'<feFuncG type="table" tableValues="{s_rgb[1]:.3f} {h_rgb[1]:.3f}"/>'
                    f'<feFuncB type="table" tableValues="{s_rgb[2]:.3f} {h_rgb[2]:.3f}"/>'.split("\n"))
    return {"shadow": lab_to_hex(shadow), "highlight": lab_to_hex(high), "shadow_rgb": s_rgb, "highlight_rgb": h_rgb,
            "filter": f'<filter id="{{id}}" color-interpolation-filters="sRGB"><feColorMatrix type="saturate" values="0"/>'
                      f'<feComponentTransfer>{table}</feComponentTransfer></filter>'}


def apply_duotone(rgb: np.ndarray, d: dict) -> np.ndarray:
    lum = color.rgb2gray(rgb)[..., None]
    return d["shadow_rgb"] + (d["highlight_rgb"] - d["shadow_rgb"]) * lum


def mesh_field(rgb: np.ndarray, sigma: float, chroma_boost: float = 1.4) -> np.ndarray:
    """The image blurred heavily (wrap-around, since the renders are periodic),
    with the chroma lifted: the colour field the CSS mesh is read from."""
    lab = color.rgb2lab(rgb)
    sm = np.stack([ndi.gaussian_filter(lab[..., i], sigma, mode="wrap") for i in range(3)], -1)
    sm[..., 1:] *= chroma_boost
    return np.clip(color.lab2rgb(sm), 0, 1)


def mesh_css(field: np.ndarray, k: int = 5, seed: int = 0) -> dict:
    """A base colour plus k radial gradients at the field's colour blobs
    (k-means over Lab and position)."""
    from scipy.cluster.vq import kmeans2
    H, W = field.shape[:2]
    small = field[::max(1, H // 64), ::max(1, W // 64)]
    h, w = small.shape[:2]
    lab = color.rgb2lab(small).reshape(-1, 3)
    yy, xx = np.mgrid[0:h, 0:w]
    pos = np.c_[yy.ravel() / h, xx.ravel() / w] * 60      # position weighted like ~60 Lab units across the tile
    X = np.c_[lab, pos]
    cent, lab_idx = kmeans2(X, k, minit="++", seed=seed)
    base = lab.mean(0)
    blobs = []
    for i in range(k):
        m = lab_idx == i
        if not m.any():
            continue
        col = lab[m].mean(0)
        y, x = yy.ravel()[m].mean() / h, xx.ravel()[m].mean() / w
        r = np.sqrt(m.mean()) * 90
        blobs.append({"x": round(float(100 * x), 1), "y": round(float(100 * y), 1), "r": round(float(r), 1), "color": lab_to_hex(col)})
    css = ", ".join(f"radial-gradient(circle at {b['x']}% {b['y']}%, {b['color']} 0%, transparent {b['r']:.0f}%)" for b in blobs)
    return {"base": lab_to_hex(base), "blobs": blobs, "css": f"{css}, {lab_to_hex(base)}"}


# ─────────────────────────────────────────────────────────────────────────────
#  Cards and gallery
# ─────────────────────────────────────────────────────────────────────────────

def hex_rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def card(rgb: np.ndarray, r: dict, duo_img: np.ndarray, mesh: np.ndarray, px: int = 256) -> Image.Image:
    font = ImageFont.load_default(size=12)
    W, H = px * 5 + 60, px + 40
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    im.paste(Image.fromarray((rgb * 255).astype(np.uint8)).resize((px, px), Image.BICUBIC), (10, 10))
    dr.text((10, px + 16), r["layer"], fill=(0, 0, 0), font=font)
    x = px + 20
    dr.rectangle([x, 10, x + px // 2 - 2, 10 + px], fill=hex_rgb(r["pair"]["a"]))
    dr.rectangle([x + px // 2 + 2, 10, x + px, 10 + px], fill=hex_rgb(r["pair"]["b"]))
    dr.text((x, px + 16), f"{r['pair']['a']} {r['pair']['name_a']} / {r['pair']['b']} {r['pair']['name_b']}  {r['pair']['opponency']:.0f} deg",
            fill=(0, 0, 0), font=font)
    x += px + 10
    n = len(r["gradient"]["image"])
    for i, (hx, hp) in enumerate(zip(r["gradient"]["image"], r["gradient"]["pair"])):
        x0, x1 = x + i * px // n, x + (i + 1) * px // n
        dr.rectangle([x0, 10, x1, 10 + px // 2 - 2], fill=hex_rgb(hx))
        dr.rectangle([x0, 10 + px // 2 + 2, x1, 10 + px], fill=hex_rgb(hp))
    dr.text((x, px + 16), "gradient: image (top) / pair (bottom)", fill=(0, 0, 0), font=font)
    x += px + 10
    im.paste(Image.fromarray((duo_img * 255).astype(np.uint8)).resize((px, px), Image.BICUBIC), (x, 10))
    dr.text((x, px + 16), f"duotone {r['duotone']['shadow']} -> {r['duotone']['highlight']}", fill=(0, 0, 0), font=font)
    x += px + 10
    im.paste(Image.fromarray((mesh * 255).astype(np.uint8)).resize((px, px), Image.BICUBIC), (x, 10))
    dr.text((x, px + 16), f"mesh field ({len(r['mesh']['blobs'])} radial gradients in CSS)", fill=(0, 0, 0), font=font)
    return im


def gallery(dst: Path, src: Path, results: list[dict]) -> None:
    rel = Path("..") / ".." / src.name if src.is_absolute() is False else src
    try:
        rel = Path(__import__("os").path.relpath(src, dst))
    except ValueError:
        pass
    rows = []
    for r in results:
        s = r["short"]
        rows.append(f"""
<section class="cw-{s}">
  <h2>{r['name']}</h2>
  <div class="row">
    <img src="{rel.as_posix()}/{r['name']}.png" width="160" height="160" alt="">
    <img src="{rel.as_posix()}/{r['name']}.png" width="160" height="160" alt="" style="filter:url(filters.svg#duo-{s})">
    <div class="sw" style="background:{r['pair']['a']}"></div><div class="sw" style="background:{r['pair']['b']}"></div>
    <div class="bar" style="background:{r['gradient']['image_css']}"></div>
    <div class="bar" style="background:{r['gradient']['pair_css']}"></div>
    <div class="mesh" style="background:{r['mesh']['css']}"></div>
  </div>
  <p>{r['pair']['name_a']} / {r['pair']['name_b']} · {r['pair']['opponency']:.0f}° apart · axis explains {100 * r['pair']['axis_share']:.0f}% of chroma
     · duotone {r['duotone']['shadow']} → {r['duotone']['highlight']}</p>
</section>""")
    (dst / "gallery.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Colorways — {src.name}</title>
<link rel="stylesheet" href="colorways.css">
<style>
body{{font:13px/1.4 system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}}
h1{{font-size:18px}} h2{{font-size:12px;font-weight:normal;color:#666;margin:0 0 6px}}
section{{margin:0 0 22px;padding:12px;background:#fff;border:1px solid #e6e6e6;border-radius:6px}}
.row{{display:flex;gap:8px;align-items:stretch;height:160px}}
.sw{{width:48px}} .bar{{width:160px}} .mesh{{width:160px}} img{{display:block}}
p{{margin:8px 0 0;color:#444}}
</style></head><body>
<h1>Colorways — {src.name}</h1>
<p>Per feature: the image · its duotone · the opponent pair · gradient through the image · pure pair gradient · mesh as CSS radial gradients. Classes and custom properties are in <code>colorways.css</code>, duotone filters in <code>filters.svg</code>.</p>
{''.join(rows)}
</body></html>""")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def analyse(src: Path, name: str, dst: Path, mesh_sigma: float) -> dict:
    rgb = np.asarray(Image.open(src / f"{name}.png").convert("RGB")).astype(np.float32) / 255
    lab = color.rgb2lab(np.stack([ndi.gaussian_filter(rgb[..., i], 1.0) for i in range(3)], -1))
    p = opponent_pair(lab)
    img_stops = gradient_stops(lab, p["proj"])
    pr_stops = pair_stops(p["a"], p["b"])
    d = duotone(p["a"], p["b"])
    mesh = mesh_field(rgb, mesh_sigma)
    m = mesh_css(mesh)
    parts = name.split("__")
    short = "".join(ch for ch in (parts[1] + "-" + parts[2]) if ch.isalnum() or ch == "-").replace("features", "f")
    r = {"name": name, "short": short, "model": parts[0], "layer": parts[1],
         "pair": {"a": lab_to_hex(p["a"]), "b": lab_to_hex(p["b"]), "lab_a": [round(float(v), 1) for v in p["a"]],
                  "lab_b": [round(float(v), 1) for v in p["b"]], "hue_a": round(p["hue_a"], 1), "hue_b": round(p["hue_b"], 1),
                  "name_a": p["name_a"], "name_b": p["name_b"], "opponency": round(p["opponency"], 1),
                  "chroma_a": round(p["chroma_a"], 1), "chroma_b": round(p["chroma_b"], 1), "axis_share": round(p["axis_share"], 3)},
         "gradient": {"image": [lab_to_hex(s) for s in img_stops], "pair": [lab_to_hex(s) for s in pr_stops],
                      "image_css": css_gradient(img_stops), "pair_css": css_gradient(pr_stops)},
         "duotone": {"shadow": d["shadow"], "highlight": d["highlight"], "svg_filter": d["filter"].format(id=f"duo-{short}")},
         "mesh": {"sigma": mesh_sigma, **m}}
    card(rgb, r, apply_duotone(rgb, d), mesh).save(dst / f"{name}.card.png", optimize=True)
    (dst / f"{name}.json").write_text(json.dumps(r, indent=1))
    return r


def pairs_strip(results: list[dict], dst: Path, h: int = 60) -> None:
    rs = sorted(results, key=lambda r: min(r["pair"]["hue_a"], r["pair"]["hue_b"]))
    w = 12
    im = Image.new("RGB", (w * len(rs), h), "white")
    dr = ImageDraw.Draw(im)
    for i, r in enumerate(rs):
        dr.rectangle([i * w, 0, (i + 1) * w - 1, h // 2 - 1], fill=hex_rgb(r["pair"]["a"]))
        dr.rectangle([i * w, h // 2, (i + 1) * w - 1, h - 1], fill=hex_rgb(r["pair"]["b"]))
    im.save(dst / "pairs.png", optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "convnext_basefeatures"))
    ap.add_argument("--dst", default="", help="default: colorways/<src folder name>")
    ap.add_argument("--only", default="", help="substring; process only files whose name contains it")
    ap.add_argument("--mesh-sigma", type=float, default=12.0, help="blur radius for the mesh field, in source px")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    src = Path(a.src)
    dst = Path(a.dst) if a.dst else HERE / "colorways" / src.name
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.glob("*.png") if not a.only or a.only in f.name)
    if not files:
        print(f"no PNGs in {src}")
        sys.exit(1)
    t0 = time.time()
    results = []
    for i, f in enumerate(files):
        out = dst / f"{f.stem}.json"
        if not a.force and out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
            results.append(json.loads(out.read_text()))
            continue
        r = analyse(src, f.stem, dst, a.mesh_sigma)
        results.append(r)
        print(f"  {i + 1}/{len(files)}  {f.stem[:48]}  {r['pair']['a']} {r['pair']['name_a']:7s} / {r['pair']['b']} {r['pair']['name_b']:7s}"
              f"  {r['pair']['opponency']:3.0f}°  axis {100 * r['pair']['axis_share']:.0f}%  ({time.time() - t0:.0f}s)", flush=True)

    # set-level files
    css = [":root{"] + [f"  --cw-{r['short']}-a:{r['pair']['a']}; --cw-{r['short']}-b:{r['pair']['b']};"
                        f" --cw-{r['short']}-shadow:{r['duotone']['shadow']}; --cw-{r['short']}-highlight:{r['duotone']['highlight']};"
                        for r in results] + ["}"]
    for r in results:
        s = r["short"]
        css += [f".cw-{s}-gradient{{background:{r['gradient']['image_css']}}}",
                f".cw-{s}-pair{{background:{r['gradient']['pair_css']}}}",
                f".cw-{s}-mesh{{background:{r['mesh']['css']}}}",
                f".cw-{s}-duotone{{filter:url(filters.svg#duo-{s})}}"]
    (dst / "colorways.css").write_text("\n".join(css) + "\n")
    (dst / "filters.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><defs>' + "".join(r["duotone"]["svg_filter"] for r in results) + "</defs></svg>")
    gallery(dst, src, results)
    pairs_strip(results, dst)
    (dst / "summary.json").write_text(json.dumps(results, indent=1))
    with open(dst / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "layer", "a", "name_a", "hue_a", "chroma_a", "b", "name_b", "hue_b", "chroma_b", "opponency", "axis_share", "shadow", "highlight"])
        for r in results:
            p = r["pair"]
            w.writerow([r["name"], r["layer"], p["a"], p["name_a"], p["hue_a"], p["chroma_a"], p["b"], p["name_b"], p["hue_b"], p["chroma_b"],
                        p["opponency"], p["axis_share"], r["duotone"]["shadow"], r["duotone"]["highlight"]])

    # which hue pairs, by layer stage
    def stage(layer: str) -> str:
        return ".".join(layer.split(".")[:2])
    pairs = sorted({tuple(sorted((r["pair"]["name_a"], r["pair"]["name_b"]))) for r in results})
    stages = sorted({stage(r["layer"]) for r in results}, key=lambda s: [int(p) if p.isdigit() else p for p in s.split(".")])
    print(f"\n  hue pairs by layer (count of features):")
    print("  " + "pair".ljust(18) + "".join(s.rjust(12) for s in stages) + "   all")
    for pr in pairs:
        row = [sum(1 for r in results if stage(r["layer"]) == s and tuple(sorted((r["pair"]["name_a"], r["pair"]["name_b"]))) == pr) for s in stages]
        print("  " + "/".join(pr).ljust(18) + "".join(f"{c:12d}" for c in row) + f"{sum(row):6d}")
    opp = [r["pair"]["opponency"] for r in results]
    print(f"\n  opponency (hue angle between the pair): median {np.median(opp):.0f}°, {100 * np.mean(np.array(opp) > 120):.0f}% of features over 120°")
    print(f"  {time.time() - t0:.0f}s → {dst}")


if __name__ == "__main__":
    main()
