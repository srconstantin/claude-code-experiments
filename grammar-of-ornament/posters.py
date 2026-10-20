#!/usr/bin/env python3
"""
Grammar of Ornament — posters: five cover and poster studies, each testing one
idea for a visual style made from the features.

  1  figure     a deep-layer motif, huge and cropped off the edge, on flat
                paper, with a small frieze of its siblings          (Haeckel)
  2  friezes    a cover built of friezes alone, every row from one motif
                family, in different arrangements and sizes          (Turing)
  3  screen     a feature printed through itself: its large form screened
                by its own pattern at a smaller scale                (poster)
  4  quiet      one convnext texture at a few percent, embossed; the type
                carries the cover                           (D'Arcy Thompson)
  5  composed   (not run by default) the network-painted poster from
                compose.py; it mixes two features' textures, which clash

Each study draws its pattern from a single source: one feature, or one motif
and its siblings.

Pictures are made here with numpy (from the features, their Real-ESRGAN
upscales, the cleaned motifs and the frieze generator), laid out in HTML
with the Mac's own typefaces, and photographed by headless Chrome.

Output: posters/<n>-<name>.html, its assets in posters/assets/, and
posters/<n>-<name>.png. Run:  python3 posters.py            # all
                              python3 posters.py --only 2,3
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color

import friezes as FR
import motifs as M

HERE = Path(__file__).resolve().parent
OUT = HERE / "posters"
ASSETS = OUT / "assets"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CW, CH = 1200, 1800          # cover size in CSS px (2:3)

FEAT = {"vgg16": HERE / "vgg16features", "resnet50": HERE / "resnet50features", "convnext_base": HERE / "convnext_basefeatures"}


def say(m: str) -> None:
    print(m, flush=True)


def feature_name(key: str) -> tuple[Path, str]:
    """('resnet50', 'layer1.1__c201') → (folder, full feature name)."""
    model, part = key.split(":")
    d = FEAT[model]
    f = next(iter(sorted(d.glob(f"{model}__{part}__*.png"))))
    return d, f.stem


def big(key: str) -> np.ndarray:
    """The feature at 4× (896 px), float RGB 0–1, and seamless: the upscaler
    draws artifacts along an image's edges, so the feature is padded with a
    border wrapped round from its opposite sides, upscaled, and the border cut
    away again (cached in cache/sr_tile/)."""
    d, n = feature_name(key)
    f = HERE / "cache" / "sr_tile" / f"{n}.png"
    if not f.exists():
        t0 = __import__("time").time()
        rgb = np.asarray(Image.open(d / f"{n}.png").convert("RGB"))
        h, w = rgb.shape[:2]
        b = 24
        up = upscale_rgb(np.pad(rgb, ((b, b), (b, b), (0, 0)), mode="wrap"), tile=136)[4 * b:4 * (b + h), 4 * b:4 * (b + w)]
        f.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((up * 255).round().astype(np.uint8)).save(f)
        say(f"  upscaled {n[:50]} seamlessly in {__import__('time').time() - t0:.0f}s")
    return seamless(np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255)


def seamless(a: np.ndarray, band: float = 0.18) -> np.ndarray:
    """Hide whatever seam is left: near the edges, cross-fade to a copy shifted
    by half a tile, whose seam lies in the middle where this one is solid."""
    h, w = a.shape[:2]
    ramp1 = lambda n: np.clip(np.minimum(np.arange(n), n - 1 - np.arange(n)) / (band * n), 0, 1)
    wgt = np.minimum.outer(ramp1(h), ramp1(w))
    wgt = (wgt * wgt * (3 - 2 * wgt))[..., None] if a.ndim == 3 else wgt * wgt * (3 - 2 * wgt)
    return a * wgt + np.roll(a, (h // 2, w // 2), (0, 1)) * (1 - wgt)


def lum(rgb: np.ndarray) -> np.ndarray:
    return color.rgb2gray(rgb)


def stretch(a: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    p, q = np.percentile(a, [lo, hi])
    return np.clip((a - p) / (q - p + 1e-9), 0, 1)


def tiled(a: np.ndarray, h: int, w: int, scale: float = 1.0, shift=(0, 0)) -> np.ndarray:
    """A periodic feature repeated to cover h × w (the renders wrap seamlessly)."""
    if scale != 1.0:
        a = ndi.zoom(a, (scale, scale) + (1,) * (a.ndim - 2), order=1, mode="grid-wrap", grid_mode=True)
    reps = (h // a.shape[0] + 2, w // a.shape[1] + 2) + (1,) * (a.ndim - 2)
    t = np.tile(a, reps)
    return t[shift[0]:shift[0] + h, shift[1]:shift[1] + w]


def hexrgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


def ramp(t: np.ndarray, *hexes: str) -> np.ndarray:
    """Map 0–1 through a run of colours."""
    cs = np.stack([hexrgb(h) for h in hexes])
    x = np.clip(t, 0, 1) * (len(cs) - 1)
    i = np.minimum(x.astype(int), len(cs) - 2)
    f = (x - i)[..., None]
    return cs[i] * (1 - f) + cs[i + 1] * f


def save(a: np.ndarray, name: str, q: int = 92) -> str:
    ASSETS.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8) if a.dtype != np.uint8 else a)
    p = ASSETS / name
    if name.endswith(".jpg"):
        im.convert("RGB").save(p, quality=q)
    else:
        im.save(p, optimize=True)
    return f"assets/{name}"


def motif(key: str, sib: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A cleaned motif's ink and body (bool arrays)."""
    d = next(iter(sorted((HERE / "vgg16motifs-clean").glob(f"*{key}*/*/motif.json")))).parent
    f, m = (d / "motif.png", d / "mask.png") if sib == 0 else (d / f"siblings/s{sib}.png", d / f"siblings/s{sib}.mask.png")
    ink = np.asarray(Image.open(f).convert("L")) < 128
    body = np.asarray(Image.open(m).convert("L")) > 127
    if body.shape != ink.shape:
        body = ndi.zoom(body.astype(np.float32), (ink.shape[0] / body.shape[0], ink.shape[1] / body.shape[1]), order=0) > 0.5
    return ink, body


def enlarge(mask: np.ndarray, s: float) -> np.ndarray:
    """A one-bit image enlarged s times with smooth edges (blur, then threshold)."""
    a = ndi.zoom(mask.astype(np.float32), s, order=1)
    return ndi.gaussian_filter(a, max(0.5, 0.45 * s)) > 0.5


def place(canvas_shape, arr: np.ndarray, y0: int, x0: int) -> np.ndarray:
    """arr pasted into an empty canvas at (y0, x0), clipped at the edges."""
    out = np.zeros(canvas_shape, arr.dtype)
    H, W = arr.shape
    ys, xs = slice(max(0, y0), min(canvas_shape[0], y0 + H)), slice(max(0, x0), min(canvas_shape[1], x0 + W))
    if ys.start < ys.stop and xs.start < xs.stop:
        out[ys, xs] = arr[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    return out


def page(title: str, css: str, body: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
html,body{{margin:0;background:#888}}
.cover{{position:relative;width:{CW}px;height:{CH}px;overflow:hidden}}
.cover>*{{position:absolute}}
img.full{{inset:0;width:100%;height:100%;object-fit:cover}}
{css}
</style></head><body><div class="cover">{body}</div></body></html>"""


def shoot(html_path: Path, png_path: Path) -> None:
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([CHROME, "--headless=new", f"--user-data-dir={prof}", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", f"--window-size={CW},{CH}", f"--screenshot={png_path}",
                        "--virtual-time-budget=4000", html_path.as_uri()], check=True, capture_output=True, timeout=120)


# ─────────────────────────────────────────────────────────────────────────────
#  1  one figure
# ─────────────────────────────────────────────────────────────────────────────
#  Every study takes its pattern from one source only: a single feature, or a
#  single motif with its siblings. Two different features' textures together
#  clash.

FIGURE_MOTIF = "features.24__c11"


def family(key: str) -> dict:
    return next(f for f in FR.motif_families() if key in f["id"])


def frieze_file(fam: dict, name: str, arr: str, colours: dict, scheme: str = "ink", seed: int = 0, bare: bool = True) -> str:
    """One frieze of a motif family as an SVG asset; bare drops its own ground so the cover's paper shows."""
    cw = {"name": "cover", "paper": colours["paper"], "ink": colours["ink"], "ink2": colours.get("ink2", colours["ink"]),
          "deep": colours["ink"], "light": colours["paper"]}
    svg, _ = FR.frieze(fam, random.Random(seed), cw, None, arr=arr, scheme=scheme, rails="none")
    if bare:
        svg = svg.replace(f'<rect width="100%" height="100%" fill="{colours["paper"]}"/>', "")
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / name).write_text(svg)
    return f"assets/{name}"


def study_figure() -> str:
    say(f"1 figure: the {FIGURE_MOTIF} motif, huge, on flat paper, with a frieze of its siblings")
    paper, fill, deep = "#e8dfc9", "#b9542d", "#231a14"
    img = np.ones((CH, CW, 3), np.float32) * hexrgb(paper)
    ink, body = motif(FIGURE_MOTIF)
    text = np.zeros((CH, CW), bool)
    text[CH - 600:CH - 60, 40:640] = True            # the title's corner stays clear
    text[40:270, 40:660] = True                      # and the author's
    ink0, body0 = ink, body
    s = 1.15 * CW / ink0.shape[1]
    while True:   # as large as possible with ~15% cropped off the right edge, shrinking until the type is clear
        ink, body = enlarge(ink0, s), enlarge(body0, s)
        H, W = body.shape
        y0, x0 = max(int(0.13 * CH), (CH - H) // 2 - 120), CW - int(0.85 * W)
        if not (place((CH, CW), body, y0, x0) & text).any() or s < 0.2:
            break
        s *= 0.95
    b, k = place((CH, CW), body, y0, x0), place((CH, CW), ink, y0, x0)
    img[b] = hexrgb(fill)
    img[b & k] = hexrgb(deep)
    src = save(img.astype(np.float32), "1-figure.png")
    fr = frieze_file(family(FIGURE_MOTIF), "1-frieze.svg", "shift", {"paper": paper, "ink": deep}, seed=3)
    css = f"""
.t{{left:70px;bottom:120px;font:400 168px/0.84 Didot,serif;color:{deep};letter-spacing:-2px}}
.t i{{font-style:italic;font-size:96px;display:block;margin:0 0 18px 6px;letter-spacing:0}}
.a{{left:76px;top:78px;font:600 21px/1.35 'Avenir Next Condensed';letter-spacing:5px;text-transform:uppercase;color:{deep}}}
.a span{{display:block;font-weight:400;letter-spacing:3px;opacity:.8}}
.fr{{left:70px;top:168px;width:560px;height:72px}}
.n{{left:76px;bottom:62px;font:italic 20px Didot;color:{deep};opacity:.7}}"""
    body_ = f"""<img class="full" src="{src}"><div class="a">Ernst Haeckel<span>with plates after layer 24</span></div>
<img class="fr" src="{fr}" style="object-fit:cover;object-position:left">
<div class="t"><i>Kunstformen</i>der<br>Natur</div><div class="n">vgg16 · features.24 · channel 11, and six of its siblings</div>"""
    return page("Kunstformen der Natur", css, body_)


# ─────────────────────────────────────────────────────────────────────────────
#  2  a cover of friezes, all from one motif family
# ─────────────────────────────────────────────────────────────────────────────

FRIEZE_MOTIF = "features.19__c117"


def study_friezes() -> str:
    say(f"2 friezes: a cover of friezes, every row from the {FRIEZE_MOTIF} family")
    fam = family(FRIEZE_MOTIF)
    col = {"paper": "#efe8da", "ink": "#2a1d17", "ink2": "#8c3b2a"}
    rows = [  # (arrangement, scheme, height px, x offset)
        ("stack", "ink", 300, -60), ("turn", "two", 140, -180), ("pulse", "ink", 210, 0),
        ("glide", "two", 120, -90), ("mirror", "ink", 260, -240), ("shift", "two", 90, -30),
        ("stack2", "ink", 250, -150),
    ]
    out, y, title_y = [], 0, 0
    for i, (arr, scheme, h, dx) in enumerate(rows):
        src = frieze_file(fam, f"2-row{i}.svg", arr, col, scheme=scheme, seed=i)
        w = h * FR.W / FR.H
        out.append(f'<img src="{src}" style="left:{dx}px;top:{y}px;width:{w:.0f}px;height:{h}px">')
        y += h - 6
        if i == 2:
            title_y = y
            y += 300
    css = f"""
.cover{{background:{col['paper']}}}
.t{{left:64px;top:{title_y + 24}px;width:1080px;font:700 92px/0.92 'Futura';letter-spacing:-1px;color:{col['ink']};text-transform:uppercase}}
.t em{{font:500 italic 40px/1 'Futura';text-transform:none;letter-spacing:0;color:{col['ink2']};display:block;margin-top:22px}}
.a{{left:68px;top:{title_y + 262}px;font:500 20px 'Futura';letter-spacing:6px;color:{col['ink2']}}}"""
    body_ = "".join(out) + f"""<div class="t">The Chemical Basis<br>of Morphogenesis<em>A. M. Turing, 1952</em></div>
<div class="a">PHIL. TRANS. R. SOC. B</div>"""
    return page("The Chemical Basis of Morphogenesis", css, body_)


# ─────────────────────────────────────────────────────────────────────────────
#  3  a feature printed through itself
# ─────────────────────────────────────────────────────────────────────────────

SCREEN_FEATURE = "resnet50:layer3.5__c191"


def study_screen() -> str:
    say(f"3 screen: {SCREEN_FEATURE} printed through its own fine pattern at a sixth of the size")
    f = lum(big(SCREEN_FEATURE))
    eq = lambda x: (np.argsort(np.argsort(x, axis=None)).reshape(x.shape) / x.size).astype(np.float32)   # flat histogram
    tone = 0.08 + 0.84 * eq(ndi.gaussian_filter(tiled(f, CH, CW, scale=2.4, shift=(420, 180)), 3.0))
    small = tiled(f, CH, CW, scale=0.4, shift=(37, 211))
    screen = eq(ndi.gaussian_filter(small, 0.8) - ndi.gaussian_filter(small, 10))   # its fine pattern only
    a = (1 - tone) > screen                          # ink coverage follows the large form's darkness
    paper, ink = hexrgb("#f1ece2"), hexrgb("#1c2a4a")
    img = np.where(a[..., None], ink, paper).astype(np.float32)
    # bare paper along the bottom for the type, below an edge that follows the picture's own tone
    edge = 0.74 * CH + 70 * ndi.gaussian_filter1d(tone[int(0.74 * CH)], 18)
    img[np.arange(CH)[:, None] > edge[None, :]] = paper
    src = save(img, "3-screen.png")
    css = """
.cover{background:#f1ece2}
.big{left:52px;bottom:48px;font:700 250px/0.8 'DIN Condensed';color:#1c2a4a;letter-spacing:-4px}
.r{right:58px;bottom:62px;text-align:right;font:500 25px/1.3 'Avenir Next Condensed';color:#1c2a4a;letter-spacing:1px}
.r b{display:block;font-weight:700;color:#d8462f;letter-spacing:4px;text-transform:uppercase;font-size:20px;margin-bottom:8px}"""
    body_ = f"""<img class="full" src="{src}"><div class="big">LAYER 3.5</div>
<div class="r"><b>Printed through itself</b>One channel at two scales:<br>the large form, screened<br>by its own small pattern<br>resnet50 · channel 191</div>"""
    return page("Layer 3.5", css, body_)


# ─────────────────────────────────────────────────────────────────────────────
#  4  quiet texture
# ─────────────────────────────────────────────────────────────────────────────

def emboss(height: np.ndarray, strength: float, light=(-1.0, -1.4)) -> np.ndarray:
    gy, gx = np.gradient(height)
    lx, ly = light
    n = np.hypot(lx, ly)
    return 1 + strength * (gx * lx + gy * ly) / n


def study_quiet() -> str:
    say("4 quiet: convnext features.3.1 c228 alone, at a few percent, embossed; the type carries the cover")
    t = lum(big("convnext_base:features.3.1__c228"))
    t = stretch(ndi.gaussian_filter(tiled(t, CH, CW, scale=1.3), 1.2))
    shade = emboss(ndi.gaussian_filter(t, 2.5), 1.4)
    paper = hexrgb("#ece7dd")
    tint = 1 - 0.035 * (t - 0.5)                      # the texture itself, at about 3% luminance
    img = paper * (tint * shade)[..., None]
    src = save(img, "4-quiet.jpg", q=94)
    css = """
.t{left:96px;top:1010px;font:400 132px/0.98 'Big Caslon';color:#2b2622;letter-spacing:-1px}
.t i{font-family:'Hoefler Text';font-style:italic;font-size:64px;color:#7b4a32;letter-spacing:0}
.a{left:100px;top:1418px;font:400 26px/1.5 'Hoefler Text';color:#2b2622;letter-spacing:6px;text-transform:uppercase}
.line{left:100px;top:1388px;width:300px;height:2px;background:#7b4a32}
.c{left:100px;bottom:84px;font:italic 21px 'Hoefler Text';color:#857a6e}"""
    body_ = f"""<img class="full" src="{src}"><div class="t">On Growth<br><i>and</i> Form</div><div class="line"></div>
<div class="a">D'Arcy Wentworth Thompson</div><div class="c">Ground: convnext_base features.3.1 · channel 228</div>"""
    return page("On Growth and Form", css, body_)


# ─────────────────────────────────────────────────────────────────────────────
#  5  the composed poster
# ─────────────────────────────────────────────────────────────────────────────

def upscale_rgb(rgb: np.ndarray, tile: int = 224, pad: int = 16) -> np.ndarray:
    """Real-ESRGAN 4× in overlapping tiles (the CPU won't take a poster in one go)."""
    import torch
    import torch.nn.functional as F
    net = M._net()
    x = rgb.astype(np.float32) / 255
    H, W = x.shape[:2]
    out = np.zeros((H * 4, W * 4, 3), np.float32)
    for y in range(0, H, tile):
        for xx in range(0, W, tile):
            y0, x0 = max(0, y - pad), max(0, xx - pad)
            y1, x1 = min(H, y + tile + pad), min(W, xx + tile + pad)
            t = torch.from_numpy(x[y0:y1, x0:x1].transpose(2, 0, 1)[None])
            with torch.no_grad():
                r = net(t)[0].clamp(0, 1).numpy().transpose(1, 2, 0)
            oy, ox = (y - y0) * 4, (xx - x0) * 4
            hh, ww = min(tile, H - y) * 4, min(tile, W - xx) * 4
            out[y * 4:y * 4 + hh, xx * 4:xx * 4 + ww] = r[oy:oy + hh, ox:ox + ww]
    return out


def study_composed(preset: str = "vgg") -> str:
    f = OUT / "composed" / f"{preset}.png"
    say(f"5 composed: {f.name}, upscaled 4×")
    rec = json.loads(f.with_suffix(".json").read_text())
    big_ = OUT / "composed" / f"{preset}.x4.jpg"
    if not big_.exists() or big_.stat().st_mtime < f.stat().st_mtime:
        up = upscale_rgb(np.asarray(Image.open(f).convert("RGB")))
        Image.fromarray((up * 255).round().astype(np.uint8)).save(big_, quality=93)
    a = np.asarray(Image.open(big_).convert("RGB")).astype(np.float32) / 255
    e = int(0.02 * a.shape[1])                       # the optimizer's padding leaves a grey rim
    a = a[e:-e, e:-e]
    img = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((CW, CH), Image.LANCZOS))
    src = save(img, f"5-composed-{preset}.jpg")
    fl, fc = rec["figure"]
    gl, gc = rec["ground"]
    css = """
.t{left:80px;top:1452px;font:700 104px/0.9 'Bodoni 72';color:#1d1916;letter-spacing:-1px}
.s{left:84px;top:1660px;font:500 21px/1.4 'Avenir Next Condensed';color:#1d1916;letter-spacing:3px;text-transform:uppercase}"""
    body_ = f"""<img class="full" src="{src}"><div class="t">Grammar of Ornament</div>
<div class="s">{rec['model']} · figure {fl} c{fc} · ground {gl} c{gc} · painted to this layout by the network</div>"""
    return page("Grammar of Ornament", css, body_)


STUDIES = {1: ("figure", study_figure), 2: ("friezes", study_friezes), 3: ("screen", study_screen),
           4: ("quiet", study_quiet), 5: ("composed", study_composed)}   # 5 is kept but not run by default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="1,2,3,4")
    ap.add_argument("--composed", default="vgg", help="which compose.py presets to set as posters (comma list)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for n in [int(x) for x in a.only.split(",")]:
        name, fn = STUDIES[n]
        jobs = [(f"{n}-{name}-{p}", lambda p=p: fn(p)) for p in a.composed.split(",")] if n == 5 else [(f"{n}-{name}", fn)]
        for stem, job in jobs:
            h = OUT / f"{stem}.html"
            h.write_text(job())
            shoot(h, OUT / f"{stem}.png")
            say(f"  wrote {h.name} and {stem}.png")


if __name__ == "__main__":
    main()
