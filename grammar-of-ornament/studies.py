#!/usr/bin/env python3
"""
Grammar of Ornament — studies: layouts after notes/design_references.md.

Each study takes its pattern from one source: one motif family (a cleaned
vgg16 motif and its siblings, vgg16motifs-clean/) or one feature. Colour is
black and white, or a palette drawn from a convnext_base feature: k-means on
the feature's own pixels, six colours sorted dark to light.

  plate       specimen plate: one motif, magnified insets of it, callouts and data
  emblem      imperfect symmetry: a motif and its mirrored sibling frame a void
  kaleido     four siblings in a rhombus split into black and white triangles
  invaded     a huge word on a black/white split, overrun by the motif's ink
  outline     text running round the motif's own silhouette
  marks       a large figure drawn with tiny copies of its siblings
  catalogue   the family as a numbered specimen sheet
  registers   a page of friezes of one family, as horizontal bands
  collage     cut-paper silhouettes of the family under one headline
  emboss      six siblings blind-embossed on coloured stock
  diptych     the same motif positive and negative
  window      the motif's outline as a window onto its own feature, in a frieze border

Run:  python3 studies.py              # all → studies/<name>.html, .png, index.html
      python3 studies.py --only plate,emboss
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2
from skimage import color, measure

import friezes as FR
import posters as P

HERE = Path(__file__).resolve().parent
OUT = HERE / "studies"
ASSETS = OUT / "assets"
CLEAN = HERE / "vgg16motifs-clean"
MONO = "'SF Mono', Menlo, monospace"
SANS = "'Helvetica Neue', Helvetica, sans-serif"


def say(m: str) -> None:
    print(m, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Sources
# ─────────────────────────────────────────────────────────────────────────────

def palette(key: str, k: int = 6) -> list[str]:
    """Six colours of a convnext feature (k-means in Lab on its pixels), dark → light."""
    f = next(iter(sorted((HERE / "convnext_basefeatures").glob(f"convnext_base__{key}__*.png"))))
    a = np.asarray(Image.open(f).convert("RGB").resize((112, 112))).reshape(-1, 3) / 255
    lab = color.rgb2lab(a[None])[0]
    c, _ = kmeans2(lab, k, minit="++", seed=1)
    c = c[np.argsort(c[:, 0])]
    rgb = np.clip(color.lab2rgb(c[None])[0], 0, 1)
    return ["#" + "".join(f"{int(round(v * 255)):02x}" for v in x) for x in rgb]


def vec(ink: np.ndarray, tol: float = 0.8, s: float = 1.0) -> str:
    """A boolean image as one even-odd SVG path (coordinates × s)."""
    rings = []
    for c in measure.find_contours(np.pad(ink, 1).astype(np.float32), 0.5):
        c = measure.approximate_polygon(c - 1, tol)
        if len(c) >= 3:
            rings.append("M" + "L".join(f"{x * s:.1f},{y * s:.1f}" for y, x in c) + "Z")
    return "".join(rings)


def family(key: str) -> list[dict]:
    """A motif and its siblings: ink and body arrays, their paths, and the record."""
    j = next(iter(sorted(CLEAN.glob(f"*{key}__*/*/motif.json"))))
    rec = json.loads(j.read_text())
    d = j.parent
    files = [(d / "motif.png", d / "mask.png", {"score": 1.0, "shift": [0, 0]})]
    files += [(d / s["file"], d / s["file"].replace(".png", ".mask.png"), s) for s in rec.get("siblings", [])]
    out = []
    for f, m, meta in files:
        if not f.exists():
            continue
        ink = np.asarray(Image.open(f).convert("L")) < 128
        body = np.asarray(Image.open(m).convert("L")) > 127
        if body.shape != ink.shape:
            body = ndi.zoom(body.astype(np.float32), (ink.shape[0] / body.shape[0], ink.shape[1] / body.shape[1]), order=0) > 0.5
        h, w = ink.shape
        out.append({"ink": ink, "body": body, "w": w, "h": h, "d": vec(ink), "mask": vec(body, 1.5),
                    "meta": meta, "rec": rec})
    return out


def short(rec: dict) -> tuple[str, str, str]:
    m = re.search(r"(\w+?)__(.+?)__c(\d+)", rec["feature"])
    return m.group(1), m.group(2), m.group(3)


def save(a: np.ndarray, name: str) -> str:
    ASSETS.mkdir(parents=True, exist_ok=True)
    if a.dtype == bool:
        Image.fromarray(a).convert("1").save(ASSETS / name)
    else:
        Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8) if a.dtype != np.uint8 else a).save(ASSETS / name)
    return f"assets/{name}"


def use(d: str, x: float, y: float, s: float, fill: str, fx: int = 1, fy: int = 1, rot: float = 0, w: float = 0, h: float = 0,
        rule: str = "evenodd", extra: str = "") -> str:
    """A path placed with its centre at (x, y), scaled by s, flipped and turned."""
    return (f'<path fill="{fill}" fill-rule="{rule}" {extra} transform="translate({x:.1f},{y:.1f}) rotate({rot:.1f}) '
            f'scale({fx * s:.4f},{fy * s:.4f}) translate({-w / 2:.1f},{-h / 2:.1f})" d="{d}"/>')


def page(title: str, W: int, H: int, css: str, body: str, bg: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
html,body{{margin:0;background:{bg}}}
.c{{position:relative;width:{W}px;height:{H}px;overflow:hidden;background:{bg}}}
.c>*{{position:absolute}}
svg.full{{left:0;top:0}}
{css}
</style></head><body><div class="c">{body}</div></body></html>"""


def shoot(h: Path, png: Path, W: int, H: int) -> None:
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([P.CHROME, "--headless=new", f"--user-data-dir={prof}", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", f"--window-size={W},{H}", f"--screenshot={png}",
                        "--virtual-time-budget=4000", h.as_uri()], check=True, capture_output=True, timeout=120)


# ─────────────────────────────────────────────────────────────────────────────
#  The studies. Each returns (title, W, H, css, body, background)
# ─────────────────────────────────────────────────────────────────────────────

def study_plate():
    key = "features.24__c119"
    fam = family(key)
    m = fam[0]
    model, layer, ch = short(m["rec"])
    W, H = 1200, 1600
    ink, paper, accent = "#111111", "#ffffff", palette("features.5.11__c273")[4]
    s = 760 / max(m["w"], m["h"])
    X0, Y0 = 70, 250
    big = m["ink"]
    src = save(~P.enlarge(big, s), "plate-big.png")
    bw, bh = int(m["w"] * s), int(m["h"] * s)
    # callout points: busiest places in the drawing, spread apart
    inner = ndi.binary_erosion(m["body"], iterations=60)            # away from the rim
    edges = np.abs(np.diff(big.astype(np.int8), axis=0, prepend=0)) + np.abs(np.diff(big.astype(np.int8), axis=1, prepend=0))
    dens = ndi.uniform_filter(edges.astype(np.float32), 61) * inner   # where the drawing is busiest
    pts = []
    for _ in range(4):
        y, x = np.unravel_index(int(np.argmax(dens)), dens.shape)
        pts.append((x, y))
        yy, xx = np.ogrid[:dens.shape[0], :dens.shape[1]]
        dens[(yy - y) ** 2 + (xx - x) ** 2 < (0.22 * max(m["w"], m["h"])) ** 2] = 0
    pts.sort(key=lambda p: p[1])
    box, crop_r = 250, 70                              # inset size on the page; half-size of the crop in motif px
    svg, insets = [], []
    for i, (x, y) in enumerate(pts):
        sub = np.pad(big, crop_r)[y:y + 2 * crop_r, x:x + 2 * crop_r]
        insets.append(save(~P.enlarge(sub, box / (2 * crop_r)), f"plate-inset{i}.png"))
        bx, by = 900, 250 + i * (box + 70)
        px, py = X0 + x * s, Y0 + y * s
        r = crop_r * s
        mag = box / (2 * crop_r * s)
        svg.append(f'<rect x="{px - r:.1f}" y="{py - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" fill="none" stroke="{ink}" stroke-width="1"/>')
        ex = bx - 30 - 22 * (3 - i)                     # each leader turns at its own x, so they never share a run
        svg.append(f'<polyline points="{px + r:.1f},{py:.1f} {ex},{py:.1f} {ex},{by + box / 2} {bx},{by + box / 2}" fill="none" stroke="{ink}" stroke-width="0.8"/>')
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{accent}"/>')
        svg.append(f'<rect x="{bx}" y="{by}" width="{box}" height="{box}" fill="none" stroke="{ink}" stroke-width="1"/>')
        svg.append(f'<image href="{insets[-1]}" x="{bx}" y="{by}" width="{box}" height="{box}"/>')
        svg.append(f'<text x="{bx}" y="{by + box + 18}" class="m">{i + 1:02d}  ×{mag:.1f}   x {x:04d}  y {y:04d}</text>')
        svg.append(f'<text x="{px - r:.1f}" y="{py - r - 6:.1f}" class="m a">{i + 1:02d}</text>')
    body = f"""
<svg class="full" width="{W}" height="{H}"><image href="{src}" x="{X0}" y="{Y0}" width="{bw}" height="{bh}"/>{''.join(svg)}
<line x1="70" y1="190" x2="{W - 70}" y2="190" stroke="{ink}" stroke-width="1"/><line x1="70" y1="{H - 120}" x2="{W - 70}" y2="{H - 120}" stroke="{ink}" stroke-width="1"/></svg>
<div class="h">Plate {ch}</div>
<div class="k">{model} · {layer} · channel {ch}<br>print 1792 px · motif {m['w']}×{m['h']} px<br>{len(fam) - 1} siblings in the same print</div>
<div class="f">Four details of one drawing made by a network looking for this channel. The squares on the left mark where each detail was taken; the number beside each inset is its magnification.</div>
<div class="g">Grammar of Ornament</div>"""
    css = f"""
text.m{{font:11px {MONO};fill:{ink};letter-spacing:.5px}} text.a{{fill:{accent};font-weight:700}}
.h{{left:66px;top:62px;font:200 96px/1 {SANS};color:{ink};letter-spacing:-2px}}
.k{{right:70px;top:78px;text-align:right;font:12px/1.6 {MONO};color:{ink}}}
.f{{left:70px;bottom:52px;width:520px;font:14px/1.45 {SANS};color:{ink}}}
.g{{right:70px;bottom:52px;font:12px {MONO};color:{accent};letter-spacing:3px;text-transform:uppercase}}"""
    return "Plate", W, H, css, body, paper


def study_emblem():
    key = "features.24__c57"
    fam = family(key)
    pal = palette("features.5.2__c189")
    ground, ink, accent = pal[0], pal[5], pal[4]
    model, layer, ch = short(fam[0]["rec"])
    W, H = 1200, 1600
    parts = []
    # an upper pair and a lower pair, each a motif and a *different* sibling mirrored: nearly, never exactly, symmetric
    pairs = [(0, 1, 360, 430, 300, 24), (2, 3, 1240, 340, 250, -20)]   # (a, b, centre y, size, half gap, lean)
    for a, b, cy, size, gap, lean in pairs:
        for k, fx, cx, rot in ((a, 1, W / 2 - gap, lean), (b, -1, W / 2 + gap, -lean)):
            m = fam[k % len(fam)]
            s = size / max(m["w"], m["h"])
            parts.append(use(m["d"], cx, cy, s, ink, fx=fx, rot=rot, w=m["w"], h=m["h"]))
    arc = f'<path id="arc" d="M{W / 2 - 230},{700} A230,230 0 0 1 {W / 2 + 230},{700}" fill="none"/>'
    body = f"""<svg class="full" width="{W}" height="{H}"><defs>{arc}</defs>{''.join(parts)}
<text class="arc"><textPath href="#arc" startOffset="50%" text-anchor="middle">a motif and one of its siblings · drawn twice by the same channel · never quite the same twice</textPath></text></svg>
<div class="d">{ch}<span>{layer}</span></div>
<div class="s">{model}</div>"""
    css = f"""
text.arc{{font:13px {MONO};fill:{accent};letter-spacing:2px}}
.d{{left:0;width:{W}px;top:720px;text-align:center;font:400 250px/0.9 Didot;color:{ink};letter-spacing:-6px}}
.d span{{display:block;font:italic 40px Didot;letter-spacing:1px;color:{accent};margin-top:22px}}
.s{{left:0;width:{W}px;bottom:46px;text-align:center;font:12px {MONO};color:{accent};letter-spacing:6px;text-transform:uppercase}}"""
    return "Emblem", W, H, css, body, ground


def study_kaleido():
    key = "features.24__c27"
    fam = family(key)
    model, layer, ch = short(fam[0]["rec"])
    W, H = 1200, 1600
    ink, paper = "#0d0d0d", "#f2f1ec"
    cx, cy, hw, hh = W / 2, 470, 540, 400
    tris = [((cx, cy), (cx - hw, cy), (cx, cy - hh)), ((cx, cy), (cx + hw, cy), (cx, cy - hh)),
            ((cx, cy), (cx - hw, cy), (cx, cy + hh)), ((cx, cy), (cx + hw, cy), (cx, cy + hh))]
    flips = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    defs, parts = [], []
    for i, (t, (fx, fy)) in enumerate(zip(tris, flips)):
        pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in t)
        defs.append(f'<clipPath id="t{i}"><polygon points="{pts}"/></clipPath>')
        dark = i in (0, 3)
        parts.append(f'<polygon points="{pts}" fill="{ink if dark else paper}"/>')
        m = fam[(i + 1) % len(fam)]
        gx, gy = (t[0][0] + t[1][0] + t[2][0]) / 3, (t[0][1] + t[1][1] + t[2][1]) / 3
        s = 0.40 * hw / max(m["w"], m["h"])
        parts.append(f'<g clip-path="url(#t{i})">' + use(m["d"], gx, gy, s, paper if dark else ink, fx=fx, fy=fy, w=m["w"], h=m["h"]) + "</g>")
    edge = f"{cx - hw},{cy} {cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh}"
    body = f"""<svg class="full" width="{W}" height="{H}"><defs>{''.join(defs)}</defs>{''.join(parts)}
<polygon points="{edge}" fill="none" stroke="{ink}" stroke-width="2"/><line x1="{cx - hw}" y1="{cy}" x2="{cx + hw}" y2="{cy}" stroke="{ink}" stroke-width="1"/></svg>
<div class="t">Four Siblings</div><div class="u">{model} · {layer} · c{ch}</div>"""
    css = f"""
.t{{left:0;width:{W}px;top:1180px;text-align:center;font:500 22px/1 {SANS};letter-spacing:8px;text-transform:uppercase;color:{ink}}}
.u{{left:0;width:{W}px;top:1214px;text-align:center;font:11px {MONO};letter-spacing:3px;color:{ink}}}"""
    return "Kaleidoscope", W, H, css, body, paper


def study_invaded():
    key = "features.21__c70"
    fam = family(key)
    model, layer, ch = short(fam[0]["rec"])
    W, H = 1200, 1600
    rng = random.Random(4)
    parts = []
    for k, (x, y, sz, rot) in enumerate([(380, 520, 620, -12), (900, 1020, 700, 18), (260, 1280, 380, 160), (980, 300, 340, 70)]):
        m = fam[k % len(fam)]
        parts.append(use(m["d"], x, y, sz / max(m["w"], m["h"]), "#fff", rot=rot, fx=rng.choice([1, -1]), w=m["w"], h=m["h"]))
    cells = "".join(f'<div class="cell" style="left:{60 + i * 270}px">{t}</div>' for i, t in enumerate(
        [f"{model}<br>{layer}", f"channel {ch}", f"{len(fam)} drawings<br>of one form", "black / white"]))
    body = f"""<div class="half"></div>
<div class="w">VGG<br>{ch}</div>
<svg class="full ink" width="{W}" height="{H}">{''.join(parts)}</svg>
<div class="grid">{cells}</div>"""
    css = f"""
.half{{left:{W // 2}px;top:0;width:{W // 2}px;height:{H}px;background:#000}}
.w{{left:40px;top:120px;font:700 430px/0.86 {SANS};letter-spacing:-22px;color:#fff;mix-blend-mode:difference}}
svg.ink{{mix-blend-mode:difference}}
.grid{{left:0;top:{H - 170}px;width:{W}px;height:110px;mix-blend-mode:difference}}
.cell{{top:0;width:250px;height:100px;border-top:1px solid #fff;padding-top:10px;font:13px/1.45 {MONO};color:#fff;position:absolute}}"""
    return "Invaded", W, H, css, body, "#fff"


def study_outline():
    key = "features.24__c11"
    fam = family(key)
    m = fam[0]
    model, layer, ch = short(m["rec"])
    pal = palette("features.5.11__c273")
    ground, ink, fill, accent = pal[5], pal[0], pal[3], pal[2]
    W, H = 1200, 1600
    s = 820 / max(m["w"], m["h"])
    X0, Y0 = (W - m["w"] * s) / 2, 300
    body_ = ndi.binary_dilation(m["body"], iterations=int(14 / s) + 1)
    rings = []
    for k, off in enumerate((0, 46)):
        g = ndi.binary_dilation(body_, iterations=int(off / s)) if off else body_
        c = max(measure.find_contours(np.pad(g, 2).astype(np.float32), 0.5), key=len) - 2
        c = np.stack([ndi.gaussian_filter1d(c[:, 0], 6, mode="wrap"), ndi.gaussian_filter1d(c[:, 1], 6, mode="wrap")], 1)[::3]
        xs, ys = X0 + c[:, 1] * s, Y0 + c[:, 0] * s
        if np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys) < 0:       # clockwise on screen, so the text sits outside
            xs, ys = xs[::-1], ys[::-1]
        rings.append(f'<path id="o{k}" d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)) + 'Z" fill="none"/>')
    phrase = f"{model} · {layer} · channel {ch} · the line a network draws when it looks for one thing · "
    texts = "".join(f'<text class="o{k}"><textPath href="#o{k}">{html.escape(phrase * 6)}</textPath></text>' for k in range(2))
    figure = (f'<path fill="{fill}" transform="translate({X0:.1f},{Y0:.1f}) scale({s:.4f})" d="{m["mask"]}"/>'
              f'<path fill="{ink}" fill-rule="evenodd" transform="translate({X0:.1f},{Y0:.1f}) scale({s:.4f})" d="{m["d"]}"/>')
    body = f"""<svg class="full" width="{W}" height="{H}"><defs>{''.join(rings)}</defs>{figure}{texts}</svg>
<div class="t">Outline</div><div class="n">c{ch}</div>"""
    css = f"""
text.o0{{font:15px {MONO};fill:{ink};letter-spacing:1px}} text.o1{{font:15px {MONO};fill:{accent};letter-spacing:1px}} text.o2{{font:15px {MONO};fill:{ink};opacity:.55;letter-spacing:1px}}
.t{{left:70px;top:70px;font:300 64px {SANS};color:{ink};letter-spacing:-1px}}
.n{{right:70px;bottom:60px;font:200 150px/1 {SANS};color:{accent};letter-spacing:-6px}}"""
    return "Outline", W, H, css, body, ground


def study_marks():
    key = "features.24__c119"
    fam = family(key)
    m = fam[0]
    model, layer, ch = short(m["rec"])
    W, H = 1200, 1600
    ink, paper = "#101010", "#f4f3ef"
    s = 1000 / max(m["w"], m["h"])
    X0, Y0 = (W - m["w"] * s) / 2, 260
    tone = ndi.gaussian_filter(m["ink"].astype(np.float32), 2) * m["body"]
    parts, placed = [], []
    rng = random.Random(2)
    nrng = np.random.default_rng(2)
    ys, xs = np.nonzero(tone > 0.45)                 # only where the drawing is ink
    cell, grid = 30.0, {}                             # spatial hash of placed copies
    for t in nrng.permutation(len(ys))[:80000]:          # dart throwing: no grid, no pixels
        y, x = ys[t], xs[t]
        v = tone[y, x]
        size = 12 + 16 * v
        gx, gy = X0 + x * s, Y0 + y * s
        ci, cj = int(gx // cell), int(gy // cell)
        near = [q for di in (-1, 0, 1) for dj in (-1, 0, 1) for q in grid.get((ci + di, cj + dj), [])]
        if any((gx - px) ** 2 + (gy - py) ** 2 < (0.45 * (size + ps)) ** 2 for px, py, ps in near):
            continue
        grid.setdefault((ci, cj), []).append((gx, gy, size))
        placed.append((gx, gy, size))
        gi = rng.randrange(1, len(fam)) if len(fam) > 1 else 0
        g = fam[gi]
        sc = size / max(g["w"], g["h"])
        parts.append(f'<use href="#s{gi}" transform="translate({gx:.1f},{gy:.1f}) rotate({rng.uniform(-180, 180):.0f}) '
                     f'scale({sc:.4f}) translate({-g["w"] / 2:.0f},{-g["h"] / 2:.0f})"/>')
    defs = "".join(f'<path id="s{i}" fill="{ink}" fill-rule="evenodd" d="{vec(g["ink"], 2.5)}"/>' for i, g in enumerate(fam))
    parts.insert(0, f"<defs>{defs}</defs>")
    body = f"""<svg class="full" width="{W}" height="{H}">{''.join(parts)}</svg>
<div class="t">Made of itself</div><div class="u">One drawing of {layer} c{ch}, set in {len(parts)} small copies of its own siblings. Each copy is as large as the drawing is dark where it stands.</div>"""
    css = f"""
.t{{left:70px;top:80px;font:500 15px {SANS};letter-spacing:6px;text-transform:uppercase;color:{ink}}}
.u{{left:70px;bottom:70px;width:560px;font:13px/1.5 {MONO};color:{ink}}}"""
    return "Marks", W, H, css, body, paper


def study_catalogue():
    key = "features.19__c73"
    fam = family(key)
    model, layer, ch = short(fam[0]["rec"])
    pal = palette("features.5.18__c21")
    paper, ink, accent = pal[5], pal[0], pal[2]
    W, H = 1200, 1600
    cols, cw, chh, x0, y0 = 2, 520, 300, 70, 230
    cells = []
    for i, m in enumerate(fam[:7]):
        cx, cy = x0 + (i % cols) * (cw + 20), y0 + (i // cols) * (chh + 20)
        s = min((cw - 160) / m["w"], (chh - 50) / m["h"])
        meta = m["meta"]
        label = "the motif as cut" if i == 0 else f"ncc {meta['score']:.3f}"
        sh = meta.get("shift", [0, 0])
        cells.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{chh}" fill="none" stroke="{ink}" stroke-width="0.8"/>'
                     + use(m["d"], cx + 150 + (cw - 150) / 2, cy + chh / 2, s, ink, w=m["w"], h=m["h"])
                     + f'<text x="{cx + 16}" y="{cy + 64}" class="n">{i:02d}</text>'
                     + f'<text x="{cx + 18}" y="{cy + chh - 42}" class="m">{label}</text>'
                     + f'<text x="{cx + 18}" y="{cy + chh - 24}" class="m">dx {sh[0]:+05d}  dy {sh[1]:+05d}</text>')
    i = 7
    tx, ty = x0 + (i % cols) * (cw + 20), y0 + (i // cols) * (chh + 20)
    body = f"""<svg class="full" width="{W}" height="{H}">{''.join(cells)}</svg>
<div class="h">Catalogue of one form</div><div class="k">{model} · {layer} · c{ch}</div>
<div class="x" style="left:{tx + 16}px;top:{ty + 16}px;width:{cw - 32}px">The first drawing was cut by hand; the others are the same form found again elsewhere in the same print, wherever the network drew it a second, third or seventh time. ncc: how closely each matches the first. dx, dy: where it was found, in print pixels.</div>"""
    css = f"""
text.n{{font:200 54px {SANS};fill:{accent};letter-spacing:-2px}} text.m{{font:11px {MONO};fill:{ink}}}
.h{{left:70px;top:90px;font:300 54px {SANS};color:{ink};letter-spacing:-1px}}
.k{{right:70px;top:112px;font:12px {MONO};color:{accent};letter-spacing:2px}}
.x{{font:13px/1.55 {MONO};color:{ink}}}"""
    return "Catalogue", W, H, css, body, paper


def study_registers():
    key = "features.24__c113"
    fams = FR.motif_families()
    fam = next(f for f in fams if f"{key}__" in f["id"])
    model, layer, ch = re.search(r"(\w+?)__(.+?)__c(\d+)", fam["id"]).groups()
    pal = palette("features.5.24__c249")
    W, H = 1200, 1600
    bands = [  # (arrangement, height, ground, ink)
        ("shift", 150, pal[5], pal[0]), ("stack", 260, pal[3], pal[0]), ("turn", 120, pal[0], pal[4]),
        ("pulse", 210, pal[4], pal[0]), ("glide", 150, pal[1], pal[5]), ("mirror", 300, pal[5], pal[2]),
    ]
    gut, y, out, notes = 46, 90, [], []
    for i, (arr, h, g, k) in enumerate(bands):
        svg, _ = FR.frieze(fam, random.Random(i), {"name": "r", "paper": g, "ink": k, "ink2": k, "deep": g, "light": k},
                           None, arr=arr, scheme="ink", rails="none")
        (ASSETS / f"reg{i}.svg").write_text(svg)
        out.append(f'<img src="assets/reg{i}.svg" style="left:0;top:{y}px;width:{W}px;height:{h}px;object-fit:cover">')
        notes.append(f'<div class="g" style="top:{y + h + 14}px">{i + 1:02d} &nbsp; {arr:<7} &nbsp; h {h}</div>')
        y += h + gut
    grid = "".join(f'<div class="v" style="left:{70 + i * 106}px"></div>' for i in range(11))
    body = "".join(out) + "".join(notes) + grid + f"""<div class="t">Registers</div><div class="k">{model} · {layer} · c{ch} · six arrangements of one family</div>"""
    css = f"""
.g{{left:70px;font:11px {MONO};color:{pal[0]};letter-spacing:2px;white-space:pre}}
.v{{top:0;width:1px;height:{H}px;background:{pal[0]};opacity:.18}}
.t{{left:70px;top:26px;font:500 14px {SANS};letter-spacing:8px;text-transform:uppercase;color:{pal[0]}}}
.k{{right:70px;top:28px;font:11px {MONO};color:{pal[0]}}}"""
    return "Registers", W, H, css, body, pal[5]


def study_collage():
    key = "features.21__c40"
    fam = family(key)
    model, layer, ch = short(fam[0]["rec"])
    pal = palette("features.5.0__c451")
    W, H = 1200, 1600
    paper, ink = "#f1eee6", pal[0]
    fills = [pal[2], pal[3], pal[4], pal[5], pal[1]]
    rng = random.Random(11)
    parts = []
    for i in range(22):
        m = fam[i % len(fam)]
        sz = rng.uniform(170, 420)
        while True:                                    # the headline's band (y 580–880) stays clear
            x, y = rng.uniform(-60, W + 60), rng.uniform(-60, H + 60)
            if y + sz / 2 < 570 or y - sz / 2 > 890:
                break
        s, rot, fx = sz / max(m["w"], m["h"]), rng.uniform(-180, 180), rng.choice([1, -1])
        parts.append(use(m["mask"], x, y, s, rng.choice(fills), fx=fx, rot=rot, w=m["w"], h=m["h"], rule="nonzero",
                         extra='style="filter:drop-shadow(0 2px 2px rgba(0,0,0,.18))"'))
        parts.append(use(m["d"], x, y, s, ink, fx=fx, rot=rot, w=m["w"], h=m["h"]))
    body = f"""<svg class="full" width="{W}" height="{H}">{''.join(parts)}</svg>
<div class="t">Twenty-two cuts<br>from one print</div><div class="k">{model} · {layer} · channel {ch}</div>"""
    css = f"""
.t{{left:60px;top:600px;font:700 118px/0.92 {SANS};color:{ink};letter-spacing:-5px}}
.k{{left:66px;top:830px;font:13px {MONO};color:{ink};letter-spacing:2px;background:{paper};padding:4px 8px}}"""
    return "Collage", W, H, css, body, paper


def study_emboss():
    key = "features.24__c66"
    fam = family(key)
    model, layer, ch = short(fam[0]["rec"])
    pal = palette("features.3.1__c228")
    stock = P.hexrgb(pal[4])
    W, H = 1200, 1600
    hmap = np.zeros((H, W), np.float32)
    for i, m in enumerate(fam[:6]):
        cx, cy = 290 + (i % 3) * 310, 560 + (i // 3) * 420
        s = 250 / max(m["w"], m["h"])
        b, k = P.enlarge(m["body"], s), P.enlarge(m["ink"], s)
        y0, x0 = int(cy - b.shape[0] / 2), int(cx - b.shape[1] / 2)
        hb, hk = P.place((H, W), b, y0, x0).astype(np.float32), P.place((H, W), k, y0, x0).astype(np.float32)
        hmap += ndi.gaussian_filter(hb, 3.0) - 0.8 * ndi.gaussian_filter(hk * hb, 1.0)
    shade = P.emboss(hmap * 11, 1.0)
    grain = np.random.default_rng(0).normal(0, 0.006, (H, W))
    img = stock * (shade + grain)[..., None]
    src = save(img.astype(np.float32), "emboss.png")
    body = f"""<img src="{src}" style="left:0;top:0;width:{W}px;height:{H}px"><div class="k">{layer} · c{ch}</div>"""
    css = f".k{{left:0;width:{W}px;bottom:80px;text-align:center;font:12px {MONO};letter-spacing:4px;color:{pal[0]}}}"
    return "Emboss", W, H, css, body, pal[4]


def study_diptych():
    key = "features.24__c154"
    fam = family(key)
    m = fam[0]
    model, layer, ch = short(m["rec"])
    W, H = 1600, 1000
    s = 640 / max(m["w"], m["h"])
    parts = [f'<rect x="{W // 2}" y="0" width="{W // 2}" height="{H}" fill="#000"/>',
             use(m["d"], W / 4, H / 2 - 20, s, "#000", w=m["w"], h=m["h"]),
             use(m["mask"], 3 * W / 4, H / 2 - 20, s, "#fff", w=m["w"], h=m["h"], rule="nonzero"),
             use(m["d"], 3 * W / 4, H / 2 - 20, s, "#000", w=m["w"], h=m["h"])]
    # on the right, the ground of the drawing is black and its ink white: paint the body white, then cut the ink back to black
    parts = parts[:2] + [use(m["d"], 3 * W / 4, H / 2 - 20, s, "#fff", w=m["w"], h=m["h"])]
    body = f"""<svg class="full" width="{W}" height="{H}">{''.join(parts)}</svg>
<div class="l">positive</div><div class="r">negative</div><div class="k">{model} · {layer} · c{ch}</div>"""
    css = f"""
.l{{left:60px;bottom:50px;font:11px {MONO};letter-spacing:4px;color:#000}}
.r{{right:60px;bottom:50px;font:11px {MONO};letter-spacing:4px;color:#fff}}
.k{{left:0;width:{W}px;top:50px;text-align:center;font:11px {MONO};letter-spacing:4px;color:#888}}"""
    return "Diptych", W, H, css, body, "#fff"


def study_window():
    key = "features.24__c86"
    fam = family(key)
    m = fam[0]
    model, layer, ch = short(m["rec"])
    pal = palette("features.5.13__c432")
    W, H = 1200, 1600
    feat = P.big(f"vgg16:{layer}__c{ch}")
    lum = P.stretch(P.lum(feat))
    duo = P.ramp(lum, pal[0], pal[2], pal[4], pal[5])          # the feature in the page's own colours
    s = 900 / max(m["w"], m["h"])
    b = P.enlarge(m["body"], s)
    y0, x0 = 330, int((W - b.shape[1]) / 2)
    mask = P.place((H, W), b, y0, x0)
    pic = P.tiled(duo, H, W, scale=1.6)
    img = np.ones((H, W, 3), np.float32) * P.hexrgb(pal[5])
    img[mask] = pic[mask]
    rim = mask & ~ndi.binary_erosion(mask, iterations=4)
    img[rim] = P.hexrgb(pal[0])
    src = save(img, "window.png")
    fr = FR.motif_families()
    ffam = next(f for f in fr if f"{key}__" in f["id"])
    svg, _ = FR.frieze(ffam, random.Random(5), {"name": "w", "paper": pal[5], "ink": pal[0], "ink2": pal[2], "deep": pal[5], "light": pal[0]},
                       None, arr="turn", scheme="two", rails="none")
    (ASSETS / "window-frieze.svg").write_text(svg)
    body = f"""<img src="{src}" style="left:0;top:0;width:{W}px;height:{H}px">
<img src="assets/window-frieze.svg" style="left:0;top:90px;width:{W}px;height:105px;object-fit:cover">
<img src="assets/window-frieze.svg" style="left:0;top:{H - 195}px;width:{W}px;height:105px;object-fit:cover;transform:scaleY(-1)">
<div class="k">{model} · {layer} · channel {ch} — the outline of one drawing, opened onto the feature it came from</div>"""
    css = f".k{{left:70px;top:{H - 70}px;font:11px {MONO};letter-spacing:2px;color:{pal[0]}}}"
    return "Window", W, H, css, body, pal[5]


STUDIES = {"plate": study_plate, "emblem": study_emblem, "kaleido": study_kaleido, "invaded": study_invaded,
           "outline": study_outline, "marks": study_marks, "catalogue": study_catalogue, "registers": study_registers,
           "collage": study_collage, "emboss": study_emboss, "diptych": study_diptych, "window": study_window}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(STUDIES))
    a = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in a.only.split(","):
        say(f"{name} …")
        title, W, H, css, body, bg = STUDIES[name]()
        h = OUT / f"{name}.html"
        h.write_text(page(title, W, H, css, body, bg))
        shoot(h, OUT / f"{name}.png", W, H)
        say(f"  wrote {h.name} and {name}.png")
    figs = "".join(f'<figure><img src="{n}.png" loading="lazy"><figcaption>{n}</figcaption></figure>' for n in STUDIES if (OUT / f"{n}.png").exists())
    (OUT / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Layout Studies</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>body{{margin:0;background:#cfcdc7;font:13px {MONO}}}
h1{{font:300 28px {SANS};margin:28px 24px 4px}}p{{margin:0 24px 14px}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,360px),1fr));gap:26px;padding:10px 24px 60px}}
figure{{margin:0}}img{{width:100%;display:block;box-shadow:0 4px 16px #0003}}figcaption{{margin-top:6px}}</style></head><body>
<h1>Layout studies</h1><p>After notes/design_references.md. One source per study; black and white or a convnext palette.</p><main>{figs}</main></body></html>""")


if __name__ == "__main__":
    main()
