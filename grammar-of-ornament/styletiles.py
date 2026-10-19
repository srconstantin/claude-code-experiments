#!/usr/bin/env python3
"""
Grammar of Ornament — style tiles: three directions for a web component
library built from the ornaments in ornament.py, one page each with the same
components, so they can be compared side by side.

  woodcut     chipped slabs, relief gouges and bands; black on paper, one hot accent
  engraving   cartouches and swallowtail ribbons, fine combs, feathers and
              hatching; violet ink on cream, ochre accent

Shapes are not rounded rectangles: designlib/tiles/shapes.js draws each
button, card, badge and alert as an SVG fitted to the element's real size
(so points and curves never stretch). Ornaments (friezes, rosettes, glyphs)
are named recipes from ornament.py with parameters, so the server can
regenerate any of them as a variant.

Every component and every ornament on a page is a specimen: hovering it shows
"↻ variant" (a new seed and slightly jittered proportions for its ornaments
and shapes), "Save" (writes a standalone specimen.html, meta.json and the
ornament SVGs into designlib/saved/<id>/, when the page is served by
server.py) and, for ornaments, "SVG" (download). The page's own buttons work:
tabs switch, "Save pattern" saves its specimen, "Export SVG" downloads the
divider ornament, "Show layers" outlines the drawn shapes.

Legibility: text always sits on solid ground; ornament lives at edges, under
and beside controls, never under words; every text and background pair is
checked against WCAG contrast (accents for focus rings and large text are
adjusted until they pass) and the check is listed at the foot of each page.

Run:  python3 styletiles.py                # → designlib/tiles/*.html
      python3 styletiles.py --screenshot   # also PNGs via headless Chrome
Then open http://127.0.0.1:8790/designlib/tiles/ with server.py running.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color

import ornament as O

HERE = Path(__file__).resolve().parent
OUT = HERE / "designlib" / "tiles"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# ─────────────────────────────────────────────────────────────────────────────
#  Colour
# ─────────────────────────────────────────────────────────────────────────────

def hex_to_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in np.clip(rgb, 0, 1))


def lab_hex(L: float, a: float, b: float) -> str:
    lab = np.array([L, a, b], float)
    for _ in range(40):
        rgb = color.lab2rgb(lab.reshape(1, 1, 3))[0, 0]
        if np.all(rgb >= -0.001) and np.all(rgb <= 1.001):
            break
        lab[1:] *= 0.93
    return rgb_to_hex(color.lab2rgb(lab.reshape(1, 1, 3))[0, 0])


def to_lab(h: str) -> np.ndarray:
    return color.rgb2lab(hex_to_rgb(h).reshape(1, 1, 3))[0, 0]


def luminance(h: str) -> float:
    c = hex_to_rgb(h)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def push_to(h: str, ground: str, ratio: float) -> str:
    """Move a colour's lightness away from the ground until it reaches the ratio."""
    lab = to_lab(h)
    step = -1 if luminance(ground) > 0.18 else 1
    while contrast(h, ground) < ratio and 2 < lab[0] < 98:
        lab[0] += step
        h = lab_hex(*lab)
    return h


def pair_ends(summary: Path) -> dict:
    """Median Lab colour of each end of each named hue pair, from colorways."""
    groups = {}
    with open(summary) as fh:
        for r in csv.DictReader(fh):
            key = tuple(sorted((r["name_a"], r["name_b"])))
            ends = {r["name_a"]: r["a"], r["name_b"]: r["b"]}
            g = groups.setdefault(key, {k: [] for k in key})
            for k in key:
                g[k].append(to_lab(ends[k]))
    return {key: {k: np.median(v, axis=0) for k, v in g.items()} for key, g in groups.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Themes
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def themes() -> dict:
    pairs = pair_ends(HERE / "colorways" / "convnext_basefeatures" / "summary.csv")
    by = lambda a, b: pairs[tuple(sorted((a, b)))]
    py, bo, byl = by("purple", "yellow"), by("blue", "orange"), by("blue", "yellow")
    T = {}
    T["woodcut"] = dict(
        title="Woodcut", blurb="Chipped slabs, relief gouges and bands, cut in black on paper with one hot accent.",
        ink="#16130f", paper="#f2ebdd", accent=lab_hex(52, bo["orange"][1] * 1.25, bo["orange"][2] * 1.1),
        card_bg="#fbf7ee", field_bg="#fbf7ee", muted="#5b544a", danger=lab_hex(42, 62, 44),
        head="'Fraunces', Georgia, serif", head_weight="900", body="'Alegreya Sans', 'Helvetica Neue', sans-serif",
        ui="'Alegreya Sans', 'Helvetica Neue', sans-serif", fonts="Fraunces:opsz,wght@9..144,700;9..144,900&family=Alegreya+Sans:wght@400;500;700;800",
        line="3px", hair="2px", pair=("blue", "orange"),
        shapes=dict(button="slab", card="slabTorn", badge="slab", alert="slab", tab="slab", hero=None))
    T["engraving"] = dict(
        title="Engraving", blurb="Cartouches and swallowtail ribbons, fine combs, feathers and hatching, in violet ink on cream with ochre.",
        ink=lab_hex(20, py["purple"][1] * 0.9, py["purple"][2] * 0.9), paper=lab_hex(95.5, py["yellow"][1] * 0.1, py["yellow"][2] * 0.16),
        accent=lab_hex(64, py["yellow"][1] * 1.1, py["yellow"][2] * 1.05), card_bg=lab_hex(97.5, 0, 4), field_bg=lab_hex(97.5, 0, 4),
        muted=lab_hex(42, py["purple"][1] * 0.3, py["purple"][2] * 0.3), danger=lab_hex(40, 55, 30),
        head="'Cormorant Garamond', Garamond, serif", head_weight="600", body="'EB Garamond', Garamond, serif",
        ui="'EB Garamond', Garamond, serif", fonts="Cormorant+Garamond:wght@500;600;700&family=EB+Garamond:wght@400;500;600",
        line="1.5px", hair="1px", pair=("purple", "yellow"),
        shapes=dict(button="cartouche", card="cartoucheCard", badge="ribbon", alert="flag", tab=None, hero="cartoucheCard"))
    for name, t in T.items():
        L = to_lab(t["ink"])[0]
        t["ink_hover"] = lab_hex(L + (12 if L < 50 else -14), *to_lab(t["ink"])[1:])
        t["focus"] = push_to(t["accent"], t["paper"], 3.2)
        if name == "woodcut":   # the hover button puts ink on the accent
            lab = to_lab(t["accent"])
            while contrast(t["ink"], t["accent"]) < 4.6 and lab[0] < 90:
                lab[0] += 1
                t["accent"] = lab_hex(*lab)
        t["on_accent"] = t["ink"] if contrast(t["ink"], t["accent"]) > contrast(t["paper"], t["accent"]) else t["paper"]
    return T


# ─────────────────────────────────────────────────────────────────────────────
#  Ornament specs: var → (recipe, parameters, colour, mode[, height])
#    mode  frieze   a seamless tile, repeated along a line
#          glyph    square, kept in proportion
#          stretch  stretched to its box (alert marks, gouges)
#          centre   a rosette on a disc of ground, to sit over a rule
# ─────────────────────────────────────────────────────────────────────────────

SPECS = {
    "woodcut": {
        "rule": ("band_frieze", dict(seed=5, width=5.8, amp=1.2), "ink", "frieze", "10px"),
        "underline": ("band_frieze", dict(seed=11, width=3.2, amp=0.7, n_gouges=7, gouge_len=0.6), "ink", "frieze", "8px"),
        "underline_focus": ("band_frieze", dict(seed=11, width=3.2, amp=0.7, n_gouges=7, gouge_len=0.6), "accent", "frieze", "8px"),
        "underline_error": ("comb_frieze", dict(seed=12, n_teeth=8, spine_w=2.4, tooth_len=4.0, tooth_w=1.6), "danger", "frieze", "11px"),
        "tab": ("band_frieze", dict(seed=13, width=4.0, amp=0.8, n_gouges=6), "ink", "frieze", "9px"),
        "divider": ("band_frieze", dict(seed=21), "ink", "frieze", "12px"),
        "divider_mark": ("rosette_petal", dict(n=6, r0=1.6, length=4.4, w=2.2, turn=0.5, taper0=0.8, taper1=0.1), "ink", "centre"),
        "check": ("check", dict(w=2.4), "paper", "glyph"),
        "ring": ("loop", dict(r=4.0, w=2.0, seed=2), "ink", "glyph"),
        "dot": ("loop", dict(r=1.6, w=2.6, seed=3), "ink", "glyph"),
        "knob": ("rosette_petal", dict(n=5, r0=0.6, length=3.0, w=1.6, turn=0.5, taper0=0.9, taper1=0.1), "paper", "glyph"),
        "caret": ("caret", dict(w=2.0), "ink", "glyph"),
        "spinner": ("rosette_petal", dict(n=8, r0=2.0, length=5.0, w=2.2, turn=0.6, taper0=0.9, taper1=0.08), "ink", "glyph"),
        "mark": ("band_mark", dict(seed=4, width=6.5, amp=0.4), "ink", "stretch"),
        "mark_error": ("band_mark", dict(seed=4, width=6.5, amp=0.4), "danger", "stretch"),
        "bullet": ("rosette_petal", dict(n=4, r0=0.5, length=2.4, w=1.5, turn=0.4, taper0=0.9, taper1=0.1), "ink", "glyph"),
        "ticks": ("comb_frieze", dict(seed=31, n_teeth=10, spine_w=0.01, tooth_len=3.0, tooth_w=1.4, angle_deg=80.0, amp=0.0), "ink", "frieze", "7px"),
        "gouge_l": ("gouges", dict(seed=7, n=3, height=12.0, length=6.0, w=2.0, side=1), "paper", "stretch"),
        "gouge_r": ("gouges", dict(seed=8, n=3, height=12.0, length=6.0, w=2.0, side=-1), "paper", "stretch"),
        "hero_rule": ("band_frieze", dict(seed=41), "ink", "frieze", "16px"),
    },
    "engraving": {
        "rule": ("hatch_frieze", dict(seed=3, w=0.7, n=14), "ink", "frieze", "9px"),
        "underline": ("feather_frieze", dict(seed=2, spine_w=0.9, tooth_w=0.6, tooth_len=4.0, spacing=3.0, n_teeth=10), "ink", "frieze", "9px"),
        "underline_focus": ("feather_frieze", dict(seed=2, spine_w=0.9, tooth_w=0.6, tooth_len=4.0, spacing=3.0, n_teeth=10), "focus", "frieze", "11px"),
        "underline_error": ("comb_frieze", dict(seed=12, spine_w=1.1, tooth_w=0.75, tooth_len=4.5), "danger", "frieze", "10px"),
        "fringe": ("comb_frieze", dict(seed=4, spine_w=1.1, tooth_w=0.75, tooth_len=6.5), "ink", "frieze", "11px"),
        "tab": ("comb_frieze", dict(seed=5, spine_w=1.1, tooth_w=0.75, tooth_len=5.0), "ink", "frieze", "10px"),
        "divider": ("feather_frieze", dict(seed=6, spine_w=0.9, tooth_w=0.6, tooth_len=4.0, spacing=3.0, n_teeth=10), "ink", "frieze", "13px"),
        "divider_mark": ("rosette_petal", dict(n=8, r0=1.4, length=4.8, w=0.9, turn=0.9, taper0=0.3, taper1=0.12), "ink", "centre"),
        "corner": ("rosette_petal", dict(n=4, r0=0.8, length=5.0, w=0.9, turn=0.9, taper0=0.3, taper1=0.12, mirror=True), "ink", "glyph"),
        "check": ("check", dict(w=1.3), "ink", "glyph"),
        "ring": ("loop", dict(r=4.0, w=0.9, seed=2), "ink", "glyph"),
        "dot": ("rosette_petal", dict(n=6, r0=0.2, length=2.2, w=0.9, turn=0.5, taper0=0.5, taper1=0.15), "ink", "glyph"),
        "knob": ("rosette_petal", dict(n=6, r0=0.4, length=2.8, w=0.7, turn=0.8, taper0=0.4, taper1=0.12), "ink", "glyph"),
        "caret": ("caret", dict(w=1.0), "ink", "glyph"),
        "spinner": ("rosette_hook", dict(n=6, r0=2.0, length=6.0, w=0.9), "ink", "glyph"),
        "mark": ("band_mark", dict(seed=4, width=3.0, amp=0.3), "ink", "stretch"),
        "mark_error": ("band_mark", dict(seed=4, width=3.0, amp=0.3), "danger", "stretch"),
        "bullet": ("rosette_petal", dict(n=4, r0=0.4, length=2.6, w=0.8, turn=0.7, taper0=0.4, taper1=0.12, mirror=True), "ink", "glyph"),
        "ticks": ("hatch_frieze", dict(seed=9, w=0.6, n=16, angle_deg=75.0), "ink", "frieze", "7px"),
        "hero_rule": ("feather_frieze", dict(seed=41, spine_w=0.9, tooth_w=0.6, tooth_len=4.0, spacing=3.0, n_teeth=10), "ink", "frieze", "15px"),
    }

}


def render_var(spec: tuple, t: dict, seed: int | None = None, jitter: float = 0.0) -> str:
    """One ornament as a CSS url() of an SVG data URI."""
    recipe, kw, colour, mode = spec[:4]
    ink = t[colour]
    out = O.make(recipe, kw, seed, jitter)
    if isinstance(out, tuple):           # friezes return (art, viewBox)
        art, vb = out
        svg = art.svg(vb, ink=ink)
    else:
        art = out
        if mode == "stretch":
            art, vb = O.fit(art, 0.3)
            svg = art.svg(vb, ink=ink)
        elif mode == "centre":
            lo, hi = art.bounds()
            r = float(max(np.abs(lo).max(), np.abs(hi).max())) + 1.6
            svg = art.svg((-r, -r, 2 * r, 2 * r), ink=ink).replace(
                'preserveAspectRatio="none">', f'preserveAspectRatio="xMidYMid meet"><circle r="{O.f1(r)}" fill="{t["paper"]}"/>', 1)
        else:
            art, (x0, y0, W, H) = O.fit(art, 0.6)
            s = max(W, H)
            svg = art.svg((x0 - (s - W) / 2, y0 - (s - H) / 2, s, s), ink=ink).replace('preserveAspectRatio="none"', 'preserveAspectRatio="xMidYMid meet"')
    return f'url("{O.data_uri(svg)}")'


def variant_vars(theme: str, names: list[str], seed: int) -> dict:
    """New versions of the named ornaments for a theme (used by server.py)."""
    t = themes()[theme]
    out = {}
    for i, n in enumerate(names):
        key = n.replace("-", "_")
        if key in SPECS[theme]:
            out[n] = render_var(SPECS[theme][key], t, seed=seed + 7919 * i, jitter=0.18)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Images
# ─────────────────────────────────────────────────────────────────────────────

def upscaled_feature(name: str, src: Path) -> Path:
    import motifs
    motifs.upscaled(src, name)
    return HERE / "cache" / "sr" / f"{name}.png"


def duotone(path: Path, shadow: str, highlight: str, out: Path) -> None:
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255
    lum = color.rgb2gray(rgb)
    lo, hi = np.percentile(lum, [2, 98])
    t = np.clip((lum - lo) / (hi - lo + 1e-9), 0, 1)
    s, h = hex_to_rgb(shadow), hex_to_rgb(highlight)
    Image.fromarray(((s + (h - s) * t[..., None]) * 255).round().astype(np.uint8)).save(out, optimize=True)


def ink_motif(path: Path, ink: str, out: Path) -> None:
    m = np.asarray(Image.open(path).convert("L")) < 128
    rgba = np.zeros(m.shape + (4,), np.uint8)
    rgba[..., :3] = (hex_to_rgb(ink) * 255).astype(np.uint8)
    rgba[..., 3] = m * 255
    Image.fromarray(rgba).save(out, optimize=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page
# ─────────────────────────────────────────────────────────────────────────────

CSS = r"""
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.6 var(--body);-webkit-font-smoothing:antialiased}
a{color:inherit}
.wrap{max-width:1120px;margin:0 auto;padding:0 32px 96px}
h1,h2,h3{font-family:var(--head);font-weight:var(--head-weight);line-height:1.08;margin:0}
.switch{display:flex;gap:18px;justify-content:flex-end;padding:14px 32px;font:500 14px var(--ui)}
.switch a{text-decoration:none;opacity:.7}.switch a.on{opacity:1;font-weight:700}

/* drawn shapes */
[data-shape]{position:relative;isolation:isolate;box-shadow:none!important;border-radius:0!important}
[data-shape]:not([data-clip]):not(.art){background:none!important}
[data-shape]>svg.shape{position:absolute;left:0;top:0;z-index:-1;overflow:visible;pointer-events:none}
svg.shape .fill{fill:var(--shape-fill,transparent);transition:fill .15s}
svg.shape .line{fill:none;stroke:var(--shape-line,transparent);stroke-width:var(--shape-stroke,0)}
svg.shape .inner{fill:none;stroke:var(--shape-line,transparent);stroke-width:var(--hair)}
svg.shape .focus{fill:none;stroke:var(--focus);stroke-width:2;display:none}
:focus-visible>svg.shape .focus,.is-focus>svg.shape .focus{display:block}
.show-layers svg.shape .fill{stroke:#e0306a;stroke-width:1;stroke-dasharray:3 3}

/* specimens */
.specimen{position:relative;display:flex;flex-direction:column;gap:12px;align-items:flex-start;padding:12px;margin:-12px;border-radius:6px;transition:box-shadow .15s}
.specimen:hover{box-shadow:0 0 0 1px color-mix(in srgb,var(--muted) 45%,transparent)}
.specimen>figcaption{font:13px var(--ui);color:var(--muted)}
.tools{position:absolute;top:-15px;right:4px;display:none;gap:4px;z-index:20}
.specimen:hover>.tools,.specimen:focus-within>.tools{display:flex}
.tools button{font:600 11px/1 system-ui,sans-serif;padding:6px 8px;border-radius:5px;border:1px solid #c9c4ba;background:#fff;color:#222;cursor:pointer}
.tools button:hover{background:#222;color:#fff}
.toast{position:fixed;left:50%;bottom:24px;transform:translate(-50%,30px);opacity:0;background:#222;color:#fff;font:14px system-ui;padding:10px 16px;border-radius:8px;transition:.25s;z-index:99}
.toast.on{opacity:1;transform:translate(-50%,0)}

/* hero */
.hero{display:grid;grid-template-columns:1.05fr 1fr;gap:24px;min-height:460px;max-width:1184px;margin:0 auto;padding:0 32px 0 0}
.hero .panel{padding:56px 40px 48px 32px;display:flex;flex-direction:column;gap:20px;justify-content:center}
.hero .eyebrow{font:600 13px var(--ui);letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.hero h1{font-size:clamp(40px,5vw,62px)}
.hero p.lede{font-size:19px;max-width:30em;margin:0}
.hero .art{background:var(--hero-art);margin:22px 0;--shape-fill:transparent;--shape-line:var(--hero-line);--shape-stroke:var(--line)}
.hero .art[data-shape]{background:var(--hero-art)!important}
.hero-rule{height:var(--hero-rule-h);background:var(--hero-rule) left center/auto 100% repeat-x;max-width:1184px;margin:6px auto 8px}
.caption{font:13px var(--ui);color:var(--muted);text-align:right;padding:6px 32px 0;max-width:1184px;margin:0 auto}

/* sections */
section{margin:68px 0 0}
section>h2{font:700 14px var(--ui);letter-spacing:.16em;text-transform:uppercase;padding-bottom:18px;margin-bottom:30px;
  background:var(--rule) left bottom/auto var(--rule-h) repeat-x}
.row{display:flex;flex-wrap:wrap;gap:40px 36px;align-items:flex-start}

/* buttons */
.btn{position:relative;font:700 16px/1 var(--ui);letter-spacing:.02em;padding:16px 34px;border:0;background:none;color:var(--on-ink);cursor:pointer;
  --shape-fill:var(--ink);--shape-line:transparent;--shape-stroke:0}
.btn.primary::after{content:"";position:absolute;left:14px;right:14px;top:calc(100% + 2px);height:var(--fringe-h);
  background:var(--fringe) left top/auto 100% repeat-x;transition:height .2s;pointer-events:none}
.btn.primary:hover,.btn.primary.is-hover{--shape-fill:var(--ink-hover)}
.btn.primary:hover::after,.btn.primary.is-hover::after{height:calc(var(--fringe-h) * 1.45)}
.btn:focus-visible{outline:none}
.btn.secondary{color:var(--ink);--shape-fill:transparent;--shape-line:var(--ink);--shape-stroke:var(--line)}
.btn.secondary:hover,.btn.secondary.is-hover{--shape-fill:var(--card-bg)}
.btn.quiet{color:var(--ink);padding:12px 4px 16px;--shape-fill:transparent;background:var(--underline) left bottom/auto var(--underline-h) repeat-x}
.btn.quiet:hover,.btn.quiet.is-hover{background-image:var(--underline-focus)}
.btn:disabled{opacity:.42;cursor:not-allowed}
.btn:disabled::after,.btn:disabled::before{display:none}

/* fields */
.field{display:flex;flex-direction:column;gap:7px;width:290px}
.field label{font:600 14px var(--ui);letter-spacing:.03em}
.field input,.field select{font:17px var(--body);color:var(--ink);border:0;border-radius:0;outline:none;width:100%;padding:11px 10px 15px;
  background:var(--underline) left bottom/auto var(--underline-h) repeat-x,var(--field-bg);appearance:none}
.field input::placeholder{color:var(--muted)}
.field select{background:var(--caret) right 10px center/14px no-repeat,var(--underline) left bottom/auto var(--underline-h) repeat-x,var(--field-bg);padding-right:34px}
.field input:focus,.field.is-focus input{background:var(--underline-focus) left bottom/auto calc(var(--underline-h) * 1.25) repeat-x,var(--field-bg)}
.field.is-error input{background:var(--underline-error) left bottom/auto var(--underline-error-h) repeat-x,var(--field-bg)}
.field .hint{font:14px var(--ui);color:var(--muted)}
.field.is-error .hint{color:var(--danger);font-weight:600}

/* choices */
.choice{display:flex;align-items:center;gap:12px;font:16px var(--ui);cursor:pointer}
.choice input{appearance:none;margin:0;width:24px;height:24px;flex:none;cursor:pointer}
.choice input[type=checkbox]{border:var(--line) solid var(--ink);border-radius:var(--check-radius);background:var(--field-bg)}
.choice input[type=checkbox]:checked{background:var(--check) center/78% no-repeat,var(--check-bg)}
.choice input[type=radio]{background:var(--ring) center/100% no-repeat}
.choice input[type=radio]:checked{background:var(--dot) center/50% no-repeat,var(--ring) center/100% no-repeat}
.choice input:focus-visible{outline:2px solid var(--focus);outline-offset:3px}
.toggle{display:flex;align-items:center;gap:12px;font:16px var(--ui);cursor:pointer}
.toggle input{position:absolute;opacity:0}
.toggle .track{width:52px;height:28px;border-radius:var(--track-radius);background:var(--field-bg);box-shadow:inset 0 0 0 var(--line) var(--ink);position:relative;transition:background .2s}
.toggle .track::after{content:"";position:absolute;top:4px;left:4px;width:20px;height:20px;border-radius:50%;background:var(--ink);transition:transform .2s}
.toggle input:checked+.track{background:var(--ink)}
.toggle input:checked+.track::after{transform:translateX(24px);background:var(--knob) center/100% no-repeat,var(--knob-bg)}
.toggle input:focus-visible+.track{outline:2px solid var(--focus);outline-offset:3px}

/* tabs */
.tabs{display:flex;gap:30px;box-shadow:inset 0 calc(-1 * var(--hair)) 0 color-mix(in srgb,var(--ink) 25%,transparent);width:100%}
.tab{position:relative;background:none;border:0;padding:12px 4px 18px;font:600 16px var(--ui);color:var(--muted);cursor:pointer}
.tab[aria-selected=true]{color:var(--ink)}
.tab[aria-selected=true]::after{content:"";position:absolute;left:-8px;right:-8px;bottom:calc(-0.45 * var(--tab-h));height:var(--tab-h);
  background:var(--tab) left top/auto 100% repeat-x}
.tabpanel{padding:22px 2px 0;max-width:40em}

/* divider */
.divider{height:40px;width:100%;background:var(--divider-mark) center/40px 40px no-repeat,var(--divider) left center/auto var(--divider-h) repeat-x}

/* cards */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:40px 36px}
.cards>.specimen{align-items:stretch}
.card{position:relative;background:var(--card-bg);--shape-fill:var(--card-bg);--shape-line:var(--ink);--shape-stroke:var(--line)}
.card .art{height:190px;background-size:cover;background-position:center}
.card .body{padding:20px 24px 30px}
.card h3{font-size:24px;margin-bottom:8px}
.card p{margin:0;font-size:16px}
.card .tag{font:600 12px var(--ui);letter-spacing:.12em;text-transform:uppercase;color:var(--muted);display:block;margin-bottom:6px}
.cardwrap{position:relative}
.cardwrap .corner{position:absolute;width:28px;height:28px;background:var(--corner) center/contain no-repeat;pointer-events:none;z-index:2}
.cardwrap .tl{top:-12px;left:-12px}.cardwrap .tr{top:-12px;right:-12px}.cardwrap .bl{bottom:-12px;left:-12px}.cardwrap .br{bottom:-12px;right:-12px}
.cardwrap .crown{position:absolute;top:-24px;left:50%;width:48px;height:48px;margin-left:-24px;background:var(--crown) center/contain no-repeat;z-index:2;pointer-events:none}

/* feedback */
.alert{display:grid;grid-template-columns:10px 1fr;gap:16px;padding:16px 34px 16px 20px;max-width:520px;--shape-fill:var(--card-bg);--shape-line:var(--ink);--shape-stroke:var(--hair)}
.alert .mark{background:var(--mark) center/100% 100% no-repeat}
.alert.error .mark{background-image:var(--mark-error)}
.alert strong{font-family:var(--ui)}
.alert p{margin:2px 0 0;font-size:16px}
.progress{width:320px}
.progress .track{height:12px;background:var(--field-bg);box-shadow:inset 0 0 0 var(--hair) var(--ink);border-radius:var(--track-radius);overflow:hidden}
.progress .bar{height:100%;width:62%;background:var(--ink);transition:width .6s}
.progress .ticks{height:var(--ticks-h);margin-top:4px;background:var(--ticks) left top/auto 100% repeat-x;opacity:.8}
.progress small{font:13px var(--ui);color:var(--muted)}
.spinner{width:52px;height:52px;background:var(--spinner) center/contain no-repeat;animation:spin 3.2s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.spinner{animation:none}*{transition:none!important}}
.badge{display:inline-block;position:relative;font:700 12px/1 var(--ui);letter-spacing:.1em;text-transform:uppercase;padding:8px 22px 8px 34px;color:var(--ink);
  --shape-fill:var(--card-bg);--shape-line:var(--ink);--shape-stroke:var(--hair)}
.badge::before{content:"";position:absolute;left:16px;top:50%;width:10px;height:10px;margin-top:-5px;background:var(--bullet) center/contain no-repeat}
.badge.solid{color:var(--on-ink);padding-left:22px;--shape-fill:var(--ink)}
.badge.solid::before{display:none}

/* ornaments */
.orns{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:34px 30px}
.orn-strip{width:100%;height:46px;background-repeat:repeat-x;background-position:left center;background-size:auto 70%}
.orn-glyph{width:72px;height:72px;background-size:contain;background-repeat:no-repeat;background-position:center}

/* prose */
.prose{max-width:38em}
.prose p{margin:0 0 1em}
.prose p.drop::first-letter{font-family:var(--head);font-weight:var(--head-weight);float:left;font-size:4.1em;line-height:.8;padding:6px 10px 0 0;color:var(--drop)}
.prose ul{list-style:none;padding:0;margin:0 0 1em}
.prose li{padding-left:26px;margin:.35em 0;background:var(--bullet) 2px .5em/12px no-repeat}
.contrast{font:13px/1.5 var(--ui);color:var(--muted);columns:2;max-width:760px}
.contrast b{color:var(--ink)}
.offline-note{display:none;font:13px var(--ui);color:var(--muted);padding:0 32px;text-align:right}
.offline .offline-note{display:block}
"""

THEME_CSS = {
    "woodcut": """
.btn.primary{padding:16px 46px}
.btn.primary::before,.btn.primary::after{content:"";position:absolute;top:18%;bottom:auto;height:64%;width:20px;left:auto;right:auto;transition:none;
  background:var(--gouge-l) center/100% 100% no-repeat}
.btn.primary::before{left:10px}
.btn.primary::after{right:10px;background-image:var(--gouge-r)}
.btn.primary:hover::after,.btn.primary.is-hover::after{height:64%}
.btn.primary:hover,.btn.primary.is-hover{--shape-fill:var(--accent);color:var(--on-accent)}
.btn.secondary:hover,.btn.secondary.is-hover{--shape-fill:var(--ink);color:var(--paper)}
.tab{padding:10px 18px 12px;--shape-fill:transparent}
.tab[aria-selected=true]{color:var(--paper);--shape-fill:var(--ink)}
.tab[aria-selected=true]::after{display:none}
.tabs{gap:8px;padding-bottom:12px;box-shadow:none;background:var(--tab) left bottom/auto var(--tab-h) repeat-x}
.card .body{padding-bottom:40px}
""",
    "engraving": """
.btn{padding:16px 40px}
.badge{padding:8px 26px 8px 36px}.badge::before{left:20px}
.alert{padding-right:44px}
""",
}


def spec_css(name: str, t: dict) -> dict:
    out = {}
    for var, spec in SPECS[name].items():
        out[var.replace("_", "-")] = render_var(spec, t)
        if len(spec) > 4:
            out[var.replace("_", "-") + "-h"] = spec[4]
    return out


def sp(kind: str, label: str, inner: str, vars_: str = "", svg: str = "", tag: str = "figure", cls: str = "") -> str:
    """A specimen wrapper: the toolbar (added by app.js) acts on it."""
    a = f' data-vars="{vars_}"' if vars_ else ""
    a += f' data-svg="{svg}"' if svg else ""
    return f'<{tag} class="specimen {cls}" data-kind="{kind}" data-label="{html.escape(label)}"{a}>{inner}<figcaption>{html.escape(label)}</figcaption></{tag}>'


def page(name: str, t: dict, art: dict, checks: list) -> str:
    S = t["shapes"]
    shape = lambda kind, **kw: (f' data-shape="{S[kind]}"' + "".join(f' data-{k}="{v}"' for k, v in kw.items())) if S.get(kind) else ""
    v = {
        "ink": t["ink"], "paper": t["paper"], "accent": t["accent"], "card-bg": t["card_bg"], "field-bg": t["field_bg"],
        "muted": t["muted"], "danger": t["danger"], "ink-hover": t["ink_hover"], "focus": t["focus"],
        "on-ink": t["paper"], "on-accent": t["on_accent"],
        "head": t["head"], "head-weight": t["head_weight"], "body": t["body"], "ui": t["ui"], "line": t["line"], "hair": t["hair"],
        "check-radius": "0" if name == "woodcut" else "3px", "check-bg": t["ink"] if name == "woodcut" else t["field_bg"],
        "track-radius": "0" if name == "woodcut" else "999px", "knob-bg": t["ink"] if name == "woodcut" else (t["accent"] if name == "filigree" else t["paper"]),
        "drop": t["focus"] if name != "woodcut" else t["ink"], "hero-art": art["hero"], "hero-line": t["accent"] if name == "filigree" else t["ink"],
        "fringe": "none", "fringe-h": "0px",
    }
    v.update(spec_css(name, t))
    root = ":root{" + ";".join(f"--{k}:{val}" for k, val in v.items()) + "}"
    nav = "".join(f'<a href="{k}.html" class="{"on" if k == name else ""}">{k.title()}</a>' for k in ("bands", "marks", "strokes"))
    nav += '<a href="/designlib/saved/">Saved</a>'

    card_html = []
    for i, c in enumerate(art["cards"]):
        deco, cvars = "", ""
        if name == "engraving":
            deco, cvars = '<i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>', "corner"
        if name == "filigree":
            deco, cvars = '<i class="crown"></i>', "crown"
        extra = {"inner": 6, "r": 22} if name == "engraving" else ({"inner": 7} if name == "filigree" else {})
        attrs = shape("card", seed=11 + i, clip="", **extra)
        card_html.append(sp("card", c["tag"],
            f'<div class="cardwrap"><article class="card"{attrs}><div class="art" style="background:{c["bg"]};background-size:cover;background-position:center;{c.get("style", "")}"></div>'
            f'<div class="body"><span class="tag">{c["tag"]}</span><h3>{c["title"]}</h3><p>{c["text"]}</p></div></article>{deco}</div>', cvars))

    def btn(cls, label, extra="", act='data-action="save-self"'):
        return f'<button class="btn {cls}"{shape("button", seed=sum(map(ord, label + cls)) % 997)} {act} {extra}>{label}</button>'
    btn_vars = "gouge-l gouge-r" if name == "woodcut" else "fringe"

    orn_items = []
    for var, spec in SPECS[name].items():
        cssvar = var.replace("_", "-")
        if spec[3] == "frieze":
            el = f'<div class="orn-strip" style="background-image:var(--{cssvar})"></div>'
        else:
            bg = t["ink"] if spec[2] == "paper" else "transparent"
            el = f'<div class="orn-glyph" style="background-image:var(--{cssvar});background-color:{bg}"></div>'
        orn_items.append(sp("ornament", f"{var.replace('_', ' ')} · {spec[0].replace('_', ' ')}", el, vars_=cssvar, svg=cssvar))

    check_rows = "".join(f'<div><b>{label}</b> {r:.1f}:1 (needs {need:g})</div>' for label, r, need in checks)
    hero_shape = shape("hero", seed=5, clip="", **({"inner": 9} if name != "woodcut" else {}), **({"r": 46} if name == "engraving" else {}))
    tabs_html = f'''<div style="width:100%"><div class="tabs" role="tablist">
    <button class="tab" role="tab" aria-selected="true" aria-controls="p1"{shape("tab", seed=3)}>Strokes</button>
    <button class="tab" role="tab" aria-selected="false" aria-controls="p2"{shape("tab", seed=4)}>Compounds</button>
    <button class="tab" role="tab" aria-selected="false" aria-controls="p3"{shape("tab", seed=6)}>Colorways</button></div>
  <div class="tabpanel" id="p1">Twelve stroke families, from hairline dashes to tapered hooks and closed loops, measured over a hundred features.</div>
  <div class="tabpanel" id="p2" hidden>Combs, feathers, branching lines, hatching and bands: structures of several marks, put back together at their junctions.</div>
  <div class="tabpanel" id="p3" hidden>Opponent colour pairs, yellow against violet most of all, read from each feature's dominant chroma axis.</div></div>'''

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{t['title']} · style tile</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family={t['fonts']}&display=swap" rel="stylesheet">
<style>{root}{CSS}{THEME_CSS[name]}</style></head><body data-theme="{name}">
<nav class="switch">{nav}</nav>
<p class="offline-note">Opened as a file: shapes and shape variants work, but saving and ornament variants need the local server (http://127.0.0.1:8790/designlib/tiles/{name}.html).</p>
<header class="hero"><div class="panel">
  <span class="eyebrow">Grammar of Ornament · {t['title']}</span>
  <h1>Ornament learned from a neural network</h1>
  <p class="lede">{t['blurb']} Every mark here was drawn from the measured proportions of strokes in vgg16's feature visualizations. Hover anything to make a variant or save it.</p>
  <div class="row" style="margin-top:12px">{btn("primary", "Begin", 'data-target="#components"', 'data-action="scroll"')}{btn("secondary", "See the ornaments", 'data-target="#ornaments"', 'data-action="scroll"')}</div>
</div><div class="art" role="img" aria-label="{art['hero_label']}"{hero_shape}></div></header>
<div class="caption">{art['hero_label']}</div>
<div class="hero-rule"></div>
<div class="wrap">

<section id="components"><h2>Buttons</h2><div class="row">
  {sp("button", "primary", btn("primary", "Save pattern"), btn_vars)}
  {sp("button", "primary, hover", btn("primary is-hover", "Save pattern"), btn_vars)}
  {sp("button", "primary, focus", btn("primary is-focus", "Save pattern"), btn_vars)}
  {sp("button", "primary, disabled", btn("primary", "Save pattern", "disabled"), btn_vars)}
  {sp("button", "secondary", btn("secondary", "Export SVG", 'data-var="divider"', 'data-action="export"'))}
  {sp("button", "secondary, hover", btn("secondary is-hover", "Export SVG", 'data-var="divider"', 'data-action="export"'))}
  {sp("button", "quiet", '<button class="btn quiet" data-action="layers">Show layers</button>', "underline underline-focus")}
</div></section>

<section><h2>Fields</h2><div class="row">
  {sp("field", "text field", '<div class="field"><label for="f1">Pattern name</label><input id="f1" placeholder="Feathered comb"><span class="hint">Shown in the library.</span></div>', "underline underline-focus")}
  {sp("field", "text field, focus", '<div class="field is-focus"><label for="f2">Repeat width</label><input id="f2" value="48 px"><span class="hint">Focused.</span></div>', "underline-focus")}
  {sp("field", "text field, error", '<div class="field is-error"><label for="f3">Stroke weight</label><input id="f3" value="0.1 px"><span class="hint">Too fine to print; use at least 0.5 px.</span></div>', "underline-error")}
  {sp("field", "select", '<div class="field"><label for="f4">Symmetry</label><select id="f4"><option>Glide reflection</option><option>Mirror</option><option>Rotation</option></select><span class="hint">How the tile repeats.</span></div>', "underline caret")}
</div></section>

<section><h2>Choices</h2><div class="row" style="gap:30px 46px">
  {sp("checkbox", "checkbox", '<label class="choice"><input type="checkbox" checked> Include hatching</label>', "check")}
  {sp("radio", "radio", '<div style="display:flex;gap:26px"><label class="choice"><input type="radio" name="r" checked> Frieze</label><label class="choice"><input type="radio" name="r"> Rosette</label></div>', "ring dot")}
  {sp("toggle", "toggle", '<div style="display:flex;gap:26px"><label class="toggle"><input type="checkbox" checked><span class="track"></span> Relief</label><label class="toggle"><input type="checkbox"><span class="track"></span> Animate</label></div>', "knob")}
</div></section>

<section><h2>Tabs</h2>{sp("tabs", "tabs", tabs_html, "tab")}</section>

<section><h2>Divider</h2>{sp("divider", "divider", '<div class="divider" role="separator"></div>', "divider divider-mark")}</section>

<section><h2>Cards · fill options</h2><div class="cards">{''.join(card_html)}</div></section>

<section><h2>Feedback</h2><div class="row" style="gap:36px 46px;align-items:center">
  {sp("alert", "alert", f'<div class="alert"{shape("alert", seed=2)}><span class="mark"></span><div><strong>Pattern saved.</strong><p>It is in the library under Combs.</p></div></div>', "mark")}
  {sp("alert", "alert, error", f'<div class="alert error"{shape("alert", seed=9)}><span class="mark"></span><div><strong>Could not export.</strong><p>The tile is wider than the page.</p></div></div>', "mark-error")}
  {sp("progress", "progress", '<div class="progress"><div class="track"><div class="bar"></div></div><div class="ticks"></div><small>Tracing strokes · 62%</small></div>', "ticks")}
  {sp("spinner", "spinner", '<div class="spinner" role="img" aria-label="Loading"></div>', "spinner")}
  {sp("badge", "badges", f'<div style="display:flex;gap:14px"><span class="badge"{shape("badge", seed=1)}>Comb</span><span class="badge"{shape("badge", seed=2)}>Feather</span><span class="badge solid"{shape("badge", seed=3)}>New</span></div>', "bullet")}
</div></section>

<section><h2>Text</h2>{sp("prose", "running text", '''<div class="prose">
  <p class="drop">Owen Jones gathered the ornament of every culture he could reach into one grammar. These are the marks a network made for itself: combs that hang from a wavering spine, feathers barbed on both sides, short tapered strokes laid side by side like an engraver's shading.</p>
  <ul><li>Ornament sits at the edges of a control, never under its words.</li><li>Text sits on solid ground at 4.5:1 or better.</li><li>Every repeating border is one seamless tile.</li></ul></div>''', "bullet")}
</section>

<section id="ornaments"><h2>Ornaments</h2><div class="orns">{''.join(orn_items)}</div></section>

<section><h2>Contrast</h2><div class="contrast">{check_rows}</div></section>
</div>
<script src="shapes.js"></script><script src="app.js"></script></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def build(screenshot: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assets = OUT / "assets"
    assets.mkdir(exist_ok=True)
    T = themes()
    rows = list(csv.DictReader(open(HERE / "compounds" / "vgg16features" / "by_feature.csv")))
    score = lambda r, words: sum(int(v) for k, v in r.items() if any(w in k for w in words))
    combiest = max(rows, key=lambda r: score(r, ("comb", "feather")) / (1 + score(r, ("mesh", "loop"))))["feature"]
    bandiest = max(rows, key=lambda r: score(r, ("band",)))["feature"]
    cw = list(csv.DictReader(open(HERE / "colorways" / "convnext_basefeatures" / "summary.csv")))
    motifs_all = sorted((HERE / "vgg16motifs").glob("*/*/motif.json"), key=lambda p: -json.loads(p.read_text())["width"])
    motif_png = motifs_all[0].parent / "motif.png"
    print(f"  feature for combs and feathers: {combiest}\n  feature for bands: {bandiest}\n  motif: {motifs_all[0].parent.relative_to(HERE)}", flush=True)

    for name, t in T.items():
        key = tuple(sorted(t["pair"]))
        match = max((r for r in cw if tuple(sorted((r["name_a"], r["name_b"]))) == key),
                    key=lambda r: min(float(r["chroma_a"]), float(r["chroma_b"])))
        mesh = json.loads((HERE / "colorways" / "convnext_basefeatures" / f"{match['name']}.json").read_text())["mesh"]["css"]
        feat = combiest if name != "woodcut" else bandiest
        up = upscaled_feature(feat, HERE / "vgg16features")
        shutil.copyfile(up, assets / f"{feat}.png")
        dark, light = sorted((t["ink"], t["card_bg"]), key=luminance)
        duo = assets / f"{name}-duotone.png"
        duotone(up, dark, light, duo)
        conv_up = upscaled_feature(match["name"], HERE / "convnext_basefeatures")
        shutil.copyfile(conv_up, assets / f"{match['name']}.png")
        cards = [
            {"bg": f"url(assets/{feat}.png)", "tag": "Fill · feature, upscaled", "title": "The feature itself",
             "text": "The render as the network made it, enlarged four times. Strongest as a picture, never behind text."},
            {"bg": f"url(assets/{duo.name})", "tag": "Fill · duotone", "title": "In the page's inks",
             "text": "The same feature mapped onto this page's two colours, so it sits with the type."},
            {"bg": mesh, "tag": "Fill · colorway mesh", "title": "Its colours, blurred",
             "text": f"Radial gradients from the opponent pair of a convnext feature, {match['name_a']} against {match['name_b']}."},
            {"bg": f"url(assets/{match['name']}.png)", "tag": "Fill · convnext feature", "title": "The source of the colours",
             "text": "The feature the mesh and the page's palette were read from."},
        ]
        if name == "woodcut":
            ink_motif(motif_png, t["ink"], assets / "motif-ink.png")
            hero = "url(assets/motif-ink.png) center/contain no-repeat"
            hero_label = "A motif cut from a vgg16 feature, printed in black."
            cards.append({"bg": "url(assets/motif-ink.png)", "style": f"background-size:contain;background-repeat:no-repeat;background-color:{t['card_bg']}",
                          "tag": "Fill · motif", "title": "A motif", "text": "A single cut-out from a feature print: right for a large spot, too busy for a small control."})
        elif name == "engraving":
            hero = f"url(assets/{duo.name}) center/cover no-repeat"
            hero_label = "A vgg16 feature rich in combs and feathers, printed in the page's ink, in a cartouche."
        else:
            hero = f"url(assets/{match['name']}.png) center/cover no-repeat"
            hero_label = f"A convnext feature in {match['name_a']} and {match['name_b']}, under a pointed arch."
        art = {"hero": hero, "hero_label": hero_label, "cards": cards}

        on_primary = t["on_accent"] if name == "filigree" else t["paper"]
        prim_bg = t["accent"] if name == "filigree" else t["ink"]
        hover_fg, hover_bg = ((t["on_accent"], t["accent"]) if name == "woodcut" else
                              (t["paper"], t["ink"]) if name == "filigree" else (t["paper"], t["ink_hover"]))
        checks = [(label, contrast(fg, bg), need) for label, fg, bg, need in (
            ("body text on ground", t["ink"], t["paper"], 4.5), ("text on cards and fields", t["ink"], t["card_bg"], 4.5),
            ("primary button label", on_primary, prim_bg, 4.5), ("primary button label, hover", hover_fg, hover_bg, 4.5),
            ("muted text on ground", t["muted"], t["paper"], 4.5), ("muted text on cards", t["muted"], t["card_bg"], 4.5),
            ("error text on ground", t["danger"], t["paper"], 4.5), ("focus ring on ground", t["focus"], t["paper"], 3.0),
            ("drop cap on ground (large)", t["focus"] if name != "woodcut" else t["ink"], t["paper"], 3.0))]
        if name == "filigree":
            checks.append(("secondary label (gold) on ground", contrast(t["accent"], t["paper"]), 4.5))
        bad = [c for c in checks if c[1] < c[2]]
        print(f"  {name:10s} " + "  ".join(f"{r:.1f}" for _, r, _ in checks) + ("   FAIL: " + ", ".join(c[0] for c in bad) if bad else "   all pass"), flush=True)
        (OUT / f"{name}.html").write_text(page(name, t, art, checks))

    print(f"  → {OUT}")

    if screenshot:
        for name in T:
            out = OUT / f"{name}.png"
            with tempfile.TemporaryDirectory() as prof:
                try:
                    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", f"--user-data-dir={prof}",
                                    "--virtual-time-budget=5000", "--window-size=1280,4600", f"--screenshot={out}", (OUT / f"{name}.html").as_uri()],
                                   capture_output=True, timeout=90)
                    print(f"  screenshot {out.name}: ok", flush=True)
                except subprocess.TimeoutExpired:
                    print(f"  screenshot {out.name}: Chrome timed out", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screenshot", action="store_true", help="render each page to PNG with headless Chrome")
    build(ap.parse_args().screenshot)
