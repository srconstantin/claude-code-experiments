#!/usr/bin/env python3
"""
Grammar of Ornament — atlas tiles: style tiles built from real compound
pieces (designlib/atlas/, made by atlas.py) instead of generated ornament.

Nothing is symmetric and nothing repeats: every button is the outline of a
different lumpy band, every card's picture is clipped to the outline of a
different blob, combs hang from one corner of a button, crosshatch clusters
serve as checks, bullets and alert marks, a single feather is stretched into a
divider. "↻ variant" swaps a piece for another of the same kind.

  herbarium   ink on bone with an indigo accent; the hero is a specimen
              sheet of atlas pieces, labelled like pressed plants
  bloom       cream and gold on deep green; pictures (the features
              themselves and the colorway meshes) live inside blob outlines

Shares the page machinery of styletiles.py (shapes.js draws the shapes,
app.js the specimen toolbars) and the same legibility rules: text on solid
ground, ornament beside and under controls, never under words, WCAG checks.

Run:  python3 atlastiles.py [--screenshot]      # → designlib/tiles/{herbarium,bloom}.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

import styletiles as ST
from styletiles import contrast, lab_hex, luminance, push_to, to_lab, sp

HERE = Path(__file__).resolve().parent
OUT = HERE / "designlib" / "tiles"
ATLAS = HERE / "designlib" / "atlas" / "atlas.json"


# ─────────────────────────────────────────────────────────────────────────────
#  Atlas
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_of(d: str) -> tuple[float, float, float, float]:
    nums = np.array([float(v) for v in re.findall(r"-?\d+\.?\d*", d)]).reshape(-1, 2)
    return nums[:, 0].min(), nums[:, 1].min(), nums[:, 0].max(), nums[:, 1].max()


def _poly_area(d: str) -> float:
    tot = 0.0
    for ring in d.split("M")[1:]:
        p = np.array([float(v) for v in re.findall(r"-?\d+\.?\d*", ring)]).reshape(-1, 2)
        x, y = p[:, 0], p[:, 1]
        tot += abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    return tot


@lru_cache(maxsize=1)
def atlas() -> dict:
    """Pieces by group, plus outline shapes (normalized to their own bbox),
    graded by solidity (area / bbox area) so labels never sit across a neck."""
    pieces = json.loads(ATLAS.read_text())["pieces"]
    by = {}
    for p in pieces:
        by.setdefault(p["group"], []).append(p)
    shapes = {"button": [], "card": []}
    for p in pieces:
        if not p["outline"]:
            continue
        x0, y0, x1, y1 = _bbox_of(p["outline"])
        W, H = x1 - x0, y1 - y0
        sol = _poly_area(p["outline"]) / max(W * H, 1e-6)
        d = re.sub(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", lambda m: f"{float(m.group(1)) - x0:.2f},{float(m.group(2)) - y0:.2f}", p["outline"])
        rec = {"id": p["id"], "w": round(W, 2), "h": round(H, 2), "d": d, "solidity": round(sol, 3)}
        if p["group"] == "band" and 2.4 <= W / H <= 7 and sol >= 0.64:
            shapes["button"].append(rec)
        if p["group"] == "blob" and 0.55 <= W / H <= 1.9 and sol >= 0.5:
            shapes["card"].append(rec)
    return {"by": by, "shapes": shapes, "ids": {p["id"]: p for p in pieces}}


def piece_uri(p: dict, ink: str, relief: str, mode: str = "glyph", outline_only: bool = False, stroke: float = 0.7) -> str:
    pad = 0.6
    W, H = p["w"] + 2 * pad, p["h"] + 2 * pad
    if outline_only and p["outline"]:
        body = f'<path fill="none" stroke="{ink}" stroke-width="{stroke}" stroke-linejoin="round" d="{p["outline"]}"/>'
    else:
        body = f'<path fill="{ink}" d="{p["ink"]}"/>' + (f'<path fill="{relief}" d="{p["relief"]}"/>' if p["relief"] else "")
    par = "none" if mode == "stretch" else ("xMinYMid meet" if mode == "left" else "xMidYMid meet")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.2f} {H:.2f}" preserveAspectRatio="{par}">'
           f'<g transform="translate({pad},{pad})">{body}</g></svg>')
    return f'url("{ST.O.data_uri(svg)}")'


# ─────────────────────────────────────────────────────────────────────────────
#  Themes and their pieces
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def themes() -> dict:
    T = {}
    T["herbarium"] = dict(
        title="Herbarium", blurb="Real pieces of the features, pressed flat: lumpy bands for buttons, blobs for pictures, repeated combs and feathers for lines, crosshatch clusters for marks.",
        ink=lab_hex(16, -6, 3), paper=lab_hex(94, 0.5, 6), accent=lab_hex(38, 22, -48), card_bg=lab_hex(97, 0, 4), field_bg=lab_hex(97, 0, 4),
        muted=lab_hex(44, -4, 4), danger=lab_hex(43, 58, 42),
        head="'Instrument Serif', Georgia, serif", head_weight="400", body="'Instrument Sans', 'Helvetica Neue', sans-serif",
        ui="'Instrument Sans', 'Helvetica Neue', sans-serif", fonts="Instrument+Serif:ital@0;1&family=Instrument+Sans:wght@400;500;600;700",
        line="1.6px", hair="1px", dark=False)
    T["bloom"] = dict(
        title="Bloom", blurb="The same pieces as frames: pictures of the features and their colour meshes inside blob outlines, gold feathers and combs on deep green.",
        ink=lab_hex(92, 2, 12), paper=lab_hex(14, -16, 2), accent=lab_hex(78, 6, 58), card_bg=lab_hex(19, -16, 2), field_bg=lab_hex(18, -15, 2),
        muted=lab_hex(70, -8, 8), danger=lab_hex(68, 46, 30),
        head="'Young Serif', Georgia, serif", head_weight="400", body="'Figtree', 'Helvetica Neue', sans-serif",
        ui="'Figtree', 'Helvetica Neue', sans-serif", fonts="Young+Serif&family=Figtree:wght@400;500;600;700",
        line="1.4px", hair="1px", dark=True)
    for t in T.values():
        t["focus"] = push_to(t["accent"], t["paper"], 3.2)
        t["on_accent"] = t["ink"] if contrast(t["ink"], t["accent"]) > contrast(t["paper"], t["accent"]) else t["paper"]
        L = to_lab(t["ink"])[0]
        t["ink_hover"] = lab_hex(L + (12 if L < 50 else -14), *to_lab(t["ink"])[1:])
    return T


# var → (group or id, index, colour, mode, options)
SPECS = {
    "herbarium": {
        "underline": ("feather", 1, "ink", "strip", {}), "underline_focus": ("feather", 1, "accent", "strip", {}),
        "underline_error": ("comb", 0, "danger", "strip", {}),
        "rule": ("hatching", 0, "ink", "strip", {}), "rule2": ("comb", 2, "ink", "strip", {}), "rule3": ("feather", 3, "ink", "strip", {}),
        "tab": ("comb", 1, "ink", "strip", {}),
        "divider": ("feather", 0, "ink", "strip", {}), "divider_mark": ("crosshatch", 7, "accent", "glyph", {}),
        "check": ("crosshatch", 13, "ink", "glyph", {}), "box": ("blob", 7, "ink", "stretch", {"outline_only": True, "stroke": 0.9}),
        "ring": ("blob", 9, "ink", "stretch", {"outline_only": True, "stroke": 0.8}), "dot": ("blob", 10, "accent", "glyph", {}),
        "knob": ("blob", 11, "paper", "glyph", {}),
        "spinner": ("comb", 9, "ink", "glyph", {}),
        "mark": ("crosshatch", 4, "accent", "glyph", {}), "mark_error": ("comb", 7, "danger", "glyph", {}),
        "bullet": ("crosshatch", 9, "ink", "glyph", {}), "bullet2": ("crosshatch", 15, "ink", "glyph", {}), "bullet3": ("crosshatch", 20, "ink", "glyph", {}),
        "bar": ("band", 30, "ink", "stretch", {}),
        "hero_rule": ("feather", 2, "ink", "strip", {}),
    },
    "bloom": {
        "underline": ("feather", 1, "muted", "strip", {}), "underline_focus": ("feather", 1, "accent", "strip", {}),
        "underline_error": ("comb", 0, "danger", "strip", {}),
        "rule": ("hatching", 0, "accent", "strip", {}), "rule2": ("comb", 2, "accent", "strip", {}), "rule3": ("feather", 3, "accent", "strip", {}),
        "tab": ("comb", 1, "accent", "strip", {}),
        "divider": ("feather", 0, "accent", "strip", {}), "divider_mark": ("blob", 3, "accent", "glyph", {}),
        "check": ("crosshatch", 13, "accent", "glyph", {}), "box": ("blob", 7, "ink", "stretch", {"outline_only": True, "stroke": 0.9}),
        "ring": ("blob", 9, "ink", "stretch", {"outline_only": True, "stroke": 0.8}), "dot": ("blob", 10, "accent", "glyph", {}),
        "knob": ("blob", 11, "paper", "glyph", {}),
        "spinner": ("comb", 9, "accent", "glyph", {}),
        "mark": ("crosshatch", 4, "accent", "glyph", {}), "mark_error": ("comb", 7, "danger", "glyph", {}),
        "bullet": ("crosshatch", 9, "accent", "glyph", {}), "bullet2": ("crosshatch", 15, "accent", "glyph", {}), "bullet3": ("crosshatch", 20, "accent", "glyph", {}),
        "bar": ("band", 30, "accent", "stretch", {}),
        "hero_rule": ("feather", 2, "accent", "strip", {}),
    },
}


def pick(group: str, index: int, seed: int | None = None) -> dict:
    A = atlas()
    if group in A["ids"]:
        return A["ids"][group]
    pool = A["by"][group]
    if group in ("blob",):
        pool = [p for p in pool if p["outline"]] or pool
    if seed is not None:
        return pool[np.random.default_rng(seed).integers(len(pool))]
    return pool[index % len(pool)]


def long_pieces(group: str) -> list[dict]:
    """A group's pieces that make good lines: long for their height, longest first."""
    pool = sorted(atlas()["by"][group], key=lambda p: -p["aspect"])
    return [p for p in pool if p["aspect"] >= 3.5] or pool[:4]


def strip_uri(group: str, ink: str, relief: str, seed: int, index: int | None = None, overlap: float = 0.06, H: float = 10.0) -> str:
    """One long piece, scaled to the line's height at its own proportions and
    repeated: the tile is a little shorter than the piece, and copies a period
    either side fill the overlap, so each repeat runs into the next."""
    pool = long_pieces(group)
    p = pool[index % len(pool)] if index is not None else pool[np.random.default_rng(seed).integers(len(pool))]
    s = H / p["h"]
    P = p["w"] * s * (1 - overlap)
    g = lambda x: (f'<g transform="translate({x:.2f},0) scale({s:.4f})"><path fill="{ink}" d="{p["ink"]}"/>'
                   + (f'<path fill="{relief}" d="{p["relief"]}"/>' if p["relief"] else "") + "</g>")
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {P:.2f} {H:.2f}" preserveAspectRatio="xMinYMid meet">{g(-P)}{g(0)}{g(P)}</svg>'
    return f'url("{ST.O.data_uri(svg)}")'


def render_var(theme: str, var: str, seed: int | None = None) -> str:
    group, index, colour, mode, opts = SPECS[theme][var]
    t = themes()[theme]
    relief = t["card_bg"] if not t["dark"] else t["paper"]
    if mode == "strip":
        return strip_uri(group, t[colour], relief, seed or 0, None if seed is not None else index, **opts)
    return piece_uri(pick(group, index, seed), t[colour], relief, mode, **opts)


def variant_vars(theme: str, names: list[str], seed: int) -> dict:
    out = {}
    for i, n in enumerate(names):
        key = n.replace("-", "_")
        if key in SPECS[theme]:
            out[n] = render_var(theme, key, seed + 7919 * i)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  The hero: a specimen sheet
# ─────────────────────────────────────────────────────────────────────────────

def specimen_sheet(t: dict, seed: int = 3, cols: int = 3, rows: int = 4) -> str:
    """Atlas pieces laid out like pressed plants, each with a small label."""
    rng = np.random.default_rng(seed)
    groups = ["comb", "feather", "crosshatch", "blob", "branch", "hatching", "comb", "band", "feather", "crosshatch", "comb", "branch"]
    cw, ch = 100.0, 74.0
    W, H = cols * cw, rows * ch
    parts = []
    used = set()
    for k, g in enumerate(groups[:cols * rows]):
        pool = [p for p in atlas()["by"][g] if p["id"] not in used]
        p = pool[rng.integers(len(pool))]
        used.add(p["id"])
        cx, cy = (k % cols) * cw + cw / 2, (k // cols) * ch + ch / 2 - 6
        s = min((cw - 22) / p["w"], (ch - 30) / p["h"]) * float(rng.uniform(0.75, 1.0))
        ang = float(rng.uniform(-14, 14))
        tf = f"translate({cx:.1f},{cy:.1f}) rotate({ang:.1f}) scale({s:.3f}) translate({-p['w'] / 2:.2f},{-p['h'] / 2:.2f})"
        parts.append(f'<g transform="{tf}"><path fill="{t["ink"]}" d="{p["ink"]}"/>' +
                     (f'<path fill="{t["paper"]}" d="{p["relief"]}"/>' if p["relief"] else "") + "</g>")
        lab = p["type"]
        feat = re.sub(r"__224px.*", "", p["feature"]).replace("vgg16__", "").replace("__", " ")
        parts.append(f'<text x="{cx:.1f}" y="{(k // cols) * ch + ch - 8:.1f}" text-anchor="middle" font-family="Instrument Serif, Georgia, serif" '
                     f'font-style="italic" font-size="6.2" fill="{t["muted"]}">{html.escape(lab)} · {html.escape(feat)}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}">{"".join(parts)}</svg>'
    return f'url("{ST.O.data_uri(svg)}")'


# ─────────────────────────────────────────────────────────────────────────────
#  Page
# ─────────────────────────────────────────────────────────────────────────────

CSS = r"""
/* atlas shapes: every shaped element gets its own piece (shapes.js "atlas") */
.btn{padding:18px 44px;font-weight:600;letter-spacing:.01em}
  background:var(--fringe) left top/contain no-repeat;transition:transform .25s;transform-origin:left top;pointer-events:none}
.btn.primary:hover,.btn.primary.is-hover{--shape-fill:var(--accent);color:var(--on-accent)}
.btn.quiet{background:var(--underline) left bottom/auto 12px repeat-x}
.btn.quiet:hover,.btn.quiet.is-hover{background-image:var(--underline-focus)}
.field input,.field select{background:var(--underline) left bottom/auto 12px repeat-x,var(--field-bg);padding-bottom:18px}
.field select{background:var(--underline) left bottom/auto 12px repeat-x,var(--field-bg)}
.field input:focus,.field.is-focus input{background:var(--underline-focus) left bottom/auto 14px repeat-x,var(--field-bg)}
.field.is-error input{background:var(--underline-error) left bottom/auto 14px repeat-x,var(--field-bg)}
.choice input[type=checkbox]{border:0;border-radius:0;background:var(--box) center/100% 100% no-repeat;width:26px;height:26px}
.choice input[type=checkbox]:checked{background:var(--check) center/80% no-repeat,var(--box) center/100% 100% no-repeat}
.choice input[type=radio]{background:var(--ring) center/100% 100% no-repeat;width:26px;height:24px}
.choice input[type=radio]:checked{background:var(--dot) center/62% no-repeat,var(--ring) center/100% 100% no-repeat}
.toggle .track{border-radius:999px}
.toggle input:checked+.track::after{background:var(--knob) center/100% no-repeat,var(--ink)}
.tabs{box-shadow:none}
.tab[aria-selected=true]::after{left:-4px;right:-10px;height:16px;bottom:-10px;background:var(--tab) left center/auto 100% no-repeat}
section>h2{background:var(--rule) left bottom/auto 15px repeat-x;padding-bottom:24px}
section:nth-of-type(3n+1)>h2{background-image:var(--rule2)}
section:nth-of-type(3n+2)>h2{background-image:var(--rule3)}
.divider{height:56px;background:var(--divider-mark) 71% 40%/46px 46px no-repeat,var(--divider) left center/auto 26px repeat-x}
.cards{gap:64px 44px}
.card{--shape-fill:transparent;--shape-line:transparent;--shape-stroke:0;background:none}
.card .art{height:230px;--shape-fill:transparent;--shape-line:var(--ink);--shape-stroke:var(--line)}
.card .body{padding:16px 6px 0}
.alert{grid-template-columns:40px 1fr;padding:26px 46px 26px 34px;--shape-fill:var(--card-bg);--shape-line:var(--ink);--shape-stroke:var(--hair)}
.alert .mark{width:40px;height:40px;background:var(--mark) center/contain no-repeat;align-self:center}
.alert.error .mark{background-image:var(--mark-error)}
.progress .track{height:16px;border-radius:999px;background:var(--field-bg)}
.progress .bar{background:var(--bar) left center/100% 100% no-repeat}
.progress .ticks{display:none}
.spinner{width:64px;height:64px}
.badge{padding:9px 24px;--shape-fill:var(--card-bg);--shape-line:var(--ink);--shape-stroke:var(--hair)}
.badge::before{display:none}
.badge.solid{--shape-fill:var(--ink);color:var(--paper);padding-left:24px}
.prose li{padding-left:34px;background:var(--bullet) 0 .1em/24px 24px no-repeat}
.prose li:nth-child(2){background-image:var(--bullet2)}.prose li:nth-child(3){background-image:var(--bullet3)}
.hero .art{background:var(--hero-art) center/contain no-repeat;margin:28px 0}
"""

THEME_CSS = {
    "herbarium": "",
    "bloom": """
.btn.primary{--shape-fill:var(--accent);color:var(--on-accent)}
.btn.primary:hover,.btn.primary.is-hover{--shape-fill:var(--ink);color:var(--paper)}
.btn.secondary{color:var(--accent);--shape-line:var(--accent)}
.badge.solid{--shape-fill:var(--accent);color:var(--on-accent)}
.card .art{--shape-line:var(--accent);height:250px}
.hero .art{background:var(--hero-art) center/cover no-repeat;--shape-line:var(--accent)}
.toggle input:checked+.track{background:var(--accent)}
.progress .bar{background-color:transparent}
""",
}


def page(name: str, t: dict, art: dict, checks: list) -> str:
    A = atlas()
    v = {
        "ink": t["ink"], "paper": t["paper"], "accent": t["accent"], "card-bg": t["card_bg"], "field-bg": t["field_bg"],
        "muted": t["muted"], "danger": t["danger"], "ink-hover": t["ink_hover"], "focus": t["focus"],
        "on-ink": t["paper"], "on-accent": t["on_accent"], "head": t["head"], "head-weight": t["head_weight"],
        "body": t["body"], "ui": t["ui"], "line": t["line"], "hair": t["hair"],
        "check-radius": "0", "check-bg": "transparent", "track-radius": "999px", "knob-bg": t["ink"],
        "drop": t["focus"], "hero-art": art["hero"], "hero-line": t["ink"], "fringe-h": "0px", "fringe": "none",
        "hero-rule-h": "24px", "rule-h": "14px", "underline-h": "13px", "underline-error-h": "15px", "tab-h": "16px",
        "divider-h": "28px", "ticks-h": "0px",
    }
    for var in SPECS[name]:
        v[var.replace("_", "-")] = render_var(name, var)
    root = ":root{" + ";".join(f"--{k}:{val}" for k, val in v.items()) + "}"
    shapes_json = json.dumps({"button": A["shapes"]["button"], "card": A["shapes"]["card"]}, separators=(",", ":"))

    base = ST.page("woodcut", {**ST.themes()["woodcut"], **t, "title": t["title"], "blurb": t["blurb"], "fonts": t["fonts"],
                               "shapes": dict(button="atlas", card=None, badge="atlas", alert="atlas", tab=None, hero="atlas")},
                   art, checks)
    # swap the woodcut-specific bits for this page's
    base = re.sub(r"<style>.*?</style>", lambda m: f"<style>{root}{ST.CSS}{CSS}{THEME_CSS[name]}</style>", base, count=1, flags=re.S)
    base = base.replace('data-theme="woodcut"', f'data-theme="{name}"').replace("tiles/woodcut.html", f"tiles/{name}.html")
    base = base.replace(f'<a href="{name}.html" class="">', f'<a href="{name}.html" class="on">')
    base = base.replace('data-shape="atlas"', 'data-shape="atlas" data-group="button"')
    base = base.replace('class="hero"><div class="panel">', 'class="hero"><div class="panel">', 1)
    # hero art: the specimen sheet unshaped (herbarium) or a picture clipped to a blob (bloom)
    if name == "herbarium":
        base = re.sub(r'(<div class="art" role="img"[^>]*?) data-shape="atlas" data-group="button"[^>]*>', r"\1>", base, count=1)
    else:
        base = re.sub(r'(<div class="art" role="img"[^>]*?) data-shape="atlas" data-group="button"([^>]*)>',
                      r'\1 data-shape="atlas" data-group="card" data-seed="2" data-clip=""\2>', base, count=1)
    # cards: picture clipped to a blob outline, a piece hanging off a corner, no card frame
    base = re.sub(r'<div class="art" style="', lambda m: '<div class="art" data-shape="atlas" data-group="card" data-clip="" style="', base)
    seeds = iter(range(3, 999, 7))
    base = re.sub(r'data-group="card" data-clip=""(?! data-seed)', lambda m: f'data-group="card" data-seed="{next(seeds)}" data-clip=""', base)
    # specimen vars: which ornaments each kind uses
    base = base.replace(' data-vars="gouge-l gouge-r"', '')
    base = base.replace('data-vars="check"', 'data-vars="check box"').replace('data-vars="mark"', 'data-vars="mark"')
    base = base.replace('data-vars="bullet"', 'data-vars="bullet bullet2 bullet3"').replace('data-vars="ticks"', 'data-vars="bar"')
    base = base.replace('data-vars="divider divider-mark"', 'data-vars="divider divider-mark"')
    # the ornaments section: this page's pieces
    orns = "".join(sp("ornament", f"{var.replace('_', ' ')} · {SPECS[name][var][0]}",
                      f'<div class="orn-glyph" style="width:100%;height:80px;background-image:var(--{var.replace("_", "-")});background-color:{t["ink"] if SPECS[name][var][2] == "paper" else "transparent"}"></div>',
                      vars_=var.replace("_", "-"), svg=var.replace("_", "-")) for var in SPECS[name])
    base = re.sub(r'(<div class="orns">).*?(</div></section>\s*<section><h2>Contrast)', lambda m: m.group(1) + orns + m.group(2), base, flags=re.S)
    base = base.replace("<script src=\"shapes.js\"></script>", f"<script>window.GOAtlas={shapes_json};</script><script src=\"shapes.js\"></script>")
    return base


def build(screenshot: bool = False) -> None:
    import csv
    T = themes()
    assets = OUT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    cw = list(csv.DictReader(open(HERE / "colorways" / "convnext_basefeatures" / "summary.csv")))
    rows = list(csv.DictReader(open(HERE / "compounds" / "vgg16features" / "by_feature.csv")))
    score = lambda r, words: sum(int(v) for k, v in r.items() if any(w in k for w in words))
    by_comb = sorted(rows, key=lambda r: -score(r, ("comb", "feather")))
    for name, t in T.items():
        if name == "herbarium":
            feats = [r["feature"] for r in by_comb[:2]]
            ups = [ST.upscaled_feature(f, HERE / "vgg16features") for f in feats]
            for f, u in zip(feats, ups):
                shutil.copyfile(u, assets / f"{f}.png")
            duo = assets / f"{name}-duotone.png"
            ST.duotone(ups[0], t["ink"], t["card_bg"], duo)
            vivid = max((r for r in cw if {r["name_a"], r["name_b"]} == {"purple", "yellow"}), key=lambda r: min(float(r["chroma_a"]), float(r["chroma_b"])))
            mesh = json.loads((HERE / "colorways" / "convnext_basefeatures" / f"{vivid['name']}.json").read_text())["mesh"]["css"]
            cards = [
                {"bg": f"url(assets/{feats[0]}.png)", "tag": "Fill · feature", "title": "The comb feature", "text": "Clipped to the outline of a blob from the atlas."},
                {"bg": f"url(assets/{duo.name})", "tag": "Fill · duotone", "title": "In ink and bone", "text": "The feature mapped onto the page's two colours, inside a different blob."},
                {"bg": mesh, "tag": "Fill · colorway mesh", "title": "Violet against yellow", "text": "A convnext feature's opponent pair, blurred, inside a hatched blob."},
                {"bg": f"url(assets/{feats[1]}.png)", "tag": "Fill · feature", "title": "A second comb feature", "text": "Each card takes its own blob; ↻ variant swaps it for another."},
            ]
            hero, label = specimen_sheet(t), "A sheet of real pieces from the atlas, each labelled with its type and the feature it came from."
        else:
            vivid = lambda r: min(float(r["chroma_a"]), float(r["chroma_b"]))
            picks = [max((r for r in cw if {r["name_a"], r["name_b"]} == pair), key=vivid)
                     for pair in ({"green", "purple"}, {"purple", "yellow"}, {"blue", "orange"})]
            ups = [ST.upscaled_feature(r["name"], HERE / "convnext_basefeatures") for r in picks]
            for r, u in zip(picks, ups):
                shutil.copyfile(u, assets / f"{r['name']}.png")
            mesh = json.loads((HERE / "colorways" / "convnext_basefeatures" / f"{picks[0]['name']}.json").read_text())["mesh"]["css"]
            cards = [
                {"bg": f"url(assets/{picks[0]['name']}.png)", "tag": "Fill · feature", "title": f"{picks[0]['name_a'].title()} and {picks[0]['name_b']}", "text": "A convnext feature inside the outline of a blob from vgg16."},
                {"bg": mesh, "tag": "Fill · colorway mesh", "title": "Its colours, blurred", "text": "The same feature's opponent pair as a mesh, in another blob."},
                {"bg": f"url(assets/{picks[1]['name']}.png)", "tag": "Fill · feature", "title": f"{picks[1]['name_a'].title()} and {picks[1]['name_b']}", "text": "Pictures carry the colour; the page stays two inks."},
                {"bg": f"url(assets/{picks[2]['name']}.png)", "tag": "Fill · feature", "title": f"{picks[2]['name_a'].title()} and {picks[2]['name_b']}", "text": "↻ variant swaps the blob for another."},
            ]
            hero, label = f"url(assets/{picks[1]['name']}.png)", "A convnext feature inside the outline of a lobed band from vgg16."
        art = {"hero": hero, "hero_label": label, "cards": cards}
        checks = [(lab, contrast(fg, bg), need) for lab, fg, bg, need in (
            ("body text on ground", t["ink"], t["paper"], 4.5), ("text on cards and fields", t["ink"], t["card_bg"], 4.5),
            ("primary button label", t["on_accent"] if t["dark"] else t["paper"], t["accent"] if t["dark"] else t["ink"], 4.5),
            ("primary button label, hover", t["paper"] if t["dark"] else t["on_accent"], t["ink"] if t["dark"] else t["accent"], 4.5),
            ("muted text on ground", t["muted"], t["paper"], 4.5), ("muted text on cards", t["muted"], t["card_bg"], 4.5),
            ("error text on ground", t["danger"], t["paper"], 4.5), ("focus ring on ground", t["focus"], t["paper"], 3.0),
            ("drop cap on ground (large)", t["focus"], t["paper"], 3.0))]
        bad = [c for c in checks if c[1] < c[2]]
        print(f"  {name:10s} " + "  ".join(f"{r:.1f}" for _, r, _ in checks) + ("   FAIL: " + ", ".join(c[0] for c in bad) if bad else "   all pass"), flush=True)
        (OUT / f"{name}.html").write_text(page(name, t, art, checks))
        if screenshot:
            out = OUT / f"{name}.png"
            with tempfile.TemporaryDirectory() as prof:
                subprocess.run([ST.CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", f"--user-data-dir={prof}",
                                "--virtual-time-budget=5000", "--window-size=1280,4600", f"--screenshot={out}", (OUT / f"{name}.html").as_uri()],
                               capture_output=True, timeout=90)
            print(f"  screenshot {out.name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screenshot", action="store_true")
    build(ap.parse_args().screenshot)
