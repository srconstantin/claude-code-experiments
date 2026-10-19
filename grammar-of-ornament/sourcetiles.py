#!/usr/bin/env python3
"""
Grammar of Ornament — source tiles: one style tile per source of shapes, each
built only from real pieces in designlib/atlas/ (made by atlas.py).

  bands     the band and blob compounds. Buttons, alerts and badges are band
            outlines; pictures are clipped to blob outlines; lines are a band
            repeated; checks, dots and bullets are blobs.
  marks     hatching, crosshatching, combs, sprigs, loops, feathers and
            branching lines. Buttons and alerts are the silhouettes of crosshatch
            clusters and hatching bundles, cards their mostly convex versions; lines are a feather,
            comb or hatching bundle repeated; checks and bullets are
            crosshatch clusters, the radio ring is a curled loop, the spinner
            a comb.
  strokes   the strokelib families. Buttons, cards and alerts are framed by
            four real tapered strokes; lines are a wave stroke repeated;
            the check is a hook, the radio ring a loop stroke, bullets are
            tapered dashes.

Pieces are used whole, never with ornament attached to an outline; a line is
one piece repeated with its ends overlapping. The examples on each page (hero
and card pictures) come from vgg16motifs/, linework/, the features folders
(vgg16, resnet50, convnext_base) and the convnext colorways, and each card
names its source.

Shares shapes.js (the shapes) and app.js (specimen toolbars: ↻ variant, Save,
SVG) with the earlier tiles. Every text and background pair is checked
against WCAG contrast and listed at the foot of each page.

Run:  python3 atlas.py            # if the atlas is missing or stale
      python3 sourcetiles.py [--screenshot]   # → designlib/tiles/{bands,marks,strokes}.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

import styletiles as ST
from styletiles import contrast, lab_hex, push_to, sp, to_lab

HERE = Path(__file__).resolve().parent
OUT = HERE / "designlib" / "tiles"
ATLAS = HERE / "designlib" / "atlas" / "atlas.json"
PAGES = ("bands", "marks", "strokes")


# ─────────────────────────────────────────────────────────────────────────────
#  Atlas access
# ─────────────────────────────────────────────────────────────────────────────

NUM = re.compile(r"(-?\d+\.?\d*),(-?\d+\.?\d*)")


def _pts(d: str) -> np.ndarray:
    return np.array([[float(a), float(b)] for a, b in NUM.findall(d)])


def _area(d: str) -> float:
    tot = 0.0
    for ring in d.split("M")[1:]:
        p = _pts("M" + ring)
        if len(p) >= 3:
            x, y = p[:, 0], p[:, 1]
            tot += abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    return tot


def _normalized(d: str) -> dict:
    p = _pts(d)
    x0, y0 = p.min(0)
    W, H = p.max(0) - p.min(0)
    nd = NUM.sub(lambda m: f"{float(m.group(1)) - x0:.2f},{float(m.group(2)) - y0:.2f}", d)
    return {"w": round(float(W), 2), "h": round(float(H), 2), "d": nd, "solidity": round(_area(d) / max(W * H, 1e-6), 3)}


@lru_cache(maxsize=1)
def pieces() -> list[dict]:
    return json.loads(ATLAS.read_text())["pieces"]


def pool(spec: str) -> list[dict]:
    """Pieces for a spec: 'comb', 'comb+feather', 'stroke:5' (a strokelib family),
    'loop:round' (curled loops), 'long:band' (long pieces of a group, longest first)."""
    out = []
    for part in spec.split("+"):
        if part.startswith("stroke:"):
            fams = {int(f) for f in part[7:].split(",")}
            out += [p for p in pieces() if p["group"] == "stroke" and p.get("family") in fams]
        elif part == "loop:round":   # the roundest loops first
            out += sorted([p for p in pieces() if p["group"] == "loop"], key=lambda p: abs(np.log(max(p["aspect"], 1e-3))))[:4]
        elif part.startswith("long:"):
            g = pool(part[5:])
            out += sorted([p for p in g if p["aspect"] >= 3.5] or g, key=lambda p: -p["aspect"])
        else:
            out += [p for p in pieces() if p["group"] == part]
    return out


def inscribed(d: str, W: float, H: float, px: int = 160, margin: float = 0.03) -> list[float]:
    """The largest axis-aligned rectangle inside an outline, as fractions
    [x0, y0, x1, y1] of its box, pulled in by `margin`. Because shapes are
    scaled to fit an element, the fractions hold at any size: shapes.js sizes
    a shape so this rectangle covers the element's text box."""
    from PIL import ImageDraw
    sx = px / max(W, H)
    w, h = max(2, int(W * sx)), max(2, int(H * sx))
    im = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(im)
    for ring in d.split("M")[1:]:
        p = _pts("M" + ring)
        if len(p) >= 3:
            dr.polygon([(x * sx, y * sx) for x, y in p], fill=255)
    m = np.asarray(im) > 0
    best, box = 0, (0, 0, w, h)
    heights = np.zeros(w, int)
    for y in range(h):                      # largest rectangle in a histogram, row by row
        heights = np.where(m[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            hh = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= hh:
                sx0, sh = stack.pop()
                if sh * (x - sx0) > best:
                    best, box = sh * (x - sx0), (sx0, y - sh + 1, x, y + 1)
                start = sx0
            stack.append((start, hh))
    x0, y0, x1, y1 = box
    fx0, fy0, fx1, fy1 = x0 / w, y0 / h, x1 / w, y1 / h
    mx, my = margin * (fx1 - fx0), margin * 2 * (fy1 - fy0)
    return [round(fx0 + mx, 4), round(fy0 + my, 4), round(fx1 - mx, 4), round(fy1 - my, 4)]


def shape_set(spec: str, key: str, aspect: tuple[float, float], solidity: float, convexity: float = 0.0,
              inner: tuple[float, float] = (0.0, 0.0)) -> list[dict]:
    """Outlines ('outline' or 'hull') from a pool, normalized to their own bbox
    and filtered to a range of aspect and a minimum solidity."""
    out = []
    for p in pool(spec):
        d = p.get(key) or ""
        if not d:
            continue
        n = _normalized(d)
        if aspect[0] <= n["w"] / max(n["h"], 1e-6) <= aspect[1] and n["solidity"] >= solidity and p.get("convexity", 1.0) >= convexity:
            ins = inscribed(n["d"], n["w"], n["h"]) if key != "ink" else [0, 0, 1, 1]
            # for shapes that must hold text: the largest inner rectangle must span most of the shape,
            # so fitting it to the text never blows the shape up
            if ins[2] - ins[0] >= inner[0] and ins[3] - ins[1] >= inner[1]:
                out.append({"id": p["id"], **n, "inset": ins})
    return out


def piece_uri(p: dict, ink: str, relief: str, mode: str = "glyph", use: str = "ink", stroke: float = 0.0, rotate: bool = False) -> str:
    pad = 0.6
    W, H = p["w"] + 2 * pad, p["h"] + 2 * pad
    d = p.get(use) or p["ink"]
    if stroke:
        sw = stroke * max(p["w"], p["h"]) / 24
        body = f'<path fill="none" stroke="{ink}" stroke-width="{sw:.2f}" stroke-linejoin="round" d="{d}"/>'
    else:
        body = f'<path fill="{ink}" d="{d}"/>' + (f'<path fill="{relief}" d="{p["relief"]}"/>' if use == "ink" and p["relief"] else "")
    g = f'<g transform="translate({pad},{pad})">{body}</g>'
    if rotate:
        g = f'<g transform="translate({H:.2f},0) rotate(90)">{g}</g>'
        W, H = H, W
    par = "none" if mode == "stretch" else "xMidYMid meet"
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.2f} {H:.2f}" preserveAspectRatio="{par}">{g}</svg>'
    return f'url("{ST.O.data_uri(svg)}")'


def strip_uri(p: dict, ink: str, relief: str, overlap: float = 0.06, H: float = 10.0) -> str:
    """One long piece repeated: the tile is a little shorter than the piece and
    copies a period either side fill the overlap, so each repeat runs into the next."""
    s = H / max(p["h"], 1e-6)
    P = p["w"] * s * (1 - overlap)
    g = lambda x: (f'<g transform="translate({x:.2f},0) scale({s:.4f})"><path fill="{ink}" d="{p["ink"]}"/>'
                   + (f'<path fill="{relief}" d="{p["relief"]}"/>' if p["relief"] else "") + "</g>")
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {P:.2f} {H:.2f}" preserveAspectRatio="xMinYMid meet">{g(-P)}{g(0)}{g(P)}</svg>'
    return f'url("{ST.O.data_uri(svg)}")'


def top_profile(d: str, W: float, H: float, cols: int = 240) -> np.ndarray:
    """The top silhouette of a piece: for each column, how far down the first
    ink is, 0 at the piece's highest point and 1 at its lowest; gaps between
    marks are bridged, the result lightly smoothed."""
    from PIL import ImageDraw
    from scipy import ndimage as ndi
    s = cols / W
    h = max(4, int(H * s))
    im = Image.new("L", (cols, h), 0)
    dr = ImageDraw.Draw(im)
    for ring in d.split("M")[1:]:
        p = _pts("M" + ring)
        if len(p) >= 3:
            dr.polygon([(x * s, y * s) for x, y in p], fill=255)
    m = np.asarray(im) > 0
    top = np.where(m.any(0), m.argmax(0), np.nan).astype(float)
    ok = ~np.isnan(top)
    if ok.sum() < 2:
        return np.zeros(cols)
    top = np.interp(np.arange(cols), np.nonzero(ok)[0], top[ok])
    top = ndi.gaussian_filter1d(top, 1.2)
    return (top - top.min()) / max(np.ptp(top), 1e-6)


def edge_uri(p: dict, colour: str, use: str = "ink", amp: float = 0.75, H: float = 10.0, stretch: float = 1.0) -> str:
    """A seamless edge: the area below a piece's top silhouette, filled with the
    colour of the band it tops. The silhouette is tilted so its two ends meet
    at the same height, so repeats join without a step."""
    prof = top_profile(p.get(use) or p["ink"], p["w"], p["h"])
    n = len(prof)
    prof = prof - np.linspace(0, 1, n) * (prof[-1] - prof[0])
    prof = (prof - prof.min()) / max(np.ptp(prof), 1e-6)
    Wt = H * max(p["w"] / max(p["h"], 1e-6), 3.0) * stretch
    xs = np.linspace(0, Wt, n)
    ys = H * (1 - amp) * 0.5 + H * amp * prof
    pts = " L".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wt:.2f} {H:.2f}" preserveAspectRatio="none">'
           f'<path fill="{colour}" d="M0,{H} L{pts} L{Wt:.2f},{H} Z"/></svg>')
    return f'url("{ST.O.data_uri(svg)}")'


def frame_glyph(ps: list[dict], ink: str, T: float = 1.4, fill: str | None = None) -> str:
    """A small square drawn with four strokes (the strokes page's checkbox);
    filled, the same square solid inside its strokes."""
    S = 10.0
    parts = [f'<rect x="0" y="0" width="{S}" height="{S}" fill="{fill}"/>'] if fill else []
    for i, p in enumerate(ps[:4]):
        sx, sy = (S * 1.08) / p["w"], T / p["h"]
        if i == 0:
            tf = f"translate({-0.04 * S:.2f},{-T / 2:.2f}) scale({sx:.4f},{sy:.4f})"
        elif i == 1:
            tf = f"translate({-0.04 * S:.2f},{S - T / 2:.2f}) scale({sx:.4f},{sy:.4f})"
        elif i == 2:
            tf = f"translate({T / 2:.2f},{-0.04 * S:.2f}) rotate(90) scale({sx:.4f},{sy:.4f})"
        else:
            tf = f"translate({S + T / 2:.2f},{-0.04 * S:.2f}) rotate(90) scale({sx:.4f},{sy:.4f})"
        parts.append(f'<path fill="{ink}" transform="{tf}" d="{p["ink"]}"/>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-T:.2f} {-T:.2f} {S + 2 * T:.2f} {S + 2 * T:.2f}">{"".join(parts)}</svg>'
    return f'url("{ST.O.data_uri(svg)}")'


# ─────────────────────────────────────────────────────────────────────────────
#  Pages
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def themes() -> dict:
    eng = ST.themes()["engraving"]
    pairs = ST.pair_ends(HERE / "colorways" / "convnext_basefeatures" / "summary.csv")
    bo = pairs[("blue", "orange")]
    T = {
        "bands": dict(title="Bands", source="the band and blob compounds",
                      blurb="Every shape on this page is the outline of a real band or blob found in a vgg16 feature: lumpy bands for buttons and alerts, blobs for pictures, a band repeated for each line.",
                      ink=lab_hex(92, 2, 12), paper=lab_hex(14, -16, 2), accent=lab_hex(78, 6, 58), card_bg=lab_hex(19, -16, 2), field_bg=lab_hex(18, -15, 2),
                      muted=lab_hex(70, -8, 8), danger=lab_hex(68, 46, 30), dark=True,
                      head="'Young Serif', Georgia, serif", head_weight="400", body="'Figtree', 'Helvetica Neue', sans-serif", ui="'Figtree', 'Helvetica Neue', sans-serif",
                      fonts="Young+Serif&family=Figtree:wght@400;500;600;700", line="1.4px", hair="1px"),
        "marks": dict(title="Marks", source="hatching, crosshatching, combs, sprigs, loops, feathers and branching lines",
                      blurb="Every shape here comes from clusters of marks: buttons, cards and alerts are the silhouettes of crosshatch clusters and hatching bundles, lines are a feather or comb repeated, checks are crosshatch.",
                      ink=eng["ink"], paper=eng["paper"], accent=eng["accent"], card_bg=eng["card_bg"], field_bg=eng["field_bg"],
                      muted=eng["muted"], danger=eng["danger"], dark=False,
                      head="'Cormorant Garamond', Garamond, serif", head_weight="600", body="'EB Garamond', Garamond, serif", ui="'EB Garamond', Garamond, serif",
                      fonts="Cormorant+Garamond:wght@500;600;700&family=EB+Garamond:wght@400;500;600", line="1.5px", hair="1px"),
        "strokes": dict(title="Strokes", source="the strokelib stroke families",
                        blurb="Every shape here is drawn with single strokes from the stroke library: frames of four tapered strokes, a wave repeated for each line, a hook for the check, a loop for the radio.",
                        ink=lab_hex(20, bo["blue"][1] * 1.1, bo["blue"][2] * 1.2), paper="#fbfaf7", accent=lab_hex(52, bo["orange"][1] * 1.3, bo["orange"][2] * 1.15),
                        card_bg="#ffffff", field_bg="#ffffff", muted=lab_hex(45, 2, -10), danger=lab_hex(42, 62, 40), dark=False,
                        head="'Newsreader', Georgia, serif", head_weight="500", body="'Work Sans', 'Helvetica Neue', sans-serif", ui="'Work Sans', 'Helvetica Neue', sans-serif",
                        fonts="Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Work+Sans:wght@400;500;600;700", line="1.4px", hair="1px"),
    }
    for t in T.values():
        finish(t)
    return T


def finish(t: dict) -> dict:
    """Derived colours, each pushed until its contrast check passes."""
    t["ink"] = push_to(t["ink"], t["paper"], 7.0)
    t["muted"] = push_to(push_to(t["muted"], t["paper"], 4.6), t["card_bg"], 4.6)
    t["danger"] = push_to(t["danger"], t["paper"], 4.6)
    t["on_accent"] = t["ink"] if contrast(t["ink"], t["accent"]) > contrast(t["paper"], t["accent"]) else t["paper"]
    t["accent"] = push_to(t["accent"], t["on_accent"], 4.6)   # labels on the accent (hover buttons) must read
    if t["dark"]:
        t["accent"] = push_to(t["accent"], t["paper"], 4.6)   # the accent is also a label colour on dark pages
    t["focus"] = push_to(t["accent"], t["paper"], 3.2)
    pl, il = to_lab(t["paper"]), to_lab(t["ink"])
    t["seg"] = lab_hex(pl[0] + (5 if t["dark"] else -5), pl[1] + (il[1] - pl[1]) * 0.07, pl[2] + (il[2] - pl[2]) * 0.07)
    for k in ("ink", "muted", "danger"):
        t[k] = push_to(t[k], t["seg"], 4.6 if k != "ink" else 7.0)
    L = to_lab(t["ink"])[0]
    t["ink_hover"] = lab_hex(L + (12 if L < 50 else -14), *to_lab(t["ink"])[1:])
    return t


@lru_cache(maxsize=1)
def colorway_options(n: int = 10) -> list[dict]:
    """A spread of convnext colorways: the most vivid of each hue pair, then the next most vivid."""
    rows = list(csv.DictReader(open(HERE / "colorways" / "convnext_basefeatures" / "summary.csv")))
    vivid = lambda r: min(float(r["chroma_a"]), float(r["chroma_b"]))
    rows.sort(key=vivid, reverse=True)
    out, pairs = [], set()
    for r in rows:
        key = tuple(sorted((r["name_a"], r["name_b"])))
        if key not in pairs:
            pairs.add(key)
            out.append(r)
    out += [r for r in rows if r not in out][: max(0, n - len(out))]
    return out[:n]


def theme_from_colorway(page: str, name: str) -> dict:
    """The page's theme, recoloured from one feature's opponent pair: on light
    pages the darker end becomes the ink and the other the accent; on dark
    pages the darker end becomes the ground."""
    base = dict(themes()[page])
    cw = json.loads((HERE / "colorways" / "convnext_basefeatures" / f"{name}.json").read_text())["pair"]
    ends = sorted((np.array(cw["lab_a"]), np.array(cw["lab_b"])), key=lambda v: v[0])
    lo, hi = ends
    if base["dark"]:
        t = dict(base, paper=lab_hex(13, lo[1] * 0.4, lo[2] * 0.4), card_bg=lab_hex(18, lo[1] * 0.35, lo[2] * 0.35),
                 field_bg=lab_hex(17, lo[1] * 0.35, lo[2] * 0.35), ink=lab_hex(93, hi[1] * 0.12, hi[2] * 0.12),
                 accent=lab_hex(78, hi[1], hi[2]), muted=lab_hex(72, hi[1] * 0.12, hi[2] * 0.12), danger=lab_hex(70, 48, 30))
    else:
        t = dict(base, paper=lab_hex(96, hi[1] * 0.07, hi[2] * 0.07), card_bg=lab_hex(98.5, hi[1] * 0.03, hi[2] * 0.03),
                 field_bg=lab_hex(98.5, hi[1] * 0.03, hi[2] * 0.03), ink=lab_hex(18, lo[1] * 0.7, lo[2] * 0.7),
                 accent=lab_hex(58, hi[1] * 1.1, hi[2] * 1.1), muted=lab_hex(44, lo[1] * 0.25, lo[2] * 0.25), danger=lab_hex(42, 60, 42))
    return finish(t)


# shapes.js shape sets per page: role → (pool, key, aspect range, min solidity)
SHAPES = {
    "bands": {"button": ("band", "outline", (2.4, 7), 0.64), "card": ("band+blob", "convex", (0.5, 6.0), 0.70, 0.95),
              "blob": ("blob", "outline", (0.55, 1.9), 0.5), "alert": ("band", "outline", (2.0, 9), 0.6, 0.0, (0.55, 0.45))},
    "marks": {"button": ("hatching+crosshatch+comb+feather", "hull", (2.2, 6), 0.62), "card": ("crosshatch+comb+feather+sprig+branch+hatching", "convex", (0.5, 6.0), 0.70, 0.95),
              "alert": ("hatching+crosshatch+comb+feather", "hull", (2.0, 9), 0.58, 0.0, (0.55, 0.45))},
    "strokes": {"frame": ("stroke:1,2,3,8", "ink", (2.5, 40), 0.0)},
}

# ornaments: var → (pool, index, colour, mode, options)   mode: strip | glyph | stretch
SPECS = {
    "bands": {
        "rule": ("long:band", 0, "accent", "strip", {}), "underline": ("long:band", 3, "muted", "strip", {}),
        "underline_focus": ("long:band", 3, "accent", "strip", {}), "underline_error": ("long:band", 5, "danger", "strip", {}),
        "tab": ("long:band", 1, "accent", "strip", {}), "divider": ("long:band", 2, "accent", "strip", {}), "edge": ("long:band", 0, "paper", "edge", {"amp": 0.8}), "edge_alt": ("long:band", 2, "seg", "edge", {"amp": 0.8}),
        "box": ("blob", 7, "ink", "stretch", {"use": "outline", "stroke": 1.0}), "box_filled": ("blob", 7, "accent", "stretch", {"use": "outline"}),
        "ring": ("blob", 9, "ink", "stretch", {"use": "outline", "stroke": 1.0}), "ring_filled": ("blob", 9, "accent", "stretch", {"use": "outline"}),
        "knob": ("blob", 6, "paper", "glyph", {}), "spinner": ("blob", 2, "accent", "glyph", {}),
        "bullet": ("blob", 12, "accent", "glyph", {}), "bullet2": ("blob", 15, "accent", "glyph", {}), "bullet3": ("blob", 18, "accent", "glyph", {}),
        "bar": ("long:band", 0, "accent", "stretch", {}),
    },
    "marks": {
        "rule": ("long:hatching", 0, "ink", "strip", {}), "underline": ("long:feather", 1, "ink", "strip", {}),
        "underline_focus": ("long:feather", 1, "focus", "strip", {}), "underline_error": ("long:comb", 0, "danger", "strip", {}),
        "tab": ("long:comb", 1, "ink", "strip", {}), "divider": ("long:feather", 0, "ink", "strip", {}), "edge": ("long:hatching", 0, "paper", "edge", {"use": "hull", "amp": 0.75}), "edge_alt": ("long:feather", 1, "seg", "edge", {"use": "hull", "amp": 0.75}),
        "box": ("crosshatch", 3, "ink", "stretch", {"use": "hull", "stroke": 1.1}), "box_filled": ("crosshatch", 3, "ink", "stretch", {"use": "hull"}),
        "ring": ("loop:round", 0, "ink", "glyph", {}), "ring_filled": ("loop:round", 0, "ink", "glyph", {"use": "outline"}),
        "knob": ("crosshatch", 20, "paper", "glyph", {}), "spinner": ("comb", 9, "ink", "glyph", {}),
        "bullet": ("crosshatch", 9, "ink", "glyph", {}), "bullet2": ("crosshatch", 15, "ink", "glyph", {}), "bullet3": ("crosshatch", 20, "ink", "glyph", {}),
        "bar": ("long:hatching", 1, "ink", "stretch", {}),
    },
    "strokes": {
        "rule": ("long:stroke:5", 0, "ink", "strip", {"overlap": 0.12}), "underline": ("long:stroke:5", 3, "muted", "strip", {"overlap": 0.12}),
        "underline_focus": ("long:stroke:5", 3, "accent", "strip", {"overlap": 0.12}), "underline_error": ("long:stroke:6", 0, "danger", "strip", {"overlap": 0.1}),
        "tab": ("long:stroke:3", 0, "accent", "strip", {"overlap": 0.15}), "divider": ("long:stroke:5", 6, "ink", "strip", {"overlap": 0.12}),
        "edge": ("long:stroke:5", 9, "paper", "edge", {"amp": 0.7, "stretch": 1.6}), "edge_alt": ("long:stroke:5", 3, "seg", "edge", {"amp": 0.7, "stretch": 1.6}),
        "ring": ("stroke:11", 0, "ink", "glyph", {}), "ring_filled": ("stroke:11", 0, "accent", "glyph", {"use": "outline"}),
        "knob": ("stroke:11", 5, "paper", "glyph", {}), "spinner": ("stroke:9", 0, "accent", "glyph", {}),
        "bullet": ("stroke:6", 0, "accent", "glyph", {}), "bullet2": ("stroke:8", 1, "accent", "glyph", {}), "bullet3": ("stroke:10", 2, "accent", "glyph", {}),
        "bar": ("long:stroke:3", 0, "ink", "stretch", {}),
    },
}


def render_var(page: str, var: str, seed: int | None = None, t: dict | None = None) -> str:
    t = t or themes()[page]
    if var in ("box", "box_filled") and page == "strokes":
        ps = pool("stroke:1,2,3,8")
        rng = np.random.default_rng(seed if seed is not None else 4)
        four = [ps[i] for i in rng.choice(len(ps), 4, replace=False)]
        return frame_glyph(four, t["ink"]) if var == "box" else frame_glyph(four, t["ink"], fill=t["accent"])
    spec, index, colour, mode, opts = SPECS[page][var]
    ps = pool(spec)
    p = ps[np.random.default_rng(seed).integers(len(ps))] if seed is not None else ps[index % len(ps)]
    relief = t["paper"] if t["dark"] else t["card_bg"]
    if mode == "edge":
        return edge_uri(p, t[colour], **opts)
    if mode == "strip":
        return strip_uri(p, t[colour], relief, **opts)
    return piece_uri(p, t[colour], relief, mode, **opts)


def root_vars(name: str, t: dict) -> dict:
    """Every colour and ornament of a page as CSS custom properties."""
    dark = t["dark"]
    v = {
        "ink": t["ink"], "paper": t["paper"], "accent": t["accent"], "card-bg": t["card_bg"], "field-bg": t["field_bg"],
        "muted": t["muted"], "danger": t["danger"], "ink-hover": t["ink_hover"], "focus": t["focus"], "on-ink": t["paper"],
        "on-accent": t["on_accent"], "knob-bg": t["accent"] if dark else t["ink"], "toggle-on": t["accent"] if dark else t["ink"],
        "drop": t["focus"], "hover-fill": t["ink"] if dark else t["accent"], "hover-ink": t["paper"] if dark else t["on_accent"],
        "seg": t["seg"], "frame-line": t["accent"] if dark else t["ink"], "solid-fill": t["accent"] if dark else t["ink"], "solid-ink": t["on_accent"] if dark else t["paper"],
    }
    for var in list(SPECS[name]) + (["box", "box_filled"] if name == "strokes" else []):
        v[var.replace("_", "-")] = render_var(name, var, t=t)
    return v


def palette_vars(page: str, colorway: str) -> dict:
    """The page recoloured from a colorway (used by server.py)."""
    t = themes()[page] if colorway in ("", "default") else theme_from_colorway(page, colorway)
    return root_vars(page, t)


def variant_vars(page: str, names: list[str], seed: int) -> dict:
    out = {}
    for i, n in enumerate(names):
        key = n.replace("-", "_")
        if key in SPECS[page] or (key in ("box", "box_filled") and page == "strokes"):
            base = key.replace("_filled", "")      # a shape and its filled twin get the same piece
            out[n] = render_var(page, key, seed + 7919 * sum(map(ord, base)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Examples (hero and card pictures) from the project's folders
# ─────────────────────────────────────────────────────────────────────────────

def recolor_svg(src: Path, ink: str, ground: str, out: Path) -> None:
    s = src.read_text()
    s = s.replace("#000", "@@INK@@").replace("#fff", ground).replace("@@INK@@", ink)
    out.write_text(s)


def vivid(folder: Path, n: int = 1, skip: int = 0) -> list[Path]:
    """The most colourful images in a features folder (by mean chroma)."""
    from skimage import color
    scored = []
    for p in sorted(folder.glob("*.png")):
        rgb = np.asarray(Image.open(p).convert("RGB").resize((64, 64))).astype(np.float32) / 255
        lab = color.rgb2lab(rgb)
        scored.append((float(np.hypot(lab[..., 1], lab[..., 2]).mean()), p))
    scored.sort(reverse=True)
    return [p for _, p in scored[skip:skip + n]]


def short(name: str) -> str:
    m = re.match(r"([a-z0-9_]+)__([^_]+(?:\.[0-9]+)*)__c(\d+)", name)
    return f"{m.group(1)} · {m.group(2)} c{m.group(3)}" if m else name


def examples(page: str, t: dict, assets: Path) -> dict:
    rows = list(csv.DictReader(open(HERE / "compounds" / "vgg16features" / "by_feature.csv")))
    score = lambda r, words: sum(int(v) for k, v in r.items() if any(w in k for w in words))
    combiest = max(rows, key=lambda r: score(r, ("comb", "feather")))["feature"]
    bandiest = max(rows, key=lambda r: score(r, ("band",)))["feature"]
    hatchiest = max(rows, key=lambda r: score(r, ("crosshatching",)))["feature"]
    motifs = sorted((HERE / "vgg16motifs").glob("*/*/motif.json"), key=lambda p: -json.loads(p.read_text())["width"])
    cw = list(csv.DictReader(open(HERE / "colorways" / "convnext_basefeatures" / "summary.csv")))
    vivid_cw = lambda pair: max((r for r in cw if {r["name_a"], r["name_b"]} == pair), key=lambda r: min(float(r["chroma_a"]), float(r["chroma_b"])))
    cwj = lambda r: json.loads((HERE / "colorways" / "convnext_basefeatures" / f"{r['name']}.json").read_text())
    dark, light = sorted((t["ink"], t["card_bg"]), key=ST.luminance)

    def copy_up(name: str, folder: str) -> str:
        up = ST.upscaled_feature(name, HERE / folder)
        shutil.copyfile(up, assets / f"{name}.png")
        return f"url(assets/{name}.png)"

    def motif(i: int, ink: str) -> str:
        out = assets / f"{page}-motif-{i}.png"
        ST.ink_motif(motifs[i].parent / "motif.png", ink, out)
        return f"url(assets/{out.name})"

    def linework(name: str, kind: str, ink: str, ground: str) -> str:
        out = assets / f"{page}-{kind}-{name}.svg"
        recolor_svg(HERE / "linework" / "vgg16features" / f"{name}.{kind}.svg", ink, ground, out)
        return f"url(assets/{out.name})"

    def duo(name: str, folder: str) -> str:
        out = assets / f"{page}-duotone-{name}.png"
        ST.duotone(ST.upscaled_feature(name, HERE / folder), dark, light, out)
        return f"url(assets/{out.name})"

    res = vivid(HERE / "resnet50features", 2)
    conv = vivid(HERE / "convnext_basefeatures", 2)
    if page == "bands":
        pm, gp = vivid_cw({"purple", "yellow"}), vivid_cw({"green", "purple"})
        hero = (f"{motif(0, t['accent'])} center/contain no-repeat", f"A motif from vgg16motifs ({short(motifs[0].parent.parent.name)}), printed in gold.")
        cards = [
            (copy_up(bandiest, "vgg16features"), f"vgg16features · {short(bandiest)}", "The feature richest in bands", "Upscaled, inside the outline of a blob."),
            (copy_up(res[0].stem, "resnet50features"), f"resnet50features · {short(res[0].stem)}", "A resnet50 feature", "The most colourful of the resnet50 sample."),
            (cwj(pm)["mesh"]["css"], f"colorways · {short(pm['name'])}", f"{pm['name_a'].title()} against {pm['name_b']}", "A convnext colorway as a mesh."),
            (copy_up(gp["name"], "convnext_basefeatures"), f"convnext_basefeatures · {short(gp['name'])}", "A convnext feature", f"{gp['name_a'].title()} and {gp['name_b']}."),
            (motif(1, t["ink"]) + " center/contain no-repeat " + t["card_bg"], f"vgg16motifs · {short(motifs[1].parent.parent.name)}", "A motif", "Printed in cream on the card."),
        ]
    elif page == "marks":
        py = vivid_cw({"purple", "yellow"})
        hero = (f"{linework(combiest, 'outlines', t['ink'], t['card_bg'])} center/cover no-repeat",
                f"linework · the outlines of {short(combiest)}: its combs, feathers and hatching, in the page's ink.")
        cards = [
            (linework(hatchiest, "strokes", t["ink"], t["card_bg"]), f"linework · strokes of {short(hatchiest)}", "The crosshatch feature", "Its strokes as centerlines, inside the mostly convex outline of a mark cluster."),
            (duo(combiest, "vgg16features"), f"vgg16features · {short(combiest)}", "The comb feature", "As a duotone in violet and cream."),
            (cwj(py)["gradient"]["image_css"], f"colorways · {short(py['name'])}", "A gradient through a feature", "Seven stops along its opponent axis."),
            (motif(2, t["ink"]) + " center/contain no-repeat " + t["card_bg"], f"vgg16motifs · {short(motifs[2].parent.parent.name)}", "A motif", "Cut from a print, in the page's ink."),
            (copy_up(hatchiest, "vgg16features"), f"vgg16features · {short(hatchiest)}", "The feature itself", "Upscaled, in its own colours."),
        ]
    else:
        bo = vivid_cw({"blue", "orange"})
        hero = (f"{linework(bandiest, 'strokes', t['ink'], t['paper'])} center/cover no-repeat",
                f"linework · the strokes of {short(bandiest)}, each mark drawn at its own width, in the page's blue.")
        cards = [
            (copy_up(res[1].stem, "resnet50features"), f"resnet50features · {short(res[1].stem)}", "A resnet50 feature", "Inside a frame of four strokes."),
            (copy_up(conv[0].stem, "convnext_basefeatures"), f"convnext_basefeatures · {short(conv[0].stem)}", "A convnext feature", "The most colourful of the convnext sample."),
            (cwj(bo)["gradient"]["pair_css"], f"colorways · {short(bo['name'])}", f"{bo['name_a'].title()} to {bo['name_b']}", "A colorway's pure pair, in Lab."),
            (linework(combiest, "outlines", t["ink"], t["card_bg"]), f"linework · outlines of {short(combiest)}", "The comb feature, drawn", "Its marks as variable-width outlines."),
            (motif(3, t["ink"]) + " center/contain no-repeat " + t["card_bg"], f"vgg16motifs · {short(motifs[3].parent.parent.name)}", "A motif", "In the page's blue."),
        ]
    fill = {"bands": f"assets/{bandiest}.png", "marks": f"assets/{combiest}.png", "strokes": f"assets/{res[1].stem}.png"}[page]
    copy_up(fill[len("assets/"):-len(".png")], "vgg16features" if "vgg16" in fill else "resnet50features")
    bw = []
    for nm in (combiest, bandiest, hatchiest):
        out = assets / f"bw-outlines-{nm}.svg"
        shutil.copyfile(HERE / "linework" / "vgg16features" / f"{nm}.outlines.svg", out)
        bw.append((f"assets/{out.name}", f"linework outlines · {short(nm)}"))
    grid = []
    if page == "bands":
        vg = sorted({r["feature"] for r in rows}, key=lambda n: [int(x) if x.isdigit() else x for x in re.split(r"[._]", n)])
        picks = [(n, "vgg16features") for n in vg[:: max(1, len(vg) // 4)][:4]] + [(p.stem, "resnet50features") for p in vivid(HERE / "resnet50features", 2, 2)] \
            + [(p.stem, "convnext_basefeatures") for p in vivid(HERE / "convnext_basefeatures", 2, 2)]
        grid = [(copy_up(n, f), f, short(n)) for n, f in picks]
    return {"hero": hero, "cards": cards, "fill": (fill, "feature"), "bw": bw, "grid": grid}


# ─────────────────────────────────────────────────────────────────────────────
#  HTML
# ─────────────────────────────────────────────────────────────────────────────

CSS = r"""
svg.shape .stroke{fill:var(--shape-line,var(--ink))}
.btn{padding:18px 44px;font-weight:600;letter-spacing:.01em}
.btn.primary::after{display:none}
.btn.primary:hover,.btn.primary.is-hover{--shape-fill:var(--hover-fill);color:var(--hover-ink)}
.btn.quiet{background:var(--underline) left bottom/auto 11px repeat-x}
.btn.quiet:hover,.btn.quiet.is-hover{background-image:var(--underline-focus)}
.field input,.field select{background:var(--underline) left bottom/auto 11px repeat-x,var(--field-bg);padding-bottom:18px}
.field select{background:var(--underline) left bottom/auto 11px repeat-x,var(--field-bg)}
.field input:focus,.field.is-focus input{background:var(--underline-focus) left bottom/auto 13px repeat-x,var(--field-bg)}
.field.is-error input{background:var(--underline-error) left bottom/auto 13px repeat-x,var(--field-bg)}
.choice input[type=checkbox]{border:0;border-radius:0;background:var(--box) center/100% 100% no-repeat;width:26px;height:26px}
.choice input[type=checkbox]:checked{background:var(--box-filled) center/100% 100% no-repeat}
.choice input[type=radio]{background:var(--ring) center/100% 100% no-repeat;width:26px;height:26px}
.choice input[type=radio]:checked{background:var(--ring-filled) center/100% 100% no-repeat}
.toggle .track{border-radius:999px}
.toggle input:checked+.track{background:var(--toggle-on)}
.toggle input:checked+.track::after{background:var(--knob) center/100% no-repeat,var(--knob-bg)}
.tabs{box-shadow:none}
.tab[aria-selected=true]::after{left:-4px;right:-6px;height:12px;bottom:-6px;background:var(--tab) left center/auto 100% repeat-x}
section>h2{background:var(--rule) left bottom/auto 12px repeat-x;padding-bottom:24px}
.divider{height:52px;background:var(--divider) left center/auto 16px repeat-x}
.cards{gap:56px 40px}
.card{--shape-fill:transparent;--shape-line:transparent;--shape-stroke:0;background:none}
.card .art{height:230px;--shape-fill:transparent;--shape-line:var(--frame-line);--shape-stroke:var(--line)}
.card .body{padding:16px 6px 0}
.specimen[data-kind=alert]{width:100%;align-items:center;padding:58px 12px}
.specimen[data-kind=alert]>figcaption{margin-top:52px}
.alert{display:block;padding:6px 10px;margin:0;max-width:380px;--shape-fill:var(--card-bg);--shape-line:var(--frame-line);--shape-stroke:var(--hair)}
.alert.error{--shape-line:var(--danger)}
.alert.error strong{color:var(--danger)}
.progress .track{height:16px;border-radius:999px;background:var(--field-bg)}
.progress .bar{background:var(--bar) left center/100% 100% no-repeat}
.progress .ticks{display:none}
.spinner{width:64px;height:64px}
.badge{padding:10px 26px;--shape-fill:var(--card-bg);--shape-line:var(--frame-line);--shape-stroke:var(--hair)}
.badge::before{display:none}
.badge.solid{--shape-fill:var(--solid-fill);color:var(--solid-ink);padding-left:26px}
.prose li{padding-left:34px;background:var(--bullet) 0 .15em/22px 22px no-repeat}
.prose li:nth-child(2){background-image:var(--bullet2)}.prose li:nth-child(3){background-image:var(--bullet3)}
.hero .art{margin:28px 0;--shape-fill:transparent;--shape-line:var(--frame-line);--shape-stroke:var(--line)}
.hero .art[data-shape]{background:var(--hero-art)!important}
.hero .art:not([data-shape]){background:var(--hero-art)}
.card .tag{text-transform:none;letter-spacing:.02em;font-size:13px}
.index-note{font:14px var(--ui);color:var(--muted)}

/* view panel */
.view{position:sticky;top:0;z-index:50;display:flex;flex-wrap:wrap;gap:8px 22px;align-items:center;padding:8px 32px;font:13px/1.3 system-ui,sans-serif;
  background:#f7f6f2;color:#222;border-bottom:1px solid #d9d5cc}
.view select{font:inherit;margin-left:6px;max-width:300px}
.view .swatch{display:inline-block;width:28px;height:14px;margin-left:8px;vertical-align:middle;border-radius:2px}
.view.busy{opacity:.6;cursor:progress}
/* backgrounds: the whole page, or one or more full-width bands; text keeps a solid panel of the page's ground */
body{overflow-x:hidden}
body.bg-page{background:var(--bg-img) center/cover fixed,var(--paper)}
body.bg-page .switch,body.bg-page .caption{background:var(--paper)}
body.bg-page .hero .panel,body.bg-page .wrap>section{background:var(--paper)}
body.bg-page .wrap>section{padding:30px 34px;margin-left:-34px;margin-right:-34px}
body.bg-page .hero{padding-right:32px}
.bg-seg{position:relative;isolation:isolate}
.bg-seg::before{content:"";position:absolute;top:-34px;bottom:-34px;left:calc(50% - 50vw);right:calc(50% - 50vw);background:var(--bg-img) center/cover;z-index:-2}
.wrap>section.bg-seg{padding:66px 34px 62px;margin-left:-34px;margin-right:-34px}
.wrap>section.bg-seg::after{content:"";position:absolute;inset:0;background:var(--paper);z-index:-1}
.hero.bg-seg::before{top:0;bottom:0}
.hero.bg-seg .panel{background:var(--paper);margin:28px 0 28px 32px;padding-left:36px}
/* feature cards */
.fgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:30px 26px}
.fgrid>.specimen{align-items:stretch}
.fcard .art{height:150px;--shape-fill:transparent;--shape-line:var(--frame-line);--shape-stroke:var(--hair)}
.fcard .fbody{display:flex;flex-direction:column;padding:10px 4px 0;font:14px var(--ui)}
.fcard .fbody small{color:var(--muted);font-size:12px}

/* stacked bands with contour edges */
.hero{position:relative}
.wrap>section{position:relative;isolation:isolate;margin:0;padding:66px 0 62px}
.seg-paint{position:absolute;top:calc(-1 * var(--edge-h));bottom:0;left:calc(50% - 50vw);right:calc(50% - 50vw);z-index:-3;pointer-events:none;
  background:var(--edge) var(--edge-x,0) top/auto var(--edge-h) repeat-x,linear-gradient(var(--paper),var(--paper)) left var(--edge-h)/100% calc(100% - var(--edge-h) + 1px) no-repeat}
.wrap>section.alt>.seg-paint{background:var(--edge-alt) var(--edge-x,0) top/auto var(--edge-h) repeat-x,linear-gradient(var(--seg),var(--seg)) left var(--edge-h)/100% calc(100% - var(--edge-h) + 1px) no-repeat}
body.bg-page .seg-paint{display:none}
.caption{padding-bottom:calc(var(--edge-h) + 14px)}
.wrap>section.bg-seg::before{top:calc(-1 * var(--edge-h));bottom:0;
  -webkit-mask:var(--edge) var(--edge-x,0) top/auto var(--edge-h) repeat-x,linear-gradient(#000,#000) left var(--edge-h)/100% calc(100% - var(--edge-h) + 1px) no-repeat;
  mask:var(--edge) var(--edge-x,0) top/auto var(--edge-h) repeat-x,linear-gradient(#000,#000) left var(--edge-h)/100% calc(100% - var(--edge-h) + 1px) no-repeat}
.wrap>section.bg-seg::after{inset:40px 0 36px}
.wrap{padding-bottom:0}
/* frames of strokes: the strokes sit above a picture */
.card .art[data-shape=frame]>svg.shape,.hero .art[data-shape=frame]>svg.shape{z-index:2}
"""

PAGE_CSS = {
    "bands": """
.btn{color:var(--on-accent);--shape-fill:var(--accent)}
.btn.secondary{color:var(--accent);--shape-line:var(--accent)}
""",
    "marks": """
""",
    "strokes": """
.btn{padding:17px 40px}
.btn.secondary{--shape-line:var(--ink)}
.btn.quiet{padding-bottom:18px}
""",
}


def page(name: str, t: dict, ex: dict, checks: list) -> str:
    shp = {}
    for role, args in SHAPES[name].items():
        shp[role] = shape_set(*args)
    kind = "frame" if name == "strokes" else "atlas"

    def shape(role: str, seed: int, clip: bool = False, t_: int | None = None) -> str:
        if kind == "frame":
            return f' data-shape="frame" data-group="frame" data-seed="{seed}"' + (f' data-t="{t_}"' if t_ else "")
        return f' data-shape="atlas" data-group="{role}" data-seed="{seed}"' + (' data-clip=""' if clip else "")

    v = {"head": t["head"], "head-weight": t["head_weight"], "body": t["body"], "ui": t["ui"], "line": t["line"], "hair": t["hair"],
         "check-radius": "0", "check-bg": "transparent", "track-radius": "999px", "hero-art": ex["hero"][0], "fringe": "none", "fringe-h": "0px",
         "fill-img": f"url({ex['fill'][0]})", "edge-h": {"bands": "30px", "marks": "26px", "strokes": "20px"}[name]}
    v.update(root_vars(name, t))
    root = ":root{" + ";".join(f"--{k}:{val}" for k, val in v.items()) + "}"
    nav = "".join(f'<a href="{k}.html" class="{"on" if k == name else ""}">{k.title()}</a>' for k in PAGES) + '<a href="/designlib/saved/">Saved</a>'

    def btn(cls, label, seed, extra="", act='data-action="save-self"'):
        return f'<button class="btn {cls}"{shape("button", seed)} {act} {extra}>{label}</button>'

    cards = "".join(sp("card", tag,
                       f'<div class="cardwrap"><article class="card"><div class="art"{shape("card", 3 + 7 * i, clip=True, t_=7)} style="background:{bg};background-size:cover;background-position:center"></div>'
                       f'<div class="body"><span class="tag">{html.escape(tag)}</span><h3>{html.escape(title)}</h3><p>{html.escape(text)}</p></div></article></div>')
                    for i, (bg, tag, title, text) in enumerate(ex["cards"]))
    orns = "".join(sp("ornament", f"{var.replace('_', ' ')} · {SPECS[name][var][0] if var in SPECS[name] else 'four strokes'}",
                      f'<div class="orn-glyph" style="width:100%;height:80px;background-image:var(--{var.replace("_", "-")});'
                      f'background-color:{t["ink"] if var in SPECS[name] and SPECS[name][var][2] == "paper" else "transparent"};'
                      f'background-size:{"auto 60%" if var in SPECS[name] and SPECS[name][var][3] == "strip" else "contain"};'
                      f'background-repeat:{"repeat-x" if var in SPECS[name] and SPECS[name][var][3] == "strip" else "no-repeat"}"></div>',
                      vars_=var.replace("_", "-"), svg=var.replace("_", "-")) for var in list(SPECS[name]) + (["box", "box_filled"] if name == "strokes" else []))
    check_rows = "".join(f'<div><b>{label}</b> {r:.1f}:1 (needs {need:g})</div>' for label, r, need in checks)
    tabs_html = f'''<div style="width:100%"><div class="tabs" role="tablist">
    <button class="tab" role="tab" aria-selected="true" aria-controls="p1">Strokes</button>
    <button class="tab" role="tab" aria-selected="false" aria-controls="p2">Compounds</button>
    <button class="tab" role="tab" aria-selected="false" aria-controls="p3">Colorways</button></div>
  <div class="tabpanel" id="p1">Twelve stroke families, from hairline dashes to tapered hooks and closed loops, measured over a hundred features.</div>
  <div class="tabpanel" id="p2" hidden>Combs, feathers, branching lines, hatching and bands: structures of several marks, put back together at their junctions.</div>
  <div class="tabpanel" id="p3" hidden>Opponent colour pairs, yellow against violet most of all, read from each feature's dominant chroma axis.</div></div>'''
    hero_shape = shape("card", 2, clip=True, t_=8) if name != "bands" else ""
    view = {"page": name, "fill": ex["fill"][0], "options": [{"name": o["name"], "label": f"{o['name_a']} / {o['name_b']} · {short(o['name'])}",
                                                               "fill": f"assets/{o['name']}.png", "a": o["a"], "b": o["b"]} for o in colorway_options()]}
    cw_opts = "".join(f'<option value="{o["name"]}">{html.escape(o["label"])}</option>' for o in view["options"])
    bw_opts = "".join(f'<option value="{u}">{html.escape(lab)}</option>' for u, lab in ex["bw"])
    grid_html = ""
    if ex.get("grid"):
        cells = "".join(sp("feature-card", lab, f'<div class="fcard"><div class="art"{shape("card", 41 + 5 * i, clip=True)} style="background:{bg};background-size:cover;background-position:center"></div>'
                           f'<div class="fbody"><b>{html.escape(lab.split(" · ")[1] if " · " in lab else lab)}</b><small>{html.escape(folder)}</small></div></div>')
                        for i, (bg, folder, lab) in enumerate(ex["grid"]))
        grid_html = f'<section><h2>Feature cards</h2><div class="fgrid">{cells}</div></section>'

    box_vars = "box check"

    return bands_of(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{t['title']} · style tile</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family={t['fonts']}&display=swap" rel="stylesheet">
<style>{root}{ST.CSS}{CSS}{PAGE_CSS[name]}</style></head><body data-theme="{name}">
<nav class="switch">{nav}</nav>
<div class="view"><b>View</b>
  <label>Colorway <select name="colorway"><option value="default">the page's own</option>{cw_opts}</select><i class="swatch"></i></label>
  <label>Background on <select name="where"><option value="none">nothing</option><option value="page">the whole page</option><option value="hero">the hero band</option><option value="cards">the cards band</option><option value="alt">alternate bands</option></select></label>
  <label>Picture <select name="pic"><option value="feature">the feature, its own colours</option><option value="duotone">the feature, in the page's colours</option><option value="drawing">a drawing, in the page's colours</option></select></label>
  <label class="drawsrc" hidden>Drawing <select name="drawing">{bw_opts}</select></label>
</div>
<p class="offline-note">Opened as a file: shapes and shape variants work, but saving and ornament variants need the local server (http://127.0.0.1:8790/designlib/tiles/{name}.html).</p>
<header class="hero"><div class="panel">
  <span class="eyebrow">Grammar of Ornament · {t['title']} · from {t['source']}</span>
  <h1>Ornament learned from a neural network</h1>
  <p class="lede">{t['blurb']} Hover anything to make a variant or save it.</p>
  <div class="row" style="margin-top:12px">{btn("primary", "Begin", 1, 'data-target="#components"', 'data-action="scroll"')}{btn("secondary", "See the ornaments", 2, 'data-target="#ornaments"', 'data-action="scroll"')}</div>
</div><div class="art" role="img" aria-label="{html.escape(ex['hero'][1])}"{hero_shape}></div></header>
<div class="caption">{html.escape(ex['hero'][1])}</div>
<div class="wrap">

<section id="components"><h2>Buttons</h2><div class="row">
  {sp("button", "primary", btn("primary", "Save pattern", 3))}
  {sp("button", "primary, hover", btn("primary is-hover", "Save pattern", 4))}
  {sp("button", "primary, focus", btn("primary is-focus", "Save pattern", 5))}
  {sp("button", "primary, disabled", btn("primary", "Save pattern", 6, "disabled"))}
  {sp("button", "secondary", btn("secondary", "Export SVG", 7, 'data-var="divider"', 'data-action="export"'))}
  {sp("button", "secondary, hover", btn("secondary is-hover", "Export SVG", 8, 'data-var="divider"', 'data-action="export"'))}
  {sp("button", "quiet", '<button class="btn quiet" data-action="layers">Show layers</button>', "underline underline-focus")}
</div></section>

<section><h2>Fields</h2><div class="row">
  {sp("field", "text field", '<div class="field"><label for="f1">Pattern name</label><input id="f1" placeholder="Feathered comb"><span class="hint">Shown in the library.</span></div>', "underline")}
  {sp("field", "text field, focus", '<div class="field is-focus"><label for="f2">Repeat width</label><input id="f2" value="48 px"><span class="hint">Focused.</span></div>', "underline-focus")}
  {sp("field", "text field, error", '<div class="field is-error"><label for="f3">Stroke weight</label><input id="f3" value="0.1 px"><span class="hint">Too fine to print; use at least 0.5 px.</span></div>', "underline-error")}
  {sp("field", "select", '<div class="field"><label for="f4">Symmetry</label><select id="f4"><option>Glide reflection</option><option>Mirror</option><option>Rotation</option></select><span class="hint">How the tile repeats.</span></div>', "underline")}
</div></section>

<section><h2>Choices</h2><div class="row" style="gap:30px 46px">
  {sp("checkbox", "checkbox", '<div style="display:flex;gap:26px"><label class="choice"><input type="checkbox" checked> Include hatching</label><label class="choice"><input type="checkbox"> Include dots</label></div>', "box box-filled")}
  {sp("radio", "radio", '<div style="display:flex;gap:26px"><label class="choice"><input type="radio" name="r" checked> Frieze</label><label class="choice"><input type="radio" name="r"> Rosette</label></div>', "ring ring-filled")}
  {sp("toggle", "toggle", '<div style="display:flex;gap:26px"><label class="toggle"><input type="checkbox" checked><span class="track"></span> Relief</label><label class="toggle"><input type="checkbox"><span class="track"></span> Animate</label></div>', "knob")}
</div></section>

<section><h2>Tabs</h2>{sp("tabs", "tabs", tabs_html, "tab")}</section>

<section><h2>Divider</h2>{sp("divider", "divider", '<div class="divider" role="separator"></div>', "divider")}</section>

<section><h2>Cards · examples from the project</h2><div class="cards">{cards}</div></section>
{grid_html}

<section><h2>Feedback</h2><div class="row" style="gap:36px 46px;align-items:center">
  {sp("alert", "alert", f'<div class="alert"{shape("alert", 2)} data-contain=""><div><strong>Pattern saved.</strong><p>It is in the library under Combs.</p></div></div>')}
  {sp("alert", "alert, error", f'<div class="alert error"{shape("alert", 9)} data-contain=""><div><strong>Could not export.</strong><p>The tile is wider than the page.</p></div></div>')}
  {sp("progress", "progress", '<div class="progress"><div class="track"><div class="bar"></div></div><small>Tracing strokes · 62%</small></div>', "bar")}
  {sp("spinner", "spinner", '<div class="spinner" role="img" aria-label="Loading"></div>', "spinner")}
  {sp("badge", "badges", f'<div style="display:flex;gap:16px"><span class="badge"{shape("button", 11, t_=3)}>Comb</span><span class="badge"{shape("button", 12, t_=3)}>Feather</span><span class="badge solid"{shape("button", 13, t_=3)}>New</span></div>')}
</div></section>

<section><h2>Text</h2>{sp("prose", "running text", '''<div class="prose">
  <p class="drop">Owen Jones gathered the ornament of every culture he could reach into one grammar. These are the marks a network made for itself: combs that hang from a wavering spine, feathers barbed on both sides, short tapered strokes laid side by side like an engraver's shading.</p>
  <ul><li>Ornament is the shape itself, never an attachment to it.</li><li>Text sits on solid ground at 4.5:1 or better.</li><li>Every line is one piece, repeated.</li></ul></div>''', "bullet bullet2 bullet3")}
</section>

<section id="ornaments"><h2>Ornaments</h2><div class="orns">{orns}</div></section>

<section><h2>Contrast</h2><div class="contrast">{check_rows}</div></section>
</div>
<script>window.GOAtlas={json.dumps(shp, separators=(",", ":"))};window.GOView={json.dumps(view, separators=(",", ":"))};</script>
<script src="shapes.js"></script><script src="app.js"></script><script src="view.js"></script></body></html>""")


def bands_of(doc: str) -> str:
    """Turn the sections into stacked full-width bands: each gets a paint layer
    whose top is the page's edge contour, alternate bands (starting with the
    first, so the hero's lower edge shows) on the tinted ground."""
    counter = iter(range(1000))
    rng = np.random.default_rng(7)

    def one(m):
        i = next(counter)
        cls = ' class="alt"' if i % 2 == 0 else ""
        tag = m.group(1)
        if "class=" in tag:
            tag = tag.replace('class="', 'class="alt ' if i % 2 == 0 else 'class="', 1)
            cls = ""
        return f'{tag[:-1]}{cls} style="--edge-x:{int(rng.integers(0, 400))}px">' + '<i class="seg-paint" aria-hidden="true"></i>'
    head, sep, rest = doc.partition('<div class="wrap">')
    rest = re.sub(r'(<section[^>]*>)', one, rest)
    return head + sep + rest


def build(screenshot: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assets = OUT / "assets"
    assets.mkdir(exist_ok=True)
    for o in colorway_options():
        up = ST.upscaled_feature(o["name"], HERE / "convnext_basefeatures")
        shutil.copyfile(up, assets / f"{o['name']}.png")
    for name, t in themes().items():
        ex = examples(name, t, assets)
        dark = t["dark"]
        btn_fg, btn_bg = (t["on_accent"], t["accent"]) if dark else (t["paper"], t["ink"])
        hov_fg, hov_bg = (t["paper"], t["ink"]) if dark else (t["on_accent"], t["accent"])
        checks = [(lab, contrast(fg, bg), need) for lab, fg, bg, need in (
            ("body text on ground", t["ink"], t["paper"], 4.5), ("text on cards and fields", t["ink"], t["card_bg"], 4.5),
            ("primary button label", btn_fg, btn_bg, 4.5), ("primary button label, hover", hov_fg, hov_bg, 4.5),
            ("secondary button label", t["accent"] if dark else t["ink"], t["paper"], 4.5),
            ("muted text on ground", t["muted"], t["paper"], 4.5), ("muted text on cards", t["muted"], t["card_bg"], 4.5),
            ("text on tinted bands", t["ink"], t["seg"], 4.5), ("muted text on tinted bands", t["muted"], t["seg"], 4.5),
            ("error text on tinted bands", t["danger"], t["seg"], 4.5),
            ("error text on ground", t["danger"], t["paper"], 4.5), ("focus ring on ground", t["focus"], t["paper"], 3.0),
            ("drop cap on ground (large)", t["focus"], t["paper"], 3.0))]
        bad = [c for c in checks if c[1] < c[2]]
        print(f"  {name:8s} " + "  ".join(f"{r:.1f}" for _, r, _ in checks) + ("   FAIL: " + ", ".join(c[0] for c in bad) if bad else "   all pass"), flush=True)
        (OUT / f"{name}.html").write_text(page(name, t, ex, checks))
        if screenshot:
            out = OUT / f"{name}.png"
            with tempfile.TemporaryDirectory() as prof:
                subprocess.run([ST.CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", f"--user-data-dir={prof}",
                                "--virtual-time-budget=6000", "--window-size=1280,4800", f"--screenshot={out}", (OUT / f"{name}.html").as_uri()],
                               capture_output=True, timeout=120)
            print(f"  screenshot {out.name}", flush=True)
    (OUT / "index.html").write_text("""<!doctype html><meta charset="utf-8"><title>Style tiles</title>
<style>body{font:16px/1.5 system-ui;margin:40px;max-width:680px}a.big{display:block;font-size:22px;margin:10px 0}small a{margin-right:12px}</style>
<h1>Grammar of Ornament: style tiles</h1><p>One page per source of shapes, the same components on each. Hover anything to make a variant or save it.</p>
<a class="big" href="bands.html">Bands</a><a class="big" href="marks.html">Marks</a><a class="big" href="strokes.html">Strokes</a><a class="big" href="/designlib/saved/">Saved specimens</a>
<p><small>Earlier experiments: <a href="woodcut.html">woodcut</a><a href="engraving.html">engraving</a><a href="herbarium.html">herbarium</a><a href="bloom.html">bloom</a></small></p>""")
    print(f"  → {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screenshot", action="store_true")
    build(ap.parse_args().screenshot)
