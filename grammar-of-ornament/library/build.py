#!/usr/bin/env python3
"""
Grammar of Ornament — the web library and its site.

  .venv/bin/python library/build.py            → library/site/  (open with: python3 -m http.server -d library/site)
  .venv/bin/python library/build.py --shots    also screenshots the example pages (headless Chrome)

library/site/
  index.html, site.css, site.js      the browsing site (tabs: font, features, strokes, motifs, colorways,
                                     elements, examples, readme)
  catalog.js                         everything the site lists
  assets/                            features (PNGs), colorway cards, motif PNGs: display only
  examples/                          example web pages built from the library
  neuron-library/                    the library itself (also zipped as neuron-library.zip):
    fonts/  neuron.css  colorways.css  neuron-elements.js  elements/…  elements/sprite.svg  README.md

Elements are drawn by elements.py from the real shapes in compounds/, strokelib/ and vgg16motifs/;
every SVG is one even-odd path in currentColor.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from skimage import color as skc

import edges as EG
import elements as E
import palette as PAL

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SITE = HERE / "site"
LIB = SITE / "neuron-library"
SRC = HERE / "src"
CACHE = HERE / "cache"
FEATURES = {"resnet50": ROOT / "resnet50features", "vgg16": ROOT / "vgg16features",
            "convnext_base": ROOT / "convnext_basefeatures"}
COLORWAYS = ROOT / "colorways" / "convnext_basefeatures"
FONTS = ROOT / "typeface" / "fonts"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

N_RULE, N_MIRROR, N_REPEAT, N_FRIEZE, N_CORNER, N_MARK, N_FLEURON, N_ICONSETS = 10, 8, 8, 8, 8, 12, 12, 2
N_SHAPES, N_MOTIF_SHAPES, N_BOXES, N_LINES, N_MEDALLION = 10, 8, 8, 8, 12
EMBLEM_PX = 1024             # the long side of a filled emblem (a second 4× upscale of the motif's own crop)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- colour

def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255


def rgb_hex(c):
    c = np.clip(np.round(np.asarray(c) * 255), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c)


def lab(h):
    return skc.rgb2lab(hex_rgb(h)[None, None])[0, 0]


def from_lab(L):
    return rgb_hex(skc.lab2rgb(np.asarray(L, float)[None, None])[0, 0])


def luminance(h):
    c = hex_rgb(h)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast(a, b):
    la, lb = sorted([luminance(a), luminance(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def toward(h, ground, ratio):
    """h darkened (or lightened) in lightness only until it reaches `ratio` against ground."""
    L = lab(h)
    step = -1.5 if luminance(ground) > 0.3 else 1.5
    while contrast(from_lab(L), ground) < ratio and 2 < L[0] < 98:
        L[0] += step
    return from_lab(L)


def palettes(names):
    """The picture-drawn palette (palette.py) of each colorway feature, cached in cache/palettes.json."""
    f = CACHE / "palettes.json"
    CACHE.mkdir(exist_ok=True)
    done = json.loads(f.read_text()) if f.exists() else {}
    t, n = time.time(), 0
    for name in names:
        if name not in done:
            rgb = np.asarray(Image.open(FEATURES["convnext_base"] / f"{name}.png").convert("RGB")).astype(float) / 255
            done[name] = PAL.palette(rgb)
            n += 1
    if n:
        f.write_text(json.dumps(done))
        log(f"  palettes: {n} measured ({time.time() - t:.0f}s)")
    return done


def hue_dist(h1, h2):
    a, b = lab(h1), lab(h2)
    d = abs(np.degrees(np.arctan2(a[2], a[1])) - np.degrees(np.arctan2(b[2], b[1]))) % 360
    return min(d, 360 - d)


def theme(cw, pal):
    """A colorway from the picture's own colours (palette.py): the base is the biggest patch of colour in the
    feature, the five are five more colours present in it. The colorway's own side is the ground the five were
    chosen against: paper when the ground is light, the dark ground when it is dark; the other side is a tint of the
    base. On its own side the five are untouched pixels of the picture; on the other they are moved in lightness
    until they read (2.5:1). Ink and the dark side's text are the darkest and lightest of the five pushed to 7:1.
    accent and accent2 are the two of the five nearest the feature's opponent pair."""
    base, ground, five = pal["base"], pal["ground"], pal["five"]
    light_side = pal["ground_lab"][0] > 50
    bl = np.array(pal["base_lab"])
    paper = ground if light_side else PAL.lab_hex(PAL.tint(bl, "light"))
    dark = ground if not light_side else PAL.lab_hex(PAL.tint(bl, "dark"))
    L = [lab(h) for h in five]
    lo, hi = L[int(np.argmin([l[0] for l in L]))], L[int(np.argmax([l[0] for l in L]))]
    ink = toward(from_lab([min(lo[0], 22), lo[1] * 0.7, lo[2] * 0.7]), paper, 7.0)
    on_dark = toward(from_lab([max(hi[0], 85), hi[1] * 0.5, hi[2] * 0.5]), dark, 7.0)
    light_five = five if light_side else [toward(h, paper, 2.5) for h in five]
    dark_five = five if not light_side else [toward(h, dark, 2.5) for h in five]
    p = cw["pair"]
    ia = int(np.argmin([hue_dist(h, p["a"]) for h in five]))
    ib = int(np.argmin([hue_dist(h, p["b"]) if i != ia else 1e9 for i, h in enumerate(five)]))
    return {"paper": paper, "ink": ink, "accent": light_five[ia], "accent2": light_five[ib], "dark": dark, "on_dark": on_dark,
            "dark_accent": dark_five[ia], "dark_accent2": dark_five[ib], "base": base, "ground": ground, "tint": pal["tint"],
            "five": light_five, "dark_five": dark_five, "shares": pal["shares"], "note": pal["note"],
            "shadow": cw["duotone"]["shadow"], "highlight": cw["duotone"]["highlight"],
            "ramp": "linear-gradient(90deg," + ",".join(cw["gradient"]["image"]) + ")",
            "mesh": cw["mesh"]["css"], "mesh_base": cw["mesh"]["base"]}


# ---------------------------------------------------------------- assets

def copy_assets(cat):
    A = SITE / "assets"
    for model, src in FEATURES.items():
        dst = A / "features" / model
        dst.mkdir(parents=True, exist_ok=True)
        items = []
        for f in sorted(src.glob("*.json")):
            if f.name == "index.json" or not f.with_suffix(".png").exists():
                continue
            m = json.loads(f.read_text())
            shutil.copy2(f.with_suffix(".png"), dst / f.with_suffix(".png").name)
            rec = {"name": m["name"], "model": model, "layer": m.get("layer", ""),
                   "channel": (m.get("target") or {}).get("channel"), "src": f"assets/features/{model}/{f.stem}.png"}
            # upscales (upscale_all.py): big is the 8× (1792 px) when made, else the 4× (896); huge is the 16× (3584)
            sr, sr8, sr16 = (ROOT / "cache" / d / f"{f.stem}{ext}" for d, ext in (("sr", ".png"), ("sr8", ".jpg"), ("sr16", ".jpg")))
            big = dst / "big" / f"{f.stem}.jpg"
            if sr8.exists():
                (dst / "big").mkdir(exist_ok=True)
                if not big.exists() or big.stat().st_size != sr8.stat().st_size:
                    shutil.copy2(sr8, big)
                rec["big"] = f"assets/features/{model}/big/{f.stem}.jpg"
            elif sr.exists():
                (dst / "big").mkdir(exist_ok=True)
                if not big.exists():
                    Image.open(sr).convert("RGB").save(big, quality=86)
                rec["big"] = f"assets/features/{model}/big/{f.stem}.jpg"
            tile8 = ROOT / "cache" / "sr8_tile" / f"{f.stem}.jpg"             # seamless (tile_all.py), for backgrounds
            if tile8.exists():
                (dst / "tile").mkdir(exist_ok=True)
                tl = dst / "tile" / f"{f.stem}.jpg"
                if not tl.exists() or tl.stat().st_size != tile8.stat().st_size:
                    shutil.copy2(tile8, tl)
                rec["tile"] = f"assets/features/{model}/tile/{f.stem}.jpg"
            if sr16.exists():
                (dst / "huge").mkdir(exist_ok=True)
                huge = dst / "huge" / f"{f.stem}.jpg"
                if not huge.exists() or huge.stat().st_size != sr16.stat().st_size:
                    shutil.copy2(sr16, huge)
                rec["huge"] = f"assets/features/{model}/huge/{f.stem}.jpg"
            items.append(rec)
        items.sort(key=lambda r: (_layer_key(r["layer"]), r["channel"] or 0))
        cat["features"][model] = items
        log(f"  features {model}: {len(items)}")
    dst = A / "colorways"
    dst.mkdir(parents=True, exist_ok=True)
    cws = [json.loads(f.read_text()) for f in sorted(COLORWAYS.glob("*.json"))]
    cws = [cw for cw in cws if isinstance(cw, dict) and "pair" in cw]          # (the folder's index is not one)
    pals = palettes([cw["name"] for cw in cws])
    for cw in cws:
        f = COLORWAYS / f"{cw['name']}.json"
        card = f.with_name(f.stem + ".card.png")
        if card.exists():
            shutil.copy2(card, dst / card.name)
        feat = FEATURES["convnext_base"] / (cw["name"] + ".png")
        cat["colorways"].append({"short": cw["short"], "name": cw["name"], "layer": cw["layer"],
                                 "a": cw["pair"]["a"], "b": cw["pair"]["b"],
                                 "name_a": cw["pair"].get("name_a", ""), "name_b": cw["pair"].get("name_b", ""),
                                 "gradient": cw["gradient"]["image"], "card": f"assets/colorways/{card.name}" if card.exists() else "",
                                 "feature": f"assets/features/convnext_base/{feat.name}" if feat.exists() else "",
                                 "tile": next((r["tile"] for r in cat["features"].get("convnext_base", []) if r["name"] == cw["name"] and "tile" in r), ""),
                                 **theme(cw, pals[cw["name"]])})
    cat["colorways"].sort(key=lambda c: (_layer_key(c["layer"]), c["short"]))
    edge_sequences(cat)
    log(f"  colorways: {len(cat['colorways'])}")
    dst = A / "motifs"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(E.MOTIFS.glob("*/*/motif.png")):
        parts = f.parent.parent.name.split("__")                # vgg16__features.17__c10__… → f17-c10-<save>
        mid = "f" + parts[1].split(".")[-1] + "-" + parts[2] + "-" + f.parent.name[-4:]
        shutil.copy2(f, dst / f"{mid}.png")
        cat["motifs"].append({"id": mid, "feature": f.parent.parent.name, "png": f"assets/motifs/{mid}.png",
                              "svg": f"neuron-library/elements/emblems/{mid}.svg", "path": f.with_suffix(".svg")})
    log(f"  motifs: {len(cat['motifs'])}")
    (LIB / "fonts").mkdir(parents=True, exist_ok=True)
    for f in (SRC / "textfonts").iterdir():
        shutil.copy2(f, LIB / "fonts" / f.name)
    for name in ("Neuron-Leaf",):
        for ext in ("otf", "woff2"):
            shutil.copy2(FONTS / f"{name}.{ext}", LIB / "fonts" / f"{name}.{ext}")


def upscale_again(sq, key):
    """A motif's square of the print upscaled a second time (Real-ESRGAN 4×, as motifs.py did for the whole
    print), kept in library/cache/sr2. The input is capped at 400 px a side, since the result is capped anyway."""
    out = CACHE / "sr2" / f"{key}.png"
    if out.exists():
        return np.asarray(Image.open(out).convert("RGB"))
    import sys
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.nn.functional as F
    import motifs
    im = Image.fromarray(sq)
    if max(im.size) > 400:
        im = im.resize((400, 400), Image.LANCZOS)
    rgb = np.asarray(im).astype(np.float32) / 255
    t = time.time()
    with torch.no_grad():
        x = F.pad(torch.from_numpy(rgb.transpose(2, 0, 1)[None]), (8, 8, 8, 8), mode="reflect")
        y = motifs._net()(x)[0, :, 32:-32, 32:-32].clamp(0, 1).numpy().transpose(1, 2, 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    big = (y * 255).round().astype(np.uint8)
    Image.fromarray(big).save(out)
    log(f"    upscaled {key} {im.size[0]}→{big.shape[0]} px in {time.time() - t:.0f}s")
    return big


def edge_sequences(cat):
    """Each colorway's edge sequence (edges.py), measured on the feature's 4× upscale and cached."""
    f = CACHE / "edges.json"
    CACHE.mkdir(exist_ok=True)
    done = json.loads(f.read_text()) if f.exists() else {}
    t = time.time()
    n = 0
    for c in cat["colorways"]:
        sr = ROOT / "cache" / "sr" / f"{c['name']}.png"
        if c["name"] not in done:
            if not sr.exists():                      # the 4× upscale, made the way motifs.py makes it (cached there)
                import sys
                sys.path.insert(0, str(ROOT))
                import motifs
                log(f"    upscaling {c['short']}")
                motifs.upscaled(FEATURES["convnext_base"], c["name"])
            im = np.asarray(Image.open(sr).convert("RGB")).astype(float) / 255
            prof, seq = EG.edge_sequence(im)
            done[c["name"]] = {"profile": prof, "sequence": seq}
            n += 1
        c["edge_profile"], c["edge_sequence"], c["edge"] = done[c["name"]]["profile"], done[c["name"]]["sequence"], EG.five(done[c["name"]]["sequence"])
    if n:
        f.write_text(json.dumps(done))
        log(f"  edge sequences: {n} measured ({time.time() - t:.0f}s)")


def filled_emblems(cat):
    """Each motif's interior in the colours of its print: the print's 4× upscale cut by the motif's own outline,
    upscaled once more so a hero-sized emblem isn't pixelated. filled: RGBA colour, for the feature-filled
    emblem. lum: the same in grey with alpha, a mask whose luminance paints the highlight ink of a duotone."""
    d = LIB / "elements" / "emblems"
    for m in cat["motifs"]:
        meta = json.loads((m["path"].parent / "motif.json").read_text())
        sr = ROOT / "cache" / "sr" / f"{m['feature']}.png"
        if not sr.exists() or len(meta.get("crop") or []) < 3:
            continue
        rgb = np.asarray(Image.open(sr).convert("RGB"))
        k = rgb.shape[0] / meta["work_px"]
        xs, ys = [x * k for x, _ in meta["crop"]], [y * k for _, y in meta["crop"]]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * 1.12       # the square motifs.trim cuts
        n = int(round(2 * half))
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        sq = np.zeros((n, n, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(rgb.shape[1], x0 + n), min(rgb.shape[0], y0 + n)
        sq[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
        sq = upscale_again(sq, m["id"])
        N = min(sq.shape[0], EMBLEM_PX)
        if N < sq.shape[0]:
            sq = np.asarray(Image.fromarray(sq).resize((N, N), Image.LANCZOS))
        paths, vb = E.read_svg(m["path"])                                  # the cut outline, drawn at this size
        ink, _ = E.raster(paths, vb, N)
        ink = ink[4:4 + N, 4:4 + N]
        if not ink.any():
            continue
        yy, xx = np.nonzero(ink)
        b = (max(0, yy.min() - 2), yy.max() + 3, max(0, xx.min() - 2), xx.max() + 3)
        col, a = sq[b[0]:b[1], b[2]:b[3]], (ink[b[0]:b[1], b[2]:b[3]] * 255).astype(np.uint8)
        a = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.6)))          # a soft edge, not a staircase
        im = Image.fromarray(np.dstack([col, a]), "RGBA")
        im.save(d / f"{m['id']}-filled.webp", quality=80)
        g = im.convert("LA")
        L = np.asarray(g)[..., 0].astype(float)
        lo, hi = np.percentile(L[np.asarray(g)[..., 1] > 0], [2, 98])        # stretched, so the duotone uses its range
        L = (np.clip((L - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(np.dstack([L, L, L, np.asarray(g)[..., 1]]), "RGBA").save(d / f"{m['id']}-lum.webp", quality=80)
        m["filled"] = f"neuron-library/elements/emblems/{m['id']}-filled.webp"
        m["lum"] = f"neuron-library/elements/emblems/{m['id']}-lum.webp"
    log(f"  filled emblems: {sum('filled' in m for m in cat['motifs'])}")


def _layer_key(layer):
    return [int(t) if t.isdigit() else t for t in layer.replace("layer", "layer.").split(".")]


# ---------------------------------------------------------------- elements

def pieces():
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "pieces.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    t = time.time()
    C = E.compounds()
    log(f"  traced {len(C)} compounds ({time.time() - t:.0f}s)")
    S = E.strokes()
    log(f"  traced {len(S)} strokes ({time.time() - t:.0f}s)")
    f.write_bytes(pickle.dumps((C, S)))
    return C, S


def motif_pieces(cat):
    """The motifs traced and measured like compounds (cached)."""
    f = CACHE / "motif-pieces.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    t = time.time()
    M = {}
    for m in cat["motifs"]:
        p = E.Piece(m["id"], m["path"], res=400)
        if p.ok:
            M[m["id"]] = p
    log(f"  traced {len(M)} motifs ({time.time() - t:.0f}s)")
    f.write_bytes(pickle.dumps(M))
    return M


def write(kind, name, rings, w, h, cat, source, **extra):
    d = LIB / "elements" / kind
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.svg").write_text(E.svg(rings, round(w, 1), round(h, 1), title=f"{name} · from {source}"))
    rec = {"id": name, "kind": kind, "file": f"elements/{kind}/{name}.svg", "w": round(w, 1), "h": round(h, 1),
           "source": source, **extra}
    cat["elements"].append(rec)
    return rec


def short(ref):
    fam, n = ref.split("/")
    return fam.replace("family-", "f").replace("-branching-line", "-branch") + "-" + n.replace("stroke-", "")


def build_elements(cat, C, S):
    used = set()
    rules = E.pick(C, E.s_rule, N_RULE)
    for p in rules:
        write("dividers", f"rule-{short(p.ref)}", *E.divider_flat(p), cat, p.ref, style="rule")
    centres = {p.family: p for p in E.pick(C, E.s_compact, 60, per_family=1)}
    compact = E.pick(C, E.s_compact, N_MARK)
    for p in rules[:N_MIRROR]:
        c = centres.get(p.family) or compact[0]
        write("dividers", f"mirror-{short(p.ref)}", *E.divider_mirror(p, c), cat, f"{p.ref} + {c.ref}", style="mirror")
    reps = E.pick(S, lambda p: p.aspect + 10 * p.line if p.aspect > 1.5 and p.parts == 1 else -1e9, N_REPEAT)
    for p in reps:
        rings, w, h = E.divider_repeat(p, n=2)                  # one glide-reflected pair: a seamless tile
        write("tiles", f"dash-{short(p.ref)}", rings, w, h, cat, p.ref, style="repeat")
    for p in E.pick(C, E.s_tile, N_FRIEZE):
        write("tiles", f"strip-{short(p.ref)}", *E.frieze(p), cat, p.ref, style="strip")
    # long single strokes: the edges of frames and drawn boxes, and lines on their own
    lines_ = E.pick(S, lambda p: p.line * 2 + min(p.aspect, 10) / 10 if p.aspect >= 4 and p.parts == 1 else -1e9, 16, per_family=2)
    for k, st in enumerate(lines_[:N_LINES]):
        d = LIB / "elements" / "lines"
        d.mkdir(parents=True, exist_ok=True)
        for o in ("t", "b", "l", "r"):            # the same stroke in four orientations (masks can't be flipped in CSS)
            r, w, h = E.line(st, o)
            (d / f"line-{k + 1}-{o}.svg").write_text(E.svg(r, w, h, title=f"line-{k + 1} {o} · from {st.ref}", attrs='preserveAspectRatio="none"'))
        cat["elements"].append({"id": f"line-{k + 1}", "kind": "lines", "file": f"elements/lines/line-{k + 1}-t.svg",
                                "w": w, "h": h, "source": st.ref, "n": k + 1})
    for i, p in enumerate(E.pick(C, E.s_curved, N_CORNER)):
        rings, s, _ = E.corner(p)
        name = f"corner-{short(p.ref)}"
        for where in ("tl", "tr", "bl", "br"):
            d = LIB / "elements" / "corners"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}-{where}.svg").write_text(E.svg(E.turn_corner(rings, s, where), s, s, title=f"{name} {where}"))
        write("corners", name, rings, s, s, cat, p.ref, turns=["tl", "tr", "bl", "br"])
        edge = lines_[i % N_LINES]
        write("frames", f"frame-{short(p.ref)}", *E.frame(rings, s, edge), cat, f"{p.ref} + {edge.ref}", corner=name,
              line=f"line-{i % N_LINES + 1}", edge=short(edge.ref))
    for p in compact:
        write("marks", f"mark-{short(p.ref)}", *E.mark(p), cat, p.ref)
    folds = [(4, False), (6, False), (8, True), (5, False), (6, True), (8, False)]
    for i, p in enumerate(E.pick(C, E.s_petal, N_FLEURON)):
        n, m = folds[i % len(folds)]
        write("fleurons", f"fleuron-{short(p.ref)}", *E.fleuron(p, n=n, mirror=m), cat, p.ref, fold=n, mirror=m)
    sets = E.pick(S, lambda p: p.line * 3 + min(p.aspect, 6) / 6 if 3 <= p.aspect <= 12 and p.parts == 1 else -1e9,
                  N_ICONSETS, per_family=1)
    for k, st in enumerate(sets):
        for name in E.ICONS:
            write("icons", f"{name}{'' if k == 0 else '-' + str(k + 1)}", *E.icon(name, st), cat, st.ref, set=k + 1, glyph=name)
    log(f"  elements: {len(cat['elements'])}")
    for m in cat["motifs"]:
        rings, w, h = E.emblem(m["path"])
        d = LIB / "elements" / "emblems"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{m['id']}.svg").write_text(E.svg(rings, w, h, title=f"{m['id']} · vgg16 motif"))
    log(f"  emblems: {len(cat['motifs'])}")
    filled_emblems(cat)
    # medallions: a motif turned round a centre, its long axis on the radius and its heavier end inward
    M = motif_pieces(cat)
    for m in cat["motifs"]:
        p = M.get(m["id"])
        if p:
            m["axis"], m["aspect"] = round(E.axis_deg(p), 1), round(p.aspect, 2)
    for i, p in enumerate(E.pick(list(M.values()), E.s_medallion, N_MEDALLION, per_family=99)):
        n = 8 if p.aspect >= 2.6 else 6 if p.aspect >= 1.9 else 4
        write("medallions", f"medallion-{p.ref}", *E.medallion(p, n=n, mirror=(i % 2 == 1)), cat, f"motif {p.ref}",
              fold=n, mirror=(i % 2 == 1), motif=p.ref)
    log(f"  medallions: {sum(e['kind'] == 'medallions' for e in cat['elements'])}")
    for m in cat["motifs"]:
        m.pop("path", None)
    # drawn boxes: near-rectangles whose edges are taken from single strokes
    for k in range(N_BOXES):
        picks = [lines_[(4 * k + j) % len(lines_)] for j in range(4)]
        Q = E.drawn_box(picks, seed=k)
        cat["boxes"].append({"id": f"box-{k + 1}", "clip": E.clip_path(Q), "strokes": [p.ref for p in picks]})
    log(f"  boxes: {len(cat['boxes'])}")
    d = LIB / "elements" / "strokes"
    d.mkdir(parents=True, exist_ok=True)
    for p in C:
        lo, hi = E.bbox(p.rings)
        sz = hi - lo
        k = 200 / max(sz)
        (d / f"{p.ref.replace('/', '-')}.svg").write_text(
            E.svg([(R - lo) * k for R in p.rings], round(sz[0] * k, 1), round(sz[1] * k, 1), title=p.ref))
        cat["strokes"].setdefault(p.family, []).append(p.ref.split("/")[1])
    log(f"  strokes: {len(C)} compounds in {len(cat['strokes'])} families")
    return {"rules": rules, "compact": compact, "curved": E.pick(C, E.s_curved, N_CORNER),
            "petals": E.pick(C, E.s_petal, N_FLEURON), "tiles": E.pick(C, E.s_tile, N_FRIEZE), "lines": lines_[:8]}


def sprite(cat):
    """Icons, marks and fleurons as <symbol>s: <svg><use href="sprite.svg#n-icon-menu"/></svg>."""
    out = ['<svg xmlns="http://www.w3.org/2000/svg" style="display:none">']
    for e in cat["elements"]:
        if e["kind"] in ("icons", "marks", "fleurons"):
            src = (LIB / e["file"]).read_text()
            d = src.split(' d="', 1)[1].split('"', 1)[0]
            out.append(f'<symbol id="n-{e["kind"][:-1]}-{e["id"].split("-", 1)[1] if e["kind"] != "icons" else e["id"]}" '
                       f'viewBox="0 0 {e["w"]:g} {e["h"]:g}"><path fill="currentColor" fill-rule="evenodd" d="{d}"/></symbol>')
            e["symbol"] = out[-1].split('id="', 1)[1].split('"', 1)[0]
    out.append("</svg>\n")
    (LIB / "elements" / "sprite.svg").write_text("\n".join(out))


# ---------------------------------------------------------------- css

FF = "@font-face{{font-family:'{f}';src:url(fonts/{file}) format('woff2');font-weight:{w};font-style:{st};font-display:swap}}"
TEXT_FACES = [FF.format(f="Alegreya Sans", file=f"AlegreyaSans-{w}{i}.woff2", w=w, st="italic" if i else "normal")
              for w, i in (("400", ""), ("400", "i"), ("500", ""), ("700", ""))] + [
    FF.format(f="Alegreya Sans SC", file=f"AlegreyaSansSC-{w}.woff2", w=w, st="normal") for w in ("400", "500", "700")] + [
    FF.format(f="DM Mono", file=f"DMMono-{w}{i}.woff2", w=w, st="italic" if i else "normal")
    for w, i in (("400", ""), ("400", "i"), ("500", ""))]


def css(cat):
    cw = cat["colorways"]
    m = lambda u: f"-webkit-mask-image:{u};mask-image:{u}"
    lines = ["/* Neuron library — fonts, colorways and elements from Grammar of Ornament.",
             "   Every element is an SVG mask filled with currentColor: set `color` to recolour it.",
             "   Serve the folder over http(s); browsers don't load masks from file:// pages. */", "",
             "@font-face{font-family:'Neuron Leaf';src:url(fonts/Neuron-Leaf.woff2) format('woff2'),url(fonts/Neuron-Leaf.otf) format('opentype');font-display:swap}",
             "", "/* text faces (SIL Open Font License, see fonts/OFL-*.txt): Alegreya Sans for text, its small caps for labels, DM Mono for code */"] + TEXT_FACES + [
             "", ":root{--n-paper:#f3f1ec;--n-ink:#171614;--n-accent:#8a3a1c;--n-accent2:#3d5a6c;--n-dark:#151413;--n-on-dark:#efe9dc;",
             "  --n-base:#f3f1ec;--n-c1:#8a3a1c;--n-c2:#8a6a1c;--n-c3:#4e6b3a;--n-c4:#3d5a6c;--n-c5:#6b4a7c;",
             "  --n-dark-accent:#d9825a;--n-dark-accent2:#8fb3c8;--n-dark-c1:#d9825a;--n-dark-c2:#d0ac55;--n-dark-c3:#8fb37a;--n-dark-c4:#8fb3c8;--n-dark-c5:#b596c8;",
             "  --n-mesh:none;--n-display:'Neuron Leaf',Georgia,serif;--n-text:'Alegreya Sans','Helvetica Neue',Helvetica,Arial,sans-serif;",
             "  --n-label:'Alegreya Sans SC','Alegreya Sans',sans-serif;--n-mono:'DM Mono',Menlo,Consolas,monospace}", "",
             ".n-text{font-family:var(--n-text)}",
             ".n-label{font-family:var(--n-label);font-weight:500;text-transform:lowercase;letter-spacing:.09em}",
             ".n-display{font-family:var(--n-display);font-weight:normal;letter-spacing:.01em;line-height:1.05}",
             ".n-page{background:var(--n-paper);color:var(--n-ink)}", ".n-dark{background:var(--n-dark);color:var(--n-on-dark)}",
             ".n-accent{color:var(--n-accent)}", ".n-accent2{color:var(--n-accent2)}",
             ".n-mesh{background:var(--n-mesh-base,#888);background-image:var(--n-mesh)}", "",
             "/* colorways: class n-cw-<short> on any element sets the palette for everything inside. The colours are the",
             "   feature's own: --n-base is the biggest patch of colour in it, --n-c1 … --n-c5 five more colours present in it",
             "   (ordered by hue), accent and accent2 the two of those nearest its opponent pair. Paper or the dark ground is the",
             "   base itself (or a tint of it when the base can't carry five of the picture's colours) */"]
    for c in cw:
        lines.append(f".n-cw-{c['short']}{{--n-paper:{c['paper']};--n-ink:{c['ink']};--n-accent:{c['accent']};"
                     f"--n-accent2:{c['accent2']};--n-dark:{c['dark']};--n-on-dark:{c['on_dark']};"
                     f"--n-dark-accent:{c['dark_accent']};--n-dark-accent2:{c['dark_accent2']};--n-base:{c['base']};"
                     + "".join(f"--n-c{i + 1}:{h};" for i, h in enumerate(c["five"]))
                     + "".join(f"--n-dark-c{i + 1}:{h};" for i, h in enumerate(c["dark_five"]))
                     + f"--n-shadow:{c['shadow']};--n-highlight:{c['highlight']};--n-ramp:{c['ramp']};"
                     f"--n-mesh-base:{c['mesh_base']};--n-mesh:{c['mesh']}"
                     + ("".join(f";--n-edge-{i + 1}:{h}" for i, h in enumerate(c["edge"]))
                        + f";--n-edge-profile:linear-gradient(90deg,{','.join(c['edge_profile'])})" if "edge" in c else "") + "}")
    lines += ["", "/* the dark side of a colorway: n-dark swaps the whole palette, so everything inside takes the dark ground,",
              "   its text colour and accents lifted to read on it */",
              ".n-dark{--n-paper:var(--n-dark);--n-ink:var(--n-on-dark);--n-accent:var(--n-dark-accent);--n-accent2:var(--n-dark-accent2);"
              "--n-c1:var(--n-dark-c1);--n-c2:var(--n-dark-c2);--n-c3:var(--n-dark-c3);--n-c4:var(--n-dark-c4);--n-c5:var(--n-dark-c5);"
              "--n-glow:color-mix(in srgb,var(--n-highlight) 55%,transparent);background:var(--n-paper);color:var(--n-ink)}",
              ".n-c1{color:var(--n-c1)}.n-c2{color:var(--n-c2)}.n-c3{color:var(--n-c3)}.n-c4{color:var(--n-c4)}.n-c5{color:var(--n-c5)}.n-base{background:var(--n-base)}",
              "", "/* light: a halo in the colorway's highlight (the features glow); strongest on dark grounds */",
              ":root{--n-glow:color-mix(in srgb,var(--n-highlight,#fff) 35%,transparent)}",
              ".n-glow{text-shadow:0 0 .12em var(--n-glow),0 0 .5em var(--n-glow),0 0 1.4em var(--n-glow)}",
              ".n-glow-box{filter:drop-shadow(0 0 .35em var(--n-glow)) drop-shadow(0 0 1.2em var(--n-glow))}",
              "/* fringe: the colour splitting into its opponent pair at an edge, as the features do */",
              ".n-fringe{text-shadow:-.03em 0 var(--n-accent),.03em 0 var(--n-accent2)}",
              ".n-fringe.n-glow{text-shadow:-.03em 0 var(--n-accent),.03em 0 var(--n-accent2),0 0 .5em var(--n-glow),0 0 1.4em var(--n-glow)}",
              ".n-fringe-box{box-shadow:-3px 0 var(--n-accent),3px 0 var(--n-accent2)}",
              "/* edge sequence: the colours met crossing the feature's edges, dark side (--n-edge-1) to light side (--n-edge-5).",
              "   n-outline: type filled with the light end and outlined outward through the sequence (for dark grounds);",
              "   --n-outline scales the rings' width (.5 by default, about a twentieth of the type size; 1 is twice that);",
              "   n-outline-in: filled with the dark end, outlined outward to the light end (for paper); n-fringe-edge: a fringe",
              "   in the sequence's colours; n-edged: a box with the sequence as inset bands; n-edge-line: the profile as a rule */",
              ":root{--n-edge-1:var(--n-shadow,#222);--n-edge-2:var(--n-accent2);--n-edge-3:var(--n-accent);--n-edge-4:var(--n-highlight,#fff);--n-edge-5:var(--n-highlight,#fff);--n-edge-profile:var(--n-ramp)}",
              ".n-outline{color:var(--n-edge-5);text-shadow:calc(0.018em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.016em * var(--n-outline,.5)) calc(0.009em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.009em * var(--n-outline,.5)) calc(0.016em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.000em * var(--n-outline,.5)) calc(0.018em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.009em * var(--n-outline,.5)) calc(0.016em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.016em * var(--n-outline,.5)) calc(0.009em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.018em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.016em * var(--n-outline,.5)) calc(-0.009em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.009em * var(--n-outline,.5)) calc(-0.016em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.000em * var(--n-outline,.5)) calc(-0.018em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.009em * var(--n-outline,.5)) calc(-0.016em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.016em * var(--n-outline,.5)) calc(-0.009em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.040em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.035em * var(--n-outline,.5)) calc(0.020em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.020em * var(--n-outline,.5)) calc(0.035em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.000em * var(--n-outline,.5)) calc(0.040em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.020em * var(--n-outline,.5)) calc(0.035em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.035em * var(--n-outline,.5)) calc(0.020em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.040em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.035em * var(--n-outline,.5)) calc(-0.020em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.020em * var(--n-outline,.5)) calc(-0.035em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.000em * var(--n-outline,.5)) calc(-0.040em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.020em * var(--n-outline,.5)) calc(-0.035em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.035em * var(--n-outline,.5)) calc(-0.020em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.066em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(0.057em * var(--n-outline,.5)) calc(0.033em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(0.033em * var(--n-outline,.5)) calc(0.057em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(0.000em * var(--n-outline,.5)) calc(0.066em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(-0.033em * var(--n-outline,.5)) calc(0.057em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(-0.057em * var(--n-outline,.5)) calc(0.033em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(-0.066em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(-0.057em * var(--n-outline,.5)) calc(-0.033em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(-0.033em * var(--n-outline,.5)) calc(-0.057em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(-0.000em * var(--n-outline,.5)) calc(-0.066em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(0.033em * var(--n-outline,.5)) calc(-0.057em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(0.057em * var(--n-outline,.5)) calc(-0.033em * var(--n-outline,.5)) 0 var(--n-edge-2),calc(0.096em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(0.083em * var(--n-outline,.5)) calc(0.048em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(0.048em * var(--n-outline,.5)) calc(0.083em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(0.000em * var(--n-outline,.5)) calc(0.096em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(-0.048em * var(--n-outline,.5)) calc(0.083em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(-0.083em * var(--n-outline,.5)) calc(0.048em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(-0.096em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(-0.083em * var(--n-outline,.5)) calc(-0.048em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(-0.048em * var(--n-outline,.5)) calc(-0.083em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(-0.000em * var(--n-outline,.5)) calc(-0.096em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(0.048em * var(--n-outline,.5)) calc(-0.083em * var(--n-outline,.5)) 0 var(--n-edge-1),calc(0.083em * var(--n-outline,.5)) calc(-0.048em * var(--n-outline,.5)) 0 var(--n-edge-1)}",
              ".n-outline-in{color:var(--n-edge-1);text-shadow:calc(0.018em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.016em * var(--n-outline,.5)) calc(0.009em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.009em * var(--n-outline,.5)) calc(0.016em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.000em * var(--n-outline,.5)) calc(0.018em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.009em * var(--n-outline,.5)) calc(0.016em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.016em * var(--n-outline,.5)) calc(0.009em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.018em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.016em * var(--n-outline,.5)) calc(-0.009em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.009em * var(--n-outline,.5)) calc(-0.016em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.000em * var(--n-outline,.5)) calc(-0.018em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.009em * var(--n-outline,.5)) calc(-0.016em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.016em * var(--n-outline,.5)) calc(-0.009em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.040em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.035em * var(--n-outline,.5)) calc(0.020em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.020em * var(--n-outline,.5)) calc(0.035em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.000em * var(--n-outline,.5)) calc(0.040em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.020em * var(--n-outline,.5)) calc(0.035em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.035em * var(--n-outline,.5)) calc(0.020em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.040em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.035em * var(--n-outline,.5)) calc(-0.020em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.020em * var(--n-outline,.5)) calc(-0.035em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(-0.000em * var(--n-outline,.5)) calc(-0.040em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.020em * var(--n-outline,.5)) calc(-0.035em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.035em * var(--n-outline,.5)) calc(-0.020em * var(--n-outline,.5)) 0 var(--n-edge-3),calc(0.066em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.057em * var(--n-outline,.5)) calc(0.033em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.033em * var(--n-outline,.5)) calc(0.057em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.000em * var(--n-outline,.5)) calc(0.066em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.033em * var(--n-outline,.5)) calc(0.057em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.057em * var(--n-outline,.5)) calc(0.033em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.066em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.057em * var(--n-outline,.5)) calc(-0.033em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.033em * var(--n-outline,.5)) calc(-0.057em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(-0.000em * var(--n-outline,.5)) calc(-0.066em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.033em * var(--n-outline,.5)) calc(-0.057em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.057em * var(--n-outline,.5)) calc(-0.033em * var(--n-outline,.5)) 0 var(--n-edge-4),calc(0.096em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(0.083em * var(--n-outline,.5)) calc(0.048em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(0.048em * var(--n-outline,.5)) calc(0.083em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(0.000em * var(--n-outline,.5)) calc(0.096em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(-0.048em * var(--n-outline,.5)) calc(0.083em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(-0.083em * var(--n-outline,.5)) calc(0.048em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(-0.096em * var(--n-outline,.5)) calc(0.000em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(-0.083em * var(--n-outline,.5)) calc(-0.048em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(-0.048em * var(--n-outline,.5)) calc(-0.083em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(-0.000em * var(--n-outline,.5)) calc(-0.096em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(0.048em * var(--n-outline,.5)) calc(-0.083em * var(--n-outline,.5)) 0 var(--n-edge-5),calc(0.083em * var(--n-outline,.5)) calc(-0.048em * var(--n-outline,.5)) 0 var(--n-edge-5)}",
              ".n-fringe-edge{text-shadow:-.03em 0 var(--n-edge-4),.03em 0 var(--n-edge-1),0 0 .4em color-mix(in srgb,var(--n-edge-5) 55%,transparent)}",
              ".n-edged{box-shadow:inset 0 0 0 var(--n-band,4px) var(--n-edge-1),inset 0 0 0 calc(2 * var(--n-band,4px)) var(--n-edge-2),inset 0 0 0 calc(3 * var(--n-band,4px)) var(--n-edge-3),inset 0 0 0 calc(4 * var(--n-band,4px)) var(--n-edge-4),inset 0 0 0 calc(5 * var(--n-band,4px)) var(--n-edge-5);padding:calc(5 * var(--n-band,4px) + 1.2em) calc(5 * var(--n-band,4px) + 1.4em)}",
              ".n-edge-line{display:block;height:var(--n-h,6px);border:0;background:var(--n-edge-profile);margin:var(--n-gap,1.5rem) 0}",
              "/* ramp: the colours of the feature, left to right */",
              ".n-ramp{background:var(--n-ramp)}",
              ".n-ramp-line{display:block;height:var(--n-h,6px);border:0;background:var(--n-ramp);margin:var(--n-gap,1.5rem) 0}",
              ".n-ramp-under{background:var(--n-ramp) left bottom/100% .12em no-repeat;padding-bottom:.06em}",
              ""]
    lines += ["", "/* lines: a single stroke stretched to the rule's length, its taper kept. n-line-k is horizontal (heavy end",
              "   left), n-line-v-k vertical (heavy end at the bottom); --n-h is the weight, --n-len a vertical line's length */",
              "[class*='n-line-']{display:block;border:0;background:currentColor;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;"
              "-webkit-mask-size:100% 100%;mask-size:100% 100%}",
              "[class*='n-line-']:not([class*='n-line-v-']){height:var(--n-h,3px);width:100%;margin:var(--n-gap,1.5rem) auto}",
              "[class*='n-line-v-']{width:var(--n-h,3px);height:var(--n-len,100%)}"]
    for e in cat["elements"]:
        if e["kind"] == "lines":
            lines.append(f".n-line-{e['n']}{{{m(f'url(elements/lines/{e['id']}-t.svg)')}}}")
            lines.append(f".n-line-v-{e['n']}{{{m(f'url(elements/lines/{e['id']}-l.svg)')}}}")
            e["css"] = f"n-line-{e['n']}"
    lines += ["", "/* drawn boxes: a near-rectangle whose edges waver like the single strokes they're taken from.",
              "   n-box-k: a filled panel with that edge (clip-path). n-drawn-k: the same as a drawn border; wrap the content in",
              "   one child, and set --n-line for the line's weight and color for its colour */",
              "[class*='n-box-']{padding:var(--n-pad,1.2em 1.4em)}",
              "[class*='n-drawn-']{display:grid;padding:var(--n-line,3px);background:currentColor}",
              "[class*='n-drawn-']>*{background:var(--n-paper);color:var(--n-ink);padding:var(--n-pad,1.2em 1.4em);box-sizing:border-box}"]
    for bx in cat["boxes"]:
        lines.append(f".n-box-{bx['id'][4:]},.n-drawn-{bx['id'][4:]},.n-drawn-{bx['id'][4:]}>*{{clip-path:{bx['clip']}}}")
    lines += ["", "/* elements: shared */",
              "[class*='n-divider-'],[class*='n-repeat-'],[class*='n-frieze-'],[class*='n-mark-'],[class*='n-icon-'],[class*='n-fleuron-'],[class*='n-emblem-'],[class*='n-medallion-'],[class*='n-corner-']{",
              "  background-color:currentColor;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center;",
              "  -webkit-mask-size:contain;mask-size:contain}",
              "[class*='n-divider-']{display:block;height:var(--n-h,28px);width:100%;border:0;margin:var(--n-gap,1.5rem) auto}",
              "[class*='n-repeat-']{display:block;height:var(--n-h,32px);width:100%;-webkit-mask-repeat:repeat-x;mask-repeat:repeat-x;"
              "-webkit-mask-size:auto 100%;mask-size:auto 100%;-webkit-mask-position:left center;mask-position:left center}",
              "[class*='n-mark-'],[class*='n-icon-'],[class*='n-fleuron-']{display:inline-block;width:var(--n-size,1em);height:var(--n-size,1em);vertical-align:-.15em}",
              "[class*='n-emblem-'],[class*='n-medallion-']{display:block;width:100%;aspect-ratio:1}",
              "[class*='n-corner-']{display:block;width:var(--n-size,96px);height:var(--n-size,96px)}",
              "[class*='n-frame-']{position:relative;padding:var(--n-pad,calc(var(--n-size,96px) * .6))}",
              "[class*='n-frame-']::before{content:'';position:absolute;inset:0;pointer-events:none;background:currentColor;color:inherit}", ""]
    for e in cat["elements"]:
        k, f = e["kind"], e["file"]
        if k == "dividers":
            lines.append(f".n-divider-{e['id']}{{{m(f'url({f})')};aspect-ratio:auto;max-width:calc(var(--n-h,28px) * {e['w'] / e['h']:.2f})}}")
            e["css"] = f"n-divider-{e['id']}"
        elif k == "tiles":
            lines.append(f".n-repeat-{e['id']}{{{m(f'url({f})')}}}")
            e["css"] = f"n-repeat-{e['id']}"
        elif k in ("marks", "fleurons", "icons", "medallions"):
            cls = f"n-{k[:-1]}-{e['id'].split('-', 1)[1] if k != 'icons' else e['id']}"
            lines.append(f".{cls}{{{m(f'url({f})')}}}")
            e["css"] = cls
        elif k == "corners":
            for t in e["turns"]:
                lines.append(f".n-corner-{e['id'].split('-', 1)[1]}-{t}{{{m(f'url(elements/corners/{e['id']}-{t}.svg)')}}}")
            e["css"] = f"n-corner-{e['id'].split('-', 1)[1]}-tl"
        elif k == "lines":
            continue
        elif k == "frames":
            c, ln = e["corner"], e["line"]
            layers = ", ".join(f"url(elements/corners/{c}-{t}.svg) {pos} / var(--n-size,96px) var(--n-size,96px) no-repeat"
                               for t, pos in (("tl", "left top"), ("tr", "right top"), ("bl", "left bottom"), ("br", "right bottom")))
            # the edges: one stroke per side, stretched between the corners, centred on the inset line
            at = f"calc(var(--n-size,96px) * {E.EDGE_INSET - E.EDGE_THICK / 2:.3f})"
            t, L = f"calc(var(--n-size,96px) * {E.EDGE_THICK})", "calc(100% - 2 * var(--n-size,96px))"
            edges = ", ".join([
                f"url(elements/lines/{ln}-t.svg) center {at} / {L} {t} no-repeat",
                f"url(elements/lines/{ln}-b.svg) center calc(100% - {at}) / {L} {t} no-repeat",
                f"url(elements/lines/{ln}-l.svg) {at} center / {t} {L} no-repeat",
                f"url(elements/lines/{ln}-r.svg) calc(100% - {at}) center / {t} {L} no-repeat"])
            lines.append(f".n-frame-{e['id'].split('-', 1)[1]}::before{{-webkit-mask:{layers}, {edges};mask:{layers}, {edges}}}")
            e["css"] = f"n-frame-{e['id'].split('-', 1)[1]}"
    lines += ["", "/* friezes: whole bands (1600 × 140), scaled to the width they're given; a two-ink frieze's second ink",
              "   is var(--n-frieze2) when the SVG is inlined, currentColor otherwise */",
              "[class*='n-frieze-']{display:block;width:100%;aspect-ratio:1600/140;-webkit-mask-size:100% 100%;mask-size:100% 100%}"]
    for fz in cat["friezes"]:
        lines.append(f".n-frieze-{fz['id']}{{{m('url(' + fz['file'] + ')')}}}")
    lines += ["", "/* emblems: silhouette (currentColor); -filled, the motif's own colours from its print;",
              "   -duo, a duotone in --n-shadow and --n-highlight (or any two colours set as --n-duo-dark, --n-duo-light) */",
              "[class*='n-emblem-'][class*='-filled']{background-color:transparent;-webkit-mask:none;mask:none;background-size:contain;background-repeat:no-repeat;background-position:center}",
              "[class*='n-emblem-'][class*='-duo']{position:relative;background-color:var(--n-duo-dark,var(--n-shadow));-webkit-mask-size:contain;mask-size:contain}",
              "[class*='n-emblem-'][class*='-duo']::after{content:'';position:absolute;inset:0;background:var(--n-duo-light,var(--n-highlight));"
              "-webkit-mask-size:contain;mask-size:contain;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center;mask-mode:luminance}"]
    for mo in cat["motifs"]:
        lines.append(f".n-emblem-{mo['id']}{{{m('url(elements/emblems/' + mo['id'] + '.svg)')}}}")
        if "filled" in mo:
            lines.append(f".n-emblem-{mo['id']}-filled{{background-image:url(elements/emblems/{mo['id']}-filled.webp)}}")
            lines.append(f".n-emblem-{mo['id']}-duo{{{m('url(elements/emblems/' + mo['id'] + '-lum.webp)')}}}"
                         f".n-emblem-{mo['id']}-duo::after{{{m('url(elements/emblems/' + mo['id'] + '-lum.webp)')}}}")
    (LIB / "neuron.css").write_text("\n".join(lines) + "\n")
    shutil.copy2(COLORWAYS / "colorways.css", LIB / "colorways.css")


# ---------------------------------------------------------------- js data

def js_pieces(sel):
    """The pieces the web components draw from: flat (long axis along x), heavy end left, 100 units tall."""
    out = {}
    def put(p, role):
        rings = p.flat(smooth=(role == "lines"))             # box edges keep the strokes' own curves
        lo, hi = E.bbox(rings)
        k = 100 / (hi[1] - lo[1])
        rings = [(R - lo) * k for R in rings]
        rings = [E.simplify(R, 0.25 if role == "lines" else 0.6) for R in rings]
        out.setdefault(short(p.ref), {"ref": p.ref, "w": round((hi[0] - lo[0]) * k, 1), "h": 100, "roles": [],
                                      "d": E.path_d(rings, 1)})["roles"].append(role)
    for role, ps in sel.items():
        for p in ps:
            put(p, role)
    return out


# ---------------------------------------------------------------- site

def site(cat):
    for f in SRC.rglob("*"):
        if f.is_file() and "textfonts" not in f.parts:
            dst = SITE / f.relative_to(SRC)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
    shutil.move(str(SITE / "neuron-elements.js"), LIB / "neuron-elements.js")
    shutil.copy2(SRC / "README.md", LIB / "README.md")
    (SITE / "README.md").unlink(missing_ok=True)
    pieces = (CACHE / "js-pieces.json").read_text()
    motifs = {m["id"]: {"axis": m["axis"], "aspect": m["aspect"], "filled": "filled" in m} for m in cat["motifs"] if "axis" in m}
    js = (LIB / "neuron-elements.js").read_text().replace("/*PIECES*/{}", pieces).replace("/*MOTIFS*/{}", json.dumps(motifs, separators=(",", ":")))
    (LIB / "neuron-elements.js").write_text(js)
    v = time.strftime("%Y%m%d%H%M%S")              # a fresh query string per build, so browsers don't keep old copies
    html = (SITE / "index.html").read_text()
    for f in ("neuron-library/neuron.css", "site.css", "catalog.js", "neuron-library/neuron-elements.js", "site.js"):
        html = html.replace(f'"{f}"', f'"{f}?v={v}"')
    (SITE / "index.html").write_text(html)
    for ex in (SITE / "examples").glob("*.html"):
        t = ex.read_text()
        for f in ("../neuron-library/neuron.css", "../neuron-library/neuron-elements.js"):
            t = t.replace(f'"{f}"', f'"{f}?v={v}"')
        ex.write_text(t)
    cat["readme"] = (SRC / "README.md").read_text()
    cat["pieces"] = {k: {"ref": v["ref"], "roles": v["roles"]} for k, v in json.loads(pieces).items()}
    (SITE / "catalog.js").write_text("window.NEURON = " + json.dumps(cat, separators=(",", ":")) + ";\n")


def friezes(cat):
    """The frieze sheet (friezes.py) made recolourable: the ground dropped, the main ink currentColor, a second ink
    var(--n-frieze2) (it falls back to currentColor wherever variables don't reach, as in masks and <img>).
    The originals, in the colours they were made in, are kept for the site."""
    src = ROOT / "friezes"
    d = LIB / "elements" / "friezes"
    d.mkdir(parents=True, exist_ok=True)
    orig = SITE / "assets" / "friezes"
    orig.mkdir(parents=True, exist_ok=True)
    for r in json.loads((src / "friezes.json").read_text()):
        f = src / r["file"]
        svg = f.read_text()
        ground = re.search(r'<rect width="100%" height="100%" fill="(#[0-9a-fA-F]+)"/>', svg)
        svg = svg.replace(ground.group(0), "") if ground else svg
        fills = Counter(re.findall(r'fill="(#[0-9a-fA-F]+)"', svg))
        inks = [c for c, _ in fills.most_common()]
        for i, c in enumerate(inks):
            svg = svg.replace(f'fill="{c}"', 'fill="currentColor"' if i == 0 else 'style="fill:var(--n-frieze2,currentColor)"')
        svg = svg.replace("<svg ", '<svg fill="currentColor" ', 1)
        fid = f.stem
        (d / f.name).write_text(svg)
        shutil.copy2(f, orig / f.name)
        cat["friezes"].append({"id": fid, "file": f"elements/friezes/{f.name}", "orig": f"assets/friezes/{f.name}",
                               "unit": r["unit"], "kind": r["kind"], "arrangement": r["arrangement"], "rails": r["rails"],
                               "colour": r["colour"], "reverse": r["colour"].startswith("reverse"),
                               "two": len(inks) > 1, "ground": ground.group(1) if ground else None})
    log(f"  friezes: {len(cat['friezes'])}")


EXAMPLE_IMAGES8 = {"f50-c451.jpg": "convnext_base__features.5.0__c451__224px__s0__9141f8"}   # 1792 px upscales (cache/sr8)
EXAMPLE_IMAGES = {"f21-c70.jpg": "vgg16__features.21__c70__224px__s0__edc94c",
                  "f24-c119.jpg": "vgg16__features.24__c119__224px__s0__468233",
                  "f24-c27.jpg": "vgg16__features.24__c27__224px__s0__0bcebd"}
TILE_IMAGES = {"cnx-f31-c228-tile.jpg": "convnext_base__features.3.1__c228__224px__s0__0f6e4d"}   # seamless upscales


def example_images():
    d = SITE / "examples" / "img"
    d.mkdir(parents=True, exist_ok=True)
    for out, name in EXAMPLE_IMAGES.items():
        Image.open(ROOT / "cache" / "sr" / f"{name}.png").convert("RGB").save(d / out, quality=88)
    for out, name in EXAMPLE_IMAGES8.items():
        shutil.copy2(ROOT / "cache" / "sr8" / f"{name}.jpg", d / out)
    for out, name in TILE_IMAGES.items():
        t8 = ROOT / "cache" / "sr8_tile" / f"{name}.jpg"                     # the seamless 1792 px tile when made
        if t8.exists():
            shutil.copy2(t8, d / out)
        else:
            Image.open(ROOT / "cache" / "sr_tile" / f"{name}.png").convert("RGB").save(d / out, quality=88)
    shutil.copy2(SITE / "assets" / "motifs" / "f24-c119-8852.png", d / "f24-c119-motif.png")


def zip_library():
    z = SITE / "neuron-library.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in sorted(LIB.rglob("*")):
            if p.is_file():
                f.write(p, p.relative_to(SITE))
    log(f"  zip: {z.stat().st_size / 1e6:.1f} MB")


def shots():
    """Screenshots of the example pages for the Examples tab (needs masks over http, so a local server)."""
    port = 8811
    srv = subprocess.Popen(["python3", "-m", "http.server", str(port), "-d", str(SITE)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    try:
        for f in sorted((SITE / "examples").glob("*.html")):
            out = SITE / "examples" / (f.stem + ".png")
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1440,1800",
                            "--virtual-time-budget=4000", f"--screenshot={out}",
                            f"http://127.0.0.1:{port}/examples/{f.name}"], capture_output=True, timeout=90)
            log(f"  screenshot {out.name}")
    finally:
        srv.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", action="store_true")
    a = ap.parse_args()
    t = time.time()
    if LIB.exists():
        shutil.rmtree(LIB)                       # only the generated library folder, rebuilt below
    LIB.mkdir(parents=True)
    cat = {"features": {}, "colorways": [], "motifs": [], "strokes": {}, "elements": [], "friezes": [], "boxes": [], "built": time.strftime("%Y-%m-%d")}
    log("assets"); copy_assets(cat)
    log("elements"); C, S = pieces(); sel = build_elements(cat, C, S)
    friezes(cat)
    sprite(cat)
    CACHE.mkdir(exist_ok=True)
    (CACHE / "js-pieces.json").write_text(json.dumps(js_pieces(sel), separators=(",", ":")))
    log("css"); css(cat)
    log("site"); site(cat); example_images()
    zip_library()
    if a.shots:
        shots()
    log(f"built {SITE} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
