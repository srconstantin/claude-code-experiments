#!/usr/bin/env python3
"""
Grammar of Ornament — atlas: real compound pieces, cleaned up for design use.

Unlike ornament.py, nothing here is generated: every piece is an actual
compound found by compounds.py in a vgg16 feature (a comb, a crosshatch
cluster, a blobby band, a feather...), chosen automatically, turned so that
its long axis is horizontal (combs with their teeth hanging down), lightly
smoothed, and written as vector paths. Pieces keep their asymmetry and their
irregularities; that is the point.

For each type the pieces are ranked by a score that favours clear examples:
  combs        many teeth, aligned, evenly spaced, few cycles
  feathers     many barbs, evenly spaced, few cycles
  crosshatch   many strokes, crossing near 90°
  hatching     many strokes, evenly spaced
  bands        large, with an outline that can serve as a shape
  branching    long spine
and at most two pieces come from any one feature.

Written into designlib/atlas/:
  atlas.json          every piece: id, type, group, feature, size (source px),
                      aspect, and its paths: "ink" (black), "relief" (white
                      marks over masses), "outline" (masses only: the outer
                      contour, usable as a shape for a button or a picture)
  <group>/<id>.svg    each piece as a standalone SVG
  atlas.png           contact sheet, one row per type

Run:  python3 atlas.py                    # linework/vgg16features → designlib/atlas
      python3 atlas.py --per-type 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import compounds as C
from linework import offset_rings

HERE = Path(__file__).resolve().parent
OUT = HERE / "designlib" / "atlas"

# type → group (what a piece is good for)
GROUPS = {
    "fine crosshatching": "crosshatch", "crosshatching": "crosshatch",
    "curved band": "blob", "lobed wavy band": "blob", "hatched blob": "blob",
    "wavy band": "band", "straight band": "band", "hatched wavy band": "band", "hatched straight band": "band",
    "straight comb": "comb", "curved comb": "comb", "wavy comb": "comb",
    "wavy feather": "feather", "curved feather": "feather", "straight feather": "feather",
    "fine hatching": "hatching", "hatching": "hatching", "fine wavy hatching": "hatching", "wavy hatching": "hatching",
    "wavy branching line": "branch", "curved branching line": "branch", "straight branching line": "branch",
    "loop": "loop", "wavy sprig": "sprig", "curved sprig": "sprig", "straight sprig": "sprig",
}


def score(c: dict) -> float:
    d, g = c["desc"], GROUPS[c["type"]]
    if g == "comb":
        return d.get("n_side", 0) * (0.3 + d.get("side_alignment", 0)) * (1.2 - min(d.get("spacing_cv", 1), 1)) / (1 + d.get("cycles", 0))
    if g == "feather":
        return d.get("n_side", 0) * (1.2 - min(d.get("spacing_cv", 1), 1)) / (1 + d.get("cycles", 0))
    if g == "crosshatch":
        return d.get("n", 0) * d.get("cross_angle", 0) / 90
    if g == "hatching":
        return d.get("n", 0) * (1.2 - min(d.get("spacing_cv", 1), 1)) * d.get("straightness", 1)
    if g in ("blob", "band"):
        return d.get("area", 0) * (1 + d.get("hatch_bundles", 0))
    if g == "branch":
        return d.get("spine_length", 0) * (1 + d.get("n_side", 0)) / (1 + d.get("cycles", 0))
    if g == "loop":   # a clear ring: curled (the spine nearly closes on itself), big enough, few extra branches
        curl = max(0.0, 1 - min(d.get("spine_straightness", 1), 1))
        return d.get("spine_length", 0) * curl ** 3 / (1 + d.get("n_side", 0) + d.get("n_deep", 0))
    if g == "sprig":
        return d.get("spine_length", 0) * (1 + min(d.get("n_side", 0), 4)) / (1 + d.get("cycles", 0))
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry
# ─────────────────────────────────────────────────────────────────────────────

def chaikin(r: np.ndarray, n: int = 2) -> np.ndarray:
    for _ in range(n):
        q = 0.75 * r + 0.25 * np.roll(r, -1, 0)
        p = 0.25 * r + 0.75 * np.roll(r, -1, 0)
        r = np.empty((2 * len(r), 2))
        r[0::2], r[1::2] = q, p
    return r


def signed_area(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def shapes_of(scene) -> list[tuple[str, list[np.ndarray]]]:
    """A compound's scene as filled rings: ('ink'|'relief', rings)."""
    out = []
    for kind, data, black in scene.items:
        role = "ink" if black else "relief"
        if kind == "rings":
            out.append((role, [chaikin(np.asarray(r, float), 2) for r in data if len(r) >= 3], "mass"))
        elif kind == "stroke":
            pts, w, closed = data
            if len(pts) >= 2:
                out.append((role, offset_rings(pts[:, 0], pts[:, 1], np.asarray(w, float), closed), "stroke"))
        else:
            x, y, r = data
            a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            out.append((role, [np.c_[x + r * np.cos(a), y + r * np.sin(a)]], "disc"))
    return out


def orient(shapes, group: str):
    """Rotate so the long axis is horizontal; combs with teeth hanging down."""
    pts = np.vstack([r for _, rings, _ in shapes for r in rings])
    c = pts.mean(0)
    if group in ("crosshatch", "blob", "loop", "stroke-free"):
        R = np.eye(2)
    else:
        u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        a = np.arctan2(vt[0, 1], vt[0, 0])
        R = np.array([[np.cos(-a), -np.sin(-a)], [np.sin(-a), np.cos(-a)]])
    out = [(role, [(r - c) @ R.T for r in rings], kind) for role, rings, kind in shapes]
    if group == "comb":
        # the spine is the widest ink; if it sits below the mass of the teeth, flip
        allp = np.vstack([r for role, rings, _ in out if role == "ink" for r in rings])
        wide = [r for role, rings, kind in out if role == "ink" and kind == "stroke" for r in rings]
        if wide:
            biggest = max(wide, key=lambda r: abs(signed_area(r)))
            if biggest[:, 1].mean() > np.median(allp[:, 1]):
                out = [(role, [r * np.array([1, -1]) for r in rings], kind) for role, rings, kind in out]
    pts = np.vstack([r for _, rings, _ in out for r in rings])
    lo = pts.min(0)
    return [(role, [r - lo for r in rings], kind) for role, rings, kind in out], pts.max(0) - lo


def hull(rings: list[np.ndarray], W: float, H: float, px: int = 220, close: float = 0.11) -> np.ndarray | None:
    """The silhouette of a cluster of marks: rasterize, close the gaps between
    marks with a disc about `close` of the piece's size, fill holes, take the
    outer contour, smooth it. Returns a ring in piece coordinates."""
    from scipy import ndimage as ndi
    from skimage import measure, morphology
    s = px / max(W, H)
    pad = int(px * close) + 4
    im = Image.new("L", (int(W * s) + 2 * pad, int(H * s) + 2 * pad), 0)
    dr = ImageDraw.Draw(im)
    for r in rings:
        dr.polygon([(x * s + pad, y * s + pad) for x, y in r], fill=255)
    m = np.asarray(im) > 0
    r = max(2, int(px * close))
    # closing with a disc of radius r, via two distance transforms (fast for a large disc)
    dil = ndi.distance_transform_edt(~m) <= r
    m = ndi.distance_transform_edt(dil) > r
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n == 0:
        return None
    m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    cs = measure.find_contours(np.pad(m, 1).astype(float), 0.5)
    c = max(cs, key=len) - 1
    c = measure.approximate_polygon(c, 1.0)
    ring = np.c_[(c[:, 1] - pad) / s, (c[:, 0] - pad) / s]
    return chaikin(ring, 2)


CONVEX_CLOSE = 0.55
CONVEX_GROUPS = ("band", "blob", "hatching", "crosshatch", "comb", "feather", "sprig", "branch")


def convexish(ink: list[np.ndarray], W: float, H: float) -> tuple[np.ndarray | None, dict]:
    """A mostly convex version of a piece (a band, a blob, a cluster of marks): a closing with a disc about
    half the piece's size fills the bays between lumps but keeps the irregular
    overall contour (a closing never goes beyond the convex hull)."""
    from scipy.spatial import ConvexHull
    cx = hull(ink, W, H, close=CONVEX_CLOSE)
    if cx is None:
        return None, {}
    area = abs(signed_area(cx))
    span = cx.max(0) - cx.min(0)
    return cx, {"convexity": round(area / ConvexHull(cx).volume, 3), "rectangularity": round(area / float(span[0] * span[1]), 3)}


def rings_of(d: str) -> list[np.ndarray]:
    return [np.array([[float(a), float(b)] for a, b in re.findall(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", part)]) for part in d.split("M")[1:]]


def refresh_convex() -> None:
    """Recompute only the convex shapes of an existing atlas (fast)."""
    data = json.loads((OUT / "atlas.json").read_text())
    n = 0
    for p in data["pieces"]:
        if p["group"] in CONVEX_GROUPS and p["ink"]:
            cx, met = convexish(rings_of(p["ink"]), p["w"], p["h"])
            p["convex"] = d_of([cx]) if cx is not None else ""
            p.update(met)
            n += 1
    (OUT / "atlas.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"  recomputed {n} convex shapes")


def d_of(rings: list[np.ndarray], k: float = 1.0) -> str:
    f = lambda v: f"{v * k:.2f}".rstrip("0").rstrip(".")
    return " ".join("M" + " L".join(f"{f(x)},{f(y)}" for x, y in r) + " Z" for r in rings)


def piece_svg(p: dict, ink: str = "#000", relief: str = "#fff", pad: float = 0.5) -> str:
    W, H = p["w"] + 2 * pad, p["h"] + 2 * pad
    body = f'<g transform="translate({pad},{pad})"><path fill="{ink}" fill-rule="nonzero" d="{p["ink"]}"/>'
    if p["relief"]:
        body += f'<path fill="{relief}" fill-rule="nonzero" d="{p["relief"]}"/>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.2f} {H:.2f}" width="{W * 8:.0f}" height="{H * 8:.0f}">{body}</g></svg>')


def strokes_from_strokelib(src: Path, lib: Path = HERE / "strokelib" / "vgg16features-whole") -> list[dict]:
    """The representative single strokes of every strokelib family (whole
    marks, both ends free), as atlas pieces in group "stroke": straight and
    wave families turned horizontal, hooks and loops left as drawn."""
    fam = json.loads((lib / "families.json").read_text())["families"]
    tables, out = {}, []
    (OUT / "stroke").mkdir(parents=True, exist_ok=True)
    for f in fam:
        label = f["label"]
        free = any(w in label for w in ("hook", "loop"))
        for j, r in enumerate(f["representatives"]):
            if r["feature"] not in tables:
                tables[r["feature"]] = json.loads((src / f"{r['feature']}.strokes.json").read_text())
            t = tables[r["feature"]]
            s = t["strokes"][r["id"]]
            k = t["k"]
            pts = np.asarray(s["pts"], float) / k
            w = np.asarray(s["w"], float) / k
            rings = offset_rings(pts[:, 0], pts[:, 1], w, s["closed"])
            if not rings:
                continue
            shapes, (W, H) = orient([("ink", rings, "stroke")], "stroke-free" if free else "stroke")
            ink = [r for _, rs, _ in shapes for r in rs]
            pid = f"stroke-{f['id']:02d}-{j:02d}"
            p = {"id": pid, "type": label, "group": "stroke", "family": f["id"], "feature": r["feature"], "w": round(float(W), 2),
                 "h": round(float(H), 2), "aspect": round(float(W / max(H, 1e-6)), 3), "score": 0, "ink": d_of(ink), "relief": "",
                 "outline": d_of([max(ink, key=lambda q: abs(signed_area(q)))]) if s["closed"] else ""}
            (OUT / "stroke" / f"{pid}.svg").write_text(piece_svg(p))
            out.append(p)
    print(f"  strokes from strokelib: {len(out)}", flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "linework" / "vgg16features"))
    ap.add_argument("--per-type", type=int, default=12)
    ap.add_argument("--convex-only", action="store_true", help="only recompute the mostly convex shapes of an existing atlas")
    a = ap.parse_args()
    if a.convex_only:
        refresh_convex()
        return
    src = Path(a.src)
    t0 = time.time()
    found = {}
    files = sorted(src.glob("*.strokes.json"))
    for i, p in enumerate(files):
        d = json.loads(p.read_text())
        for c in C.find_compounds(d, 4.0, 4, 3.0, 2.5):
            if c["type"] not in GROUPS:
                continue
            lo, hi = c["scene"].bounds()
            ext = float((hi - lo).max()) / d["source_px"]
            if ext > 0.55 or ext < 0.04:
                continue
            c["feature"] = d["name"]
            found.setdefault(c["type"], []).append(c)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(files)} features scanned ({time.time() - t0:.0f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pieces = []
    for typ, cs in sorted(found.items()):
        g = GROUPS[typ]
        cs.sort(key=score, reverse=True)
        per_feat, chosen = {}, []
        for c in cs:
            if per_feat.get(c["feature"], 0) >= 2:
                continue
            per_feat[c["feature"]] = per_feat.get(c["feature"], 0) + 1
            chosen.append(c)
            if len(chosen) >= a.per_type:
                break
        (OUT / g).mkdir(exist_ok=True)
        for j, c in enumerate(chosen):
            shapes, (W, H) = orient(shapes_of(c["scene"]), g)
            ink = [r for role, rings, _ in shapes if role == "ink" for r in rings]
            rel = [r for role, rings, _ in shapes if role == "relief" for r in rings]
            masses = [rings for role, rings, kind in shapes if role == "ink" and kind == "mass"]
            outline = ""
            if masses:
                outer = max((r for rings in masses for r in rings), key=lambda r: abs(signed_area(r)))
                outline = d_of([outer])
            elif g == "loop" and ink:   # a loop's outer edge is its outline
                outline = d_of([max(ink, key=lambda r: abs(signed_area(r)))])
            hl = hull(ink, float(W), float(H)) if g in ("hatching", "crosshatch", "comb", "feather", "sprig", "branch", "loop") and ink else None
            # a mostly convex version of a band or blob: a closing with a disc a third of the piece's size fills the
            # bays between lumps but keeps the irregular overall contour (closing never goes beyond the convex hull)
            cx, cmet = convexish(ink, float(W), float(H)) if g in CONVEX_GROUPS and ink else (None, {})
            pid = f"{re.sub(r'[^a-z]+', '-', typ)}-{j:02d}"
            p = {"id": pid, "type": typ, "group": g, "feature": c["feature"], "w": round(float(W), 2), "h": round(float(H), 2),
                 "aspect": round(float(W / max(H, 1e-6)), 3), "score": round(score(c), 2),
                 "ink": d_of(ink), "relief": d_of(rel), "outline": outline, "hull": d_of([hl]) if hl is not None else "",
                 "convex": d_of([cx]) if cx is not None else "", **cmet}
            (OUT / g / f"{pid}.svg").write_text(piece_svg(p))
            pieces.append(p)
        print(f"  {typ:24s} {len(cs):6d} found, {len(chosen)} kept → {g}/", flush=True)
    pieces.extend(strokes_from_strokelib(src))
    (OUT / "atlas.json").write_text(json.dumps({"pieces": pieces}, separators=(",", ":")))

    # contact sheet (each piece rasterized with rsvg-convert, fitted to its cell)
    import subprocess
    import tempfile
    font = ImageFont.load_default(size=13)
    row_of = lambda p: (p["group"], p["type"] + (f" · family {p['family']}" if "family" in p else ""))
    types = sorted({row_of(p) for p in pieces})
    per_row = max(sum(1 for p in pieces if row_of(p) == r) for r in types)
    cell, lw = 120, 230
    sheet = Image.new("RGB", (lw + cell * per_row, cell * len(types)), "white")
    dr = ImageDraw.Draw(sheet)
    with tempfile.TemporaryDirectory() as tmp:
        for row, (grp, typ) in enumerate(types):
            dr.text((10, row * cell + 10), f"{typ}\n→ {grp}", fill=(0, 0, 0), font=font)
            if row:
                dr.line([(0, row * cell), (sheet.width, row * cell)], fill=(225, 225, 225))
            for j, p in enumerate([q for q in pieces if row_of(q) == (grp, typ)]):
                svg = Path(tmp) / "p.svg"
                png = Path(tmp) / "p.png"
                svg.write_text(piece_svg(p))
                big = "-w" if p["aspect"] >= 1 else "-h"
                subprocess.run(["rsvg-convert", big, str(cell - 16), "-b", "white", str(svg), "-o", str(png)], check=True)
                im = Image.open(png).convert("RGB")
                im.thumbnail((cell - 12, cell - 12))
                sheet.paste(im, (lw + j * cell + (cell - im.width) // 2, row * cell + (cell - im.height) // 2))
    sheet.save(OUT / "atlas.png", optimize=True)
    print(f"  {len(pieces)} pieces, {time.time() - t0:.0f}s → {OUT}")

if __name__ == "__main__":
    main()
