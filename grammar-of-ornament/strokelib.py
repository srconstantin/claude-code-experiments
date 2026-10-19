#!/usr/bin/env python3
"""
Grammar of Ornament — strokelib: the families of strokes in a feature set, and
representative strokes from each.

Reads the per-stroke tables that linework.py writes (<name>.strokes.json),
describes every stroke by its shape — length and width (log), straightness,
bend, inflections, total turn, taper, width unevenness, closed or not — and
clusters the standardized descriptors with k-means into --k families. Each
family gets a label read off its typical member ("long tapered wave, fine"),
and its --per-family representatives are the members nearest its centre, no
more than two from any one feature.

Every representative is written as an SVG of its own: the stroke as a filled
outline with its measured width at every point (times --width-scale, no
minimum), in its original orientation, with a margin. Nothing is made constant
width.

Written into <dst>/:
  families.json      every family: label, size, share, typical measurements,
                     how many strokes each layer contributes, the representatives
  families.csv       one row per family
  by_feature.csv     per feature, the share of its strokes in each family
  families.png       contact sheet: one row per family, its representatives
  family-NN-<label>/ stroke-MM.svg for each representative, sheet.png

Run:  python3 strokelib.py                                   # linework/vgg16features → strokelib/vgg16features
      python3 strokelib.py --k 16 --per-family 24 --width-scale 0.6
      python3 strokelib.py --pol ink --min-length 4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.vq import kmeans2, vq

from linework import offset_rings, rings_path, f1

HERE = Path(__file__).resolve().parent
DESCRIPTORS = ["log_length", "log_width", "straightness", "bend", "inflections", "turn", "taper", "width_cv", "closed", "free_ends"]


def free_ends(s: dict) -> int:
    """How many of the stroke's ends are its own (not cut at a junction with another mark)."""
    return 0 if s["closed"] else int(not s["j0"]) + int(not s["j1"])


# ─────────────────────────────────────────────────────────────────────────────
#  Loading and describing
# ─────────────────────────────────────────────────────────────────────────────

def load(src: Path, pol: str, min_length: float, ends: str) -> tuple[list[dict], list[dict]]:
    """All strokes of all features in src, each stroke tagged with its feature."""
    feats, strokes = [], []
    need = {"any": 0, "one-free": 1, "both-free": 2}[ends]
    for p in sorted(src.glob("*.strokes.json")):
        d = json.loads(p.read_text())
        feats.append({k: d[k] for k in ("name", "model", "layer", "k", "print_px", "source_px")})
        fi = len(feats) - 1
        for s in d["strokes"]:
            if s["pol"] == "mass" or (pol != "both" and s["pol"] != pol):   # mass spines are compounds.py's business
                continue
            if s["length"] < min_length or len(s["pts"]) < 2:
                continue
            if need and not s["closed"] and free_ends(s) < need:
                continue
            s["f"] = fi
            strokes.append(s)
    return feats, strokes


def describe(strokes: list[dict]) -> np.ndarray:
    X = np.empty((len(strokes), len(DESCRIPTORS)), np.float32)
    for i, s in enumerate(strokes):
        ends = [t for t in (s["taper_start"], s["taper_end"]) if t is not None]
        X[i] = (np.log(max(s["length"], 0.1)), np.log(max(s["width"], 0.05)), s["straightness"], min(s["bend"], 3.0),
                min(s["inflections"], 4), min(abs(s["turn"]) / np.pi, 2.0), float(np.mean(ends)) if ends else 1.0,
                min(s["width_cv"], 1.0), float(s["closed"]), free_ends(s))
    return X


def cluster(X: np.ndarray, k: int, seed: int, fit_max: int = 60000, restarts: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """k-means on standardized descriptors. Returns labels, centres (standardized), mean, std."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    rng = np.random.default_rng(seed)
    fit = Z[rng.choice(len(Z), min(fit_max, len(Z)), replace=False)]
    best = None
    for r in range(restarts):
        c, lab = kmeans2(fit, k, minit="++", seed=seed + r)
        sse = float(((fit - c[lab]) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, c)
    centres = best[1]
    labels, _ = vq(Z, centres)
    # order families by size, largest first
    order = np.argsort(-np.bincount(labels, minlength=k))
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[labels], centres[order], mu, sd


# ─────────────────────────────────────────────────────────────────────────────
#  Naming
# ─────────────────────────────────────────────────────────────────────────────

def label_for(m: dict) -> str:
    """A name read off a family's typical measurements (source px)."""
    aspect = m["length"] / max(m["width"], 0.05)
    size = "dash" if aspect < 4 else "short" if aspect < 12 else "medium" if aspect < 30 else "long"
    if m["closed"] > 0.5:
        form = "loop"
    elif m["turn"] > 4.2:
        form = "curl"
    elif m["turn"] > 1.9:
        form = "hook"
    elif m["inflections"] >= 2:
        form = "wave"
    elif m["inflections"] >= 1 and m["straightness"] < 0.9:
        form = "S-curve"
    elif m["straightness"] > 0.95:
        form = "straight"
    elif m["straightness"] > 0.8:
        form = "gentle arc"
    else:
        form = "arc"
    weight = "hairline" if m["width"] < 1.0 else "fine" if m["width"] < 2.0 else "medium" if m["width"] < 3.5 else "bold"
    adj = []
    if m["taper"] < 0.65:
        adj.append("tapered")
    elif m["taper"] > 1.35:
        adj.append("flared")
    if m["width_cv"] > 0.4:
        adj.append("uneven")
    if m["closed"] <= 0.5 and m["free_ends"] < 0.5:
        form += " segment"           # cut at a junction at both ends: a piece of a mesh, not a whole mark
    elif m["closed"] <= 0.5 and m["free_ends"] < 1.5:
        form += " branch"            # one end free, the other joins another mark
    return f"{size} {' '.join(adj) + ' ' if adj else ''}{form}, {weight}"


def typical(members: list[dict]) -> dict:
    def med(key, f=lambda s, k: s[k]):
        return float(np.median([f(s, key) for s in members]))
    ends = [t for s in members for t in (s["taper_start"], s["taper_end"]) if t is not None]
    return {"length": med("length"), "width": med("width"), "width_min": med("width_min"),
            "straightness": med("straightness"), "bend": med("bend"), "curvature": med("curvature"),
            "turn": med("turn", lambda s, k: abs(s[k])), "inflections": med("inflections"),
            "taper": float(np.median(ends)) if ends else 1.0, "width_cv": med("width_cv"),
            "closed": float(np.mean([s["closed"] for s in members])),
            "free_ends": float(np.mean([free_ends(s) for s in members]))}


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────────────────────────────────────

class Stroke:
    """A representative stroke in source pixels: centerline, widths, outline rings."""

    def __init__(self, s: dict, k: float, width_scale: float):
        self.pts = np.asarray(s["pts"], float) / k
        self.w = np.asarray(s["w"], float) * width_scale / k
        self.closed = s["closed"]
        self.rings = offset_rings(self.pts[:, 0], self.pts[:, 1], self.w, self.closed)
        allp = np.vstack(self.rings)
        self.lo, self.hi = allp.min(0), allp.max(0)

    def svg(self, margin: float, px_per_unit: float = 8.0) -> str:
        """The stroke as one filled path (nonzero, so a stroke that crosses itself
        stays solid), in source-pixel units scaled by px_per_unit."""
        x0, y0 = self.lo - margin
        W, H = (self.hi - self.lo) + 2 * margin
        shifted = [(r - (x0, y0)) * px_per_unit for r in self.rings]
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {f1(W * px_per_unit)} {f1(H * px_per_unit)}" '
                f'width="{f1(W * px_per_unit)}" height="{f1(H * px_per_unit)}">'
                f'<path fill="#000" fill-rule="nonzero" d="{rings_path(shifted)}"/></svg>')

    def raster(self, cell: int, scale: float, pad: int = 4, ss: int = 3) -> Image.Image:
        """The stroke drawn at `scale` px per source px (shrunk if it would not
        fit), as discs and quads along the centerline, so self-overlaps are solid."""
        span = self.hi - self.lo
        scale = min(scale, (cell - 2 * pad) / max(float(span.max()), 1e-6))
        off = (np.array([cell, cell]) - span * scale) / 2 - self.lo * scale
        im = Image.new("L", (cell * ss, cell * ss), 255)
        dr = ImageDraw.Draw(im)
        P = (self.pts * scale + off) * ss
        R = self.w * scale * ss / 2
        th = np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0])) if len(P) > 1 else np.zeros(1)
        nx, ny = np.sin(th), -np.cos(th)
        pairs = list(zip(range(len(P) - 1), range(1, len(P)))) + ([(len(P) - 1, 0)] if self.closed else [])
        for i, j in pairs:
            dr.polygon([(P[i, 0] + nx[i] * R[i], P[i, 1] + ny[i] * R[i]), (P[j, 0] + nx[j] * R[j], P[j, 1] + ny[j] * R[j]),
                        (P[j, 0] - nx[j] * R[j], P[j, 1] - ny[j] * R[j]), (P[i, 0] - nx[i] * R[i], P[i, 1] - ny[i] * R[i])], fill=0)
        for (x, y), r in zip(P, R):
            dr.ellipse([x - r, y - r, x + r, y + r], fill=0)
        return im.resize((cell, cell), Image.LANCZOS)


def sheet(rows: list[tuple[str, list[Stroke]]], cell: int, scale: float, label_w: int = 300) -> Image.Image:
    """One row per family; every stroke drawn at the same scale."""
    font = ImageFont.load_default(size=13)
    n = max(len(r[1]) for r in rows)
    W, H = label_w + n * cell, len(rows) * cell
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    for i, (text, strokes) in enumerate(rows):
        y = i * cell
        if i:
            dr.line([(0, y), (W, y)], fill=(225, 225, 225))
        for j, line in enumerate(text.split("\n")):
            dr.text((10, y + 10 + 16 * j), line, fill=(0, 0, 0) if j == 0 else (110, 110, 110), font=font)
        for j, st in enumerate(strokes):
            im.paste(st.raster(cell, scale).convert("RGB"), (label_w + j * cell, y))
    # scale bar: ten source pixels
    dr.line([(10, H - 12), (10 + 10 * scale, H - 12)], fill=(0, 0, 0), width=2)
    dr.text((14 + 10 * scale, H - 20), "10 source px", fill=(0, 0, 0), font=font)
    return im


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "linework" / "vgg16features"))
    ap.add_argument("--dst", default="", help="default: strokelib/<src folder name>")
    ap.add_argument("--k", type=int, default=12, help="number of families")
    ap.add_argument("--per-family", type=int, default=16, help="representatives per family")
    ap.add_argument("--width-scale", type=float, default=1.0, help="multiply every stroke's measured width (1 = as measured)")
    ap.add_argument("--pol", choices=["both", "ink", "relief"], default="both", help="black strokes, white relief marks, or both")
    ap.add_argument("--min-length", type=float, default=0.0, help="ignore strokes shorter than this (source px)")
    ap.add_argument("--ends", choices=["any", "one-free", "both-free"], default="any",
                    help="keep only strokes with at least one, or both, ends free (not cut at a junction); loops always kept")
    ap.add_argument("--cell", type=int, default=96, help="contact-sheet cell size in px")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = Path(a.src)
    dst = Path(a.dst) if a.dst else HERE / "strokelib" / src.name
    dst.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    feats, strokes = load(src, a.pol, a.min_length, a.ends)
    if not strokes:
        print(f"no strokes in {src} (run linework.py first)")
        sys.exit(1)
    print(f"  {len(strokes)} strokes from {len(feats)} features ({time.time() - t0:.0f}s)", flush=True)
    X = describe(strokes)
    labels, centres, mu, sd = cluster(X, a.k, a.seed)
    Z = (X - mu) / sd
    print(f"  clustered into {a.k} families ({time.time() - t0:.0f}s)", flush=True)

    layers = sorted({f["layer"] for f in feats}, key=lambda l: [int(p) if p.isdigit() else p for p in l.split(".")])
    families, rows, picked = [], [], []
    for fam in range(a.k):
        idx = np.nonzero(labels == fam)[0]
        members = [strokes[i] for i in idx]
        m = typical(members)
        label = label_for(m)
        dist = np.linalg.norm(Z[idx] - centres[fam], axis=1)
        reps, per_feat = [], {}
        for i in idx[np.argsort(dist)]:
            s = strokes[i]
            if per_feat.get(s["f"], 0) >= 2:
                continue
            per_feat[s["f"]] = per_feat.get(s["f"], 0) + 1
            reps.append(s)
            if len(reps) >= a.per_family:
                break
        fdir = dst / f"family-{fam:02d}-{slug(label)}"
        fdir.mkdir(exist_ok=True)
        shapes, rep_items = [], []
        for j, s in enumerate(reps):
            st = Stroke(s, feats[s["f"]]["k"], a.width_scale)
            if not st.rings:
                continue
            (fdir / f"stroke-{j:02d}.svg").write_text(st.svg(margin=max(s["width_max"] * a.width_scale, 0.5)))
            shapes.append(st)
            rep_items.append({"svg": f"{fdir.name}/stroke-{j:02d}.svg", "feature": feats[s["f"]]["name"],
                              "layer": feats[s["f"]]["layer"], "id": s["id"], "pol": s["pol"], "free_ends": free_ends(s),
                              **{key: s[key] for key in ("length", "width", "straightness", "bend", "turn", "inflections", "closed")}})
        by_layer = {ly: int(sum(feats[s["f"]]["layer"] == ly for s in members)) for ly in layers}
        pol = {"ink": int(sum(s["pol"] == "ink" for s in members)), "relief": int(sum(s["pol"] == "relief" for s in members))}
        families.append({"id": fam, "label": label, "dir": fdir.name, "n": len(members), "share": len(members) / len(strokes),
                         "typical": m, "pol": pol, "by_layer": by_layer, "representatives": rep_items})
        picked.append((fdir, shapes))
        rows.append((f"{fam:02d}  {label}\n{len(members)} strokes ({100 * len(members) / len(strokes):.0f}%)  "
                     f"L {m['length']:.1f}  w {m['width']:.2f}", shapes))
        print(f"  family {fam:02d}: {label:44s} {len(members):7d}  L {m['length']:5.1f}  w {m['width']:.2f}  "
              f"straight {m['straightness']:.2f}  turn {np.degrees(m['turn']):4.0f}°  taper {m['taper']:.2f}", flush=True)

    # one scale for every sheet: the 90th-percentile representative fits a cell; longer ones are shrunk to fit
    spans = [float((st.hi - st.lo).max()) for _, shapes in picked for st in shapes]
    scale = (a.cell - 8) / max(float(np.percentile(spans, 90)), 1.0) if spans else 1.0
    sheet(rows, a.cell, scale).save(dst / "families.png", optimize=True)
    for (fdir, shapes), row in zip(picked, rows):
        sheet([row], a.cell * 2, scale * 2, 360).save(fdir / "sheet.png", optimize=True)
    (dst / "families.json").write_text(json.dumps(
        {"src": str(src), "k": a.k, "pol": a.pol, "min_length": a.min_length, "width_scale": a.width_scale,
         "n_strokes": len(strokes), "descriptors": DESCRIPTORS, "layers": layers, "families": families}, indent=1))
    with open(dst / "families.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        tkeys = list(families[0]["typical"])
        w.writerow(["id", "label", "n", "share", "ink", "relief"] + tkeys + layers)
        for f in families:
            w.writerow([f["id"], f["label"], f["n"], f"{f['share']:.4f}", f["pol"]["ink"], f["pol"]["relief"]]
                       + [f"{f['typical'][k]:.3f}" for k in tkeys] + [f["by_layer"][ly] for ly in layers])
    with open(dst / "by_feature.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "layer", "n_strokes"] + [f"family_{f['id']:02d}" for f in families])
        for fi, f in enumerate(feats):
            mine = labels[[i for i, s in enumerate(strokes) if s["f"] == fi]]
            cnt = np.bincount(mine, minlength=a.k)
            w.writerow([f["name"], f["layer"], len(mine)] + [f"{c / max(len(mine), 1):.3f}" for c in cnt])

    print(f"\n  share of each layer's strokes by family (%):")
    print("  " + "layer".ljust(14) + "".join(f"{f['id']:>6d}" for f in families))
    for ly in layers:
        tot = sum(f["by_layer"][ly] for f in families)
        print("  " + ly.ljust(14) + "".join(f"{100 * f['by_layer'][ly] / max(tot, 1):6.0f}" for f in families))
    print(f"\n  {time.time() - t0:.0f}s → {dst}")


if __name__ == "__main__":
    main()
