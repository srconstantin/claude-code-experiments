#!/usr/bin/env python3
"""
Grammar of Ornament — friezes: a large sheet of borders to choose from.

A frieze repeats one unit along a band. Here the repeats are imperfect in the
way the networks' own repeats are: each step takes the next member of a
*family*, near-identical units cut from the same feature, rather than stamping
one copy.

Units:
  stroke families   a compound piece from the atlas (combs, feathers, sprigs,
                    branches, blobs, hatching, loops), alone or alternating with
                    its nearest piece of the same type (by aspect and size);
                    pieces of one type come from different features, so more
                    than two reads as a jumble rather than as variation
  motif families    a cleaned motif (vgg16motifs-clean/) plus its siblings, the
                    other occurrences of it in its print (recrop.py --siblings)

Arrangements (frieze symmetries, applied to the i-th repeat):
  shift    p1     translation only
  mirror   p1m1   every other repeat flipped left-right
  glide    p11g   every other repeat flipped top-bottom
  turn     p2     every other repeat turned half round
  stack    p11m   two rows, the lower a reflection of the upper
  stack2   p2mm   stack, with every other column mirrored
  drift    —      the unit slowly leans one way then the other along the band
  pulse    —      big and small units alternate (the two-scale rhythm of the prints)
Also: gap or overlap between repeats, a slight hand jitter, rails (none, hairlines,
thick and thin) or a counterband of tiny copies of the same family along both
edges (never a second source: two features' textures together clash), and a
colourway (ink on paper, reversed, or from a convnext feature's opponent pair).

Output: friezes/fNNN.svg (each one band, 1600 × 140 units, scales freely) and
friezes/index.html, the contact sheet with each frieze's recipe.

Run:  python3 friezes.py              # 200 friezes
      python3 friezes.py -n 60 --seed 3
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import math
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color, measure

HERE = Path(__file__).resolve().parent
OUT = HERE / "friezes"
ATLAS = HERE / "designlib" / "atlas" / "atlas.json"
CLEAN = HERE / "vgg16motifs-clean"
COLORWAYS = HERE / "colorways" / "convnext_basefeatures"
W, H = 1600.0, 140.0
STROKE_GROUPS = ("comb", "feather", "sprig", "branch", "blob", "hatching", "crosshatch", "loop")
ARRANGEMENTS = ["shift", "mirror", "glide", "turn", "stack", "stack2", "drift", "pulse"]


def f2(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ─────────────────────────────────────────────────────────────────────────────
#  Units and families
# ─────────────────────────────────────────────────────────────────────────────

def png_path(f: Path) -> tuple[str, float, float]:
    """A one-bit motif PNG (black on white) as one even-odd path in its own pixels."""
    a = np.asarray(Image.open(f).convert("L")) < 128
    h, w = a.shape
    rings = []
    for c in measure.find_contours(np.pad(a, 1).astype(np.float32), 0.5):
        c = measure.approximate_polygon(c - 1, 0.8)
        if len(c) >= 3:
            rings.append("M" + "L".join(f"{x:.0f},{y:.0f}" for y, x in c) + "Z")
    return "".join(rings), float(w), float(h)


def stroke_families(k: int = 2) -> list[dict]:
    pieces = json.loads(ATLAS.read_text())["pieces"]
    by = {}
    for p in pieces:
        if p["group"] in STROKE_GROUPS and p["ink"]:
            by.setdefault((p["group"], p["type"]), []).append(p)
    fams = []
    for (g, t), ps in by.items():
        for p in ps:
            key = lambda q: abs(math.log(float(q["aspect"]) / float(p["aspect"]))) + \
                0.5 * abs(math.log(float(q["w"]) * float(q["h"]) / (float(p["w"]) * float(p["h"]))))
            sib = sorted((q for q in ps if q is not p), key=key)[:k - 1]
            members = [{"d": q["ink"], "w": float(q["w"]), "h": float(q["h"]), "rule": "nonzero"} for q in [p] + sib]
            fams.append({"kind": "stroke", "label": f"{t} · {p['id']}", "group": g, "id": p["id"],
                         "aspect": float(p["aspect"]), "members": members})
    return fams


def motif_families() -> list[dict]:
    fams = []
    for j in sorted(CLEAN.glob("*/*/motif.json")):
        rec = json.loads(j.read_text())
        files = [j.parent / "motif.png"] + [j.parent / s["file"] for s in rec.get("siblings", [])]
        members = []
        for f in files:
            if not f.exists():
                continue
            d, w, h = png_path(f)
            members.append({"d": d, "w": w, "h": h, "rule": "evenodd"})
        short = re.sub(r"__224px.*", "", rec["feature"]).replace("vgg16__", "")
        fams.append({"kind": "motif", "label": f"motif {short} ({len(members) - 1} siblings)", "group": "motif",
                     "id": rec["id"], "aspect": members[0]["w"] / members[0]["h"], "members": members})
    return fams


# ─────────────────────────────────────────────────────────────────────────────
#  Colour
# ─────────────────────────────────────────────────────────────────────────────

def lab_hex(L: float, a: float, b: float) -> str:
    rgb = np.clip(color.lab2rgb(np.array([[[L, a, b]]], float))[0, 0], 0, 1)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


def hue_of(lab, chroma: float) -> tuple[float, float]:
    a, b = lab[1], lab[2]
    n = math.hypot(a, b) or 1.0
    return a / n * chroma, b / n * chroma


def colourways() -> list[dict]:
    out = []
    for f in sorted(COLORWAYS.glob("*.json")):
        j = json.loads(f.read_text())
        if not isinstance(j, dict) or not isinstance(j.get("pair"), dict):
            continue
        pa, pb = j["pair"]["lab_a"], j["pair"]["lab_b"]
        ia, ib = hue_of(pa, 30), hue_of(pb, 7)
        name = j.get("short", f.stem)
        out.append({"name": name, "ink": lab_hex(22, *ia), "paper": lab_hex(93, *ib),
                    "ink2": lab_hex(40, *hue_of(pb, 34)), "deep": lab_hex(17, *hue_of(pa, 16)), "light": lab_hex(88, *hue_of(pb, 14))})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  One frieze
# ─────────────────────────────────────────────────────────────────────────────

def frieze(fam: dict, rng: random.Random, cw: dict | None, small: dict | None,
           arr: str | None = None, scheme: str | None = None, rails: str | None = None) -> tuple[str, dict]:
    """One frieze. arr, scheme and rails may be fixed by the caller; otherwise they are drawn at random."""
    fixed = arr is not None
    arr = arr or rng.choice(ARRANGEMENTS)
    if not fixed and fam["aspect"] > 2.6 and arr in ("stack", "stack2", "pulse"):
        arr = rng.choice(["shift", "mirror", "glide", "turn"])
    rails = rails or rng.choice(["none", "none", "hair", "double", "counter"])
    gap = rng.choice([-0.18, -0.06, 0.0, 0.12, 0.3, 0.55]) if fam["kind"] == "stroke" else rng.choice([-0.04, 0.06, 0.18, 0.4])
    jitter = rng.choice([0.0, 0.0, 2.5, 5.0])        # degrees
    lie = fam["aspect"] < 0.55 and rng.random() < 0.5   # lay tall units on their side
    scheme = "plain" if cw is None else (scheme or rng.choice(["ink", "ink", "reverse", "two"]))
    if cw is None:
        ground, ink, ink2, rail = "#f4f1ea", "#151413", "#151413", "#151413"
    elif scheme == "reverse":
        ground, ink, ink2, rail = cw["deep"], cw["light"], cw["light"], cw["light"]
    else:
        ground, ink, rail = cw["paper"], cw["ink"], cw["ink"]
        ink2 = cw["ink2"] if scheme == "two" else cw["ink"]

    top, bot = 10.0, H - 10.0
    if rails in ("hair", "double"):
        top, bot = 22.0, H - 22.0
    if rails == "counter":
        top, bot = 30.0, H - 30.0
    band_h = bot - top
    stacked = arr in ("stack", "stack2")
    row_h = band_h / 2 if stacked else band_h

    defs, body = [], []
    members = fam["members"]
    if fam["kind"] == "stroke" and rng.random() < 0.5:
        members = members[:1]
    for k, m in enumerate(members):
        defs.append(f'<path id="u{k}" fill-rule="{m["rule"]}" d="{m["d"]}"/>')

    def place(k: int, cx: float, cy: float, box_h: float, fx: int, fy: int, rot: float, fill: str, scale: float = 1.0) -> float:
        m = members[k % len(members)]
        w, h = (m["h"], m["w"]) if lie else (m["w"], m["h"])
        s = box_h / h * scale * (0.86 if arr == "drift" or jitter else 1.0)
        s = min(s, 3.2 * box_h / w)                  # very long units are capped in length
        tr = f"translate({f2(cx)},{f2(cy)}) rotate({f2(rot + (90 if lie else 0))}) scale({f2(fx * s)},{f2(fy * s)}) translate({f2(-m['w'] / 2)},{f2(-m['h'] / 2)})"
        body.append(f'<use href="#u{k % len(members)}" fill="{fill}" transform="{tr}"/>')
        return w * s

    x, i = 0.0, 0
    mean_w = None
    while x < W + 40:
        k = i
        fx = -1 if (arr == "mirror" and i % 2) or (arr == "stack2" and i % 2) else 1
        fy = -1 if arr == "glide" and i % 2 else 1
        rot = 180.0 if arr == "turn" and i % 2 else 0.0
        if arr == "drift":
            rot += 14 * math.sin(i * 2 * math.pi / 9)
        rot += rng.uniform(-jitter, jitter)
        scale = 0.62 if arr == "pulse" and i % 2 else 1.0
        fill = ink2 if i % 2 else ink
        m = members[k % len(members)]
        w0, h0 = (m["h"], m["w"]) if lie else (m["w"], m["h"])
        s0 = min(row_h / h0 * scale, 3.2 * row_h / w0)
        uw = w0 * s0
        cx = x + uw / 2
        if stacked:
            place(k, cx, top + row_h / 2, row_h, fx, 1, rot, fill, scale)
            place(k, cx, top + 1.5 * row_h, row_h, fx, -1, -rot, fill, scale)
        else:
            cy = top + band_h / 2 + (band_h * (1 - scale) / 2 if arr == "pulse" else 0)
            place(k, cx, cy, row_h, fx, fy, rot, fill, scale)
        mean_w = uw if mean_w is None else 0.8 * mean_w + 0.2 * uw
        x += uw + gap * (mean_w if arr != "pulse" else max(uw, mean_w))
        i += 1

    if rails == "hair":
        for y in (12, H - 12):
            body.append(f'<rect x="0" y="{y - 0.9}" width="{W}" height="1.8" fill="{rail}"/>')
    elif rails == "double":
        body.append(f'<rect x="0" y="6" width="{W}" height="5" fill="{rail}"/><rect x="0" y="15" width="{W}" height="1.4" fill="{rail}"/>')
        body.append(f'<rect x="0" y="{H - 11}" width="{W}" height="5" fill="{rail}"/><rect x="0" y="{H - 16.4}" width="{W}" height="1.4" fill="{rail}"/>')
    elif rails == "counter":
        sm = members                                    # tiny copies of the frieze's own family: one source per frieze
        for k, m in enumerate(sm):
            defs.append(f'<path id="c{k}" fill-rule="{m["rule"]}" d="{m["d"]}"/>')
        for y, fy in ((14, 1), (H - 14, -1)):
            x2, j = 0.0, 0
            while x2 < W + 20:
                m = sm[j % len(sm)]
                s = 18 / max(m["h"], m["w"] * 0.25)
                s = min(s, 26 / m["w"]) if m["w"] * s > 26 else s
                body.append(f'<use href="#c{j % len(sm)}" fill="{rail}" transform="translate({f2(x2 + m["w"] * s / 2)},{y}) '
                            f'scale({f2(s)},{f2(fy * s)}) translate({f2(-m["w"] / 2)},{f2(-m["h"] / 2)})"/>')
                x2 += m["w"] * s + 6
                j += 1

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}">'
           f'<defs>{"".join(defs)}<clipPath id="band"><rect width="{W:.0f}" height="{H:.0f}"/></clipPath></defs>'
           f'<rect width="100%" height="100%" fill="{ground}"/><g clip-path="url(#band)">{"".join(body)}</g></svg>')
    recipe = {"unit": fam["label"], "kind": fam["kind"], "source": fam["id"], "family": len(members), "arrangement": arr,
              "gap": gap, "jitter": jitter, "on_side": lie, "rails": rails,
              "colour": "plain" if cw is None else f"{scheme} · {cw['name']}"}
    return svg, recipe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--motif-share", type=float, default=0.5)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    strokes = stroke_families()
    motifs = motif_families()
    cws = colourways()
    smalls = [f for f in strokes if f["group"] in ("hatching", "loop", "sprig") and f["aspect"] < 1.6]
    print(f"{len(strokes)} stroke families, {len(motifs)} motif families "
          f"({sum(len(m['members']) - 1 for m in motifs)} siblings), {len(cws)} colourways")
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("f[0-9][0-9][0-9].svg"):        # only this script's own output
        old.unlink()
    # spread the stroke families evenly over their groups so no one type crowds the sheet
    groups = {}
    for f in strokes:
        groups.setdefault(f["group"], []).append(f)
    for g in groups.values():
        rng.shuffle(g)
    motif_order = motifs[:]
    rng.shuffle(motif_order)
    rows = []
    gi = 0
    for n in range(a.n):
        if motifs and rng.random() < a.motif_share:
            fam = motif_order[n % len(motif_order)]
        else:
            keys = sorted(groups)
            g = groups[keys[gi % len(keys)]]
            fam = g[(gi // len(keys)) % len(g)]
            gi += 1
        cw = rng.choice(cws) if cws and rng.random() < 0.45 else None
        svg, rec = frieze(fam, rng, cw, rng.choice(smalls) if smalls else None)
        name = f"f{n:03d}"
        (OUT / f"{name}.svg").write_text(svg)
        rec["file"] = f"{name}.svg"
        rows.append(rec)
        if n % 20 == 0:
            print(f"  {name}  {rec['arrangement']:7s} {rec['rails']:8s} {rec['colour'][:22]:22s} {rec['unit'][:60]}", flush=True)
    (OUT / "friezes.json").write_text(json.dumps(rows, indent=1))
    cards = "\n".join(
        f'<figure><img loading="lazy" src="{r["file"]}" alt=""><figcaption><b>{r["file"][:-4]}</b> {html.escape(r["unit"])} · '
        f'{r["arrangement"]} · gap {r["gap"]} · {r["rails"]}{" · on its side" if r["on_side"] else ""} · {html.escape(r["colour"])}</figcaption></figure>'
        for r in rows)
    (OUT / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Friezes</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--paper:#eceae4;--ink:#1a1918;--muted:#6b6862}}
body{{margin:0;background:var(--paper);color:var(--ink);font:13px/1.4 ui-monospace,Menlo,monospace}}
header{{padding:28px 24px 8px}} h1{{font:600 28px/1 Georgia,serif;margin:0 0 6px}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,720px),1fr));gap:22px 28px;padding:16px 24px 60px}}
figure{{margin:0}} img{{width:100%;display:block;box-shadow:0 0 0 1px #0001}}
figcaption{{color:var(--muted);margin-top:5px;font-size:11px}} b{{color:var(--ink)}}
</style></head><body><header><h1>Friezes</h1><div>{len(rows)} friezes from {len(strokes)} stroke families and {len(motifs)} motif families.
Each repeat takes the next member of its family, so no two neighbours are identical. Recipes in friezes.json.</div></header>
<main>{cards}</main></body></html>""")
    print(f"wrote {len(rows)} friezes and {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
