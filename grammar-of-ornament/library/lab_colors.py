#!/usr/bin/env python3
"""Lab: colorways drawn from the image itself (palette.py), one card per colorway feature.

  .venv/bin/python library/lab_colors.py   → http://127.0.0.1:8812/lab/colors.html
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from palette import palette

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "site" / "lab"
cat = json.loads((HERE / "site" / "catalog.js").read_text()[len("window.NEURON = "):-2])


def main():
    cards = []
    for c in cat["colorways"]:
        f = ROOT / "convnext_basefeatures" / f"{c['name']}.png"
        rgb = np.asarray(Image.open(f).convert("RGB")).astype(float) / 255
        p = palette(rgb)
        cards.append((c, p))
        print(f"  {c['short']:12} base {p['base']}  five {' '.join(p['five'])}  {p['note']}", flush=True)
    print("tinted grounds:", sum(p["tint"] != "" for _, p in cards), "· adjusted colours in", sum(p["adjusted"] > 0 for _, p in cards), "of", len(cards), "cards")
    html = ["""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Lab: colours from the image</title>
<link rel="stylesheet" href="../neuron-library/neuron.css?v=lab">
<style>
body{margin:0;font:15px/1.45 var(--n-text);background:#e7e4dd;color:#1b1a18}
.wrap{max-width:1500px;margin:0 auto;padding:30px 28px 80px}
h1{font:400 34px var(--n-text);margin:0 0 4px}
.lede{max-width:800px;color:#3d3a35;margin:0 0 22px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:18px}
.card{background:#f4f2ed}
.top{display:grid;grid-template-columns:112px 1fr}
.top img{width:112px;height:112px;object-fit:cover;display:block}
.sw{display:flex;height:56px}
.sw i{flex:1;position:relative}
.sw i::after{content:attr(data-h);position:absolute;left:0;right:0;bottom:3px;text-align:center;font:9px var(--n-mono);color:#fff;mix-blend-mode:difference;letter-spacing:0}
.base{height:56px;position:relative}
.base::after{content:'base · ' attr(data-h);position:absolute;left:8px;bottom:4px;font:10px var(--n-mono);color:#fff;mix-blend-mode:difference}
.demo{padding:14px 16px;min-height:96px;position:relative}
.demo .chip{position:absolute;right:14px;top:14px;width:40px;height:40px;display:block}
.demo b{font:normal 34px/1 var(--n-display);display:block;margin-bottom:6px}
.demo span{font:500 12px var(--n-label);letter-spacing:.1em;text-transform:lowercase}
.cap{display:flex;justify-content:space-between;gap:10px;padding:8px 10px;font:11px var(--n-mono);color:#6e6a62}
.old{display:flex;height:14px}.old i{flex:1}
</style></head><body><div class="wrap">
<h1>Colours from the image</h1>
<p class="lede">For each colorway feature: the picture; its <b>base</b>, the biggest patch of colour in it; <b>five colours present
in the picture</b> (real pixels, one per patch of colour, chosen as far apart in Lab as the image allows, at a similar
saturation, each legible against the ground, ordered by hue); a panel using them, the ground (the base, or a tint of it
when the base can't carry five of the image's colours, the base then shown as the square) and the five for the type;
and, as the thin strip, the <b>current colorway</b> (paper, ink, accent, accent2, dark) for comparison. Numbers: the chroma
and lightness range of the five, how many patches qualified, and the contrast range of the five against the ground.</p>
<div class="grid">"""]
    for c, p in cards:
        five, base, ground = p["five"], p["base"], p["ground"]
        sw = "".join(f'<i style="background:{h}" data-h="{h[1:]}"></i>' for h in five)
        old = "".join(f'<i style="background:{c[k]}"></i>' for k in ("paper", "ink", "accent", "accent2", "dark"))
        chip = "" if ground == base else f'<i class="chip" style="background:{base}" title="base, as a panel"></i>'
        html.append(f'''<div class="card"><div class="top"><img src="../{c['feature']}" alt="">
  <div><div class="base" style="background:{base}" data-h="{base[1:]}"></div><div class="sw">{sw}</div></div></div>
  <div class="demo" style="background:{ground}">{chip}<b style="color:{five[0]}">Aa <span style="color:{five[1]}">Bb</span> <span style="color:{five[2]}">Cc</span> <span style="color:{five[3]}">Dd</span> <span style="color:{five[4]}">Ee</span></b>
  <span style="color:{five[2]}">{c['short']} · {c['name_a']} / {c['name_b']}</span></div>
  <div class="old">{old}</div><div class="cap"><span>{c['short']}</span><span>{p['note']}</span></div></div>''')
    html.append("</div></div></body></html>")
    (OUT / "colors.html").write_text("\n".join(html))
    print("wrote", OUT / "colors.html", len(cards), "cards")


if __name__ == "__main__":
    main()
