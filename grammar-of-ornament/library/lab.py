#!/usr/bin/env python3
"""Lab: four feature-derived treatments, each measured from one feature, as a demo page (library/site/lab/).

  edge sequence   the ordered colours across the feature's edges (dark side → light side)
  ground glass    the feature low-passed until only its large colour layout remains
  grain           the feature's dominant orientation and repeat period
  iridescence     the colorway's ramp as a conic border, with a specular sheen

  .venv/bin/python library/lab.py   → http://127.0.0.1:8812/lab/
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "site" / "lab"
OUT.mkdir(parents=True, exist_ok=True)
cat = json.loads((HERE / "site" / "catalog.js").read_text()[len("window.NEURON = "):-2])
CW = {c["short"]: c for c in cat["colorways"]}


def load(short):
    c = CW[short]
    im = np.asarray(Image.open(ROOT / "cache" / "sr" / f"{c['name']}.png").convert("RGB")).astype(float) / 255
    return c, im


def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in np.clip(rgb, 0, 1))


def kmeans(X, k, it=20, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return C


# ---------------------------------------------------------------- 1. edge sequence

def edge_sequence(im, n_off=8, step=2.0, k=20):
    """Colours met crossing the feature's strong edges, from the dark side to the light side, as (a) the median
    colour at each offset (a smooth profile) and (b) the most common quantized colour at each offset (a sequence
    of real colours, consecutive repeats collapsed)."""
    L = color.rgb2lab(im)[..., 0]
    gx, gy = ndi.gaussian_filter(L, 1.5, order=(0, 1)), ndi.gaussian_filter(L, 1.5, order=(1, 0))
    mag = np.hypot(gx, gy)
    ys, xs = np.nonzero(mag > np.percentile(mag, 93))
    nx, ny = gx[ys, xs] / mag[ys, xs], gy[ys, xs] / mag[ys, xs]          # toward the brighter side
    offs = np.arange(-n_off, n_off + 1) * step
    prof = np.zeros((len(offs), len(xs), 3))
    for i, d in enumerate(offs):
        px, py = xs + d * nx, ys + d * ny
        for ch in range(3):
            prof[i, :, ch] = ndi.map_coordinates(im[..., ch], [py, px], order=1, mode="nearest")
    lab = color.rgb2lab(prof)
    med = color.lab2rgb(np.median(lab, axis=1)[None])[0]                   # smooth profile, dark → light
    rng = np.random.default_rng(0)
    C = kmeans(color.rgb2lab(im.reshape(-1, 3)[rng.choice(im.shape[0] * im.shape[1], 6000, replace=False)][None])[0], k)
    seq = []
    for i in range(len(offs)):
        d = ((lab[i][:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        j = np.bincount(d, minlength=k).argmax()
        if not seq or seq[-1] != j:
            seq.append(j)
    cols = [hexc(color.lab2rgb(C[j][None, None])[0, 0]) for j in seq]
    return [hexc(m) for m in med], cols


# ---------------------------------------------------------------- 2. ground glass

def ground_glass(c, im, name):
    """The feature low-passed at two strengths; from the seamless tile when there is one, so it still tiles."""
    tile = ROOT / "cache" / "sr_tile" / f"{c['name']}.png"
    src, wrap = (np.asarray(Image.open(tile).convert("RGB")).astype(float) / 255, True) if tile.exists() else (im, False)
    out = {}
    for label, sigma in (("fine", 14), ("coarse", 36)):
        g = np.stack([ndi.gaussian_filter(src[..., ch], sigma, mode="wrap" if wrap else "reflect") for ch in range(3)], -1)
        f = OUT / f"{name}-glass-{label}.jpg"
        Image.fromarray((g * 255).round().astype(np.uint8)).resize((448, 448), Image.LANCZOS).save(f, quality=88)
        Lm = color.rgb2lab(g)[..., 0]
        out[label] = {"file": f.name, "L": (float(Lm.min()), float(Lm.mean()), float(Lm.max())), "tiles": wrap}
    return out


# ---------------------------------------------------------------- 3. grain

def grain(im):
    """Dominant orientation (structure tensor) and repeat period (spectrum peak along it), in the image's frame."""
    L = color.rgb2lab(im)[..., 0]
    gx, gy = ndi.gaussian_filter(L, 2, order=(0, 1)), ndi.gaussian_filter(L, 2, order=(1, 0))
    Jxx, Jyy, Jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)                  # direction of the gradient (normal to the stripes)
    coh = float(np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy))
    n = L.shape[0]
    w = np.hanning(n)
    F = np.abs(np.fft.fftshift(np.fft.fft2((L - L.mean()) * w[:, None] * w[None])))
    cy = cx = n // 2
    # power along the normal direction, radius 4..n/4
    rs = np.arange(4, n // 4)
    p = ndi.map_coordinates(F, [cy + rs * np.sin(theta), cx + rs * np.cos(theta)], order=1)
    p += ndi.map_coordinates(F, [cy - rs * np.sin(theta), cx - rs * np.cos(theta)], order=1)
    r = rs[int(np.argmax(p))]
    return {"theta": float(np.degrees(theta)), "css": float((90 + np.degrees(theta)) % 360), "coherence": coh,
            "period": float(n / r)}


# ---------------------------------------------------------------- page

def main():
    c1, im1 = load("f10-c100")
    med, seq = edge_sequence(im1)
    glass = ground_glass(c1, im1, "f10-c100")
    g = grain(im1)
    c2, im2 = load("f10-c60")
    for c in (c1, c2):
        Image.fromarray((load(c["short"])[1] * 255).astype(np.uint8)).resize((448, 448), Image.LANCZOS).save(OUT / f"{c['short']}.jpg", quality=88)
    print("edge profile", med)
    print("edge sequence", seq)
    print("grain", g)
    print("glass", {k: v["L"] for k, v in glass.items()})

    # edge sequence as stacked outlines round display type: fill = light end, shadows outward through the sequence
    light_to_dark = seq[::-1]
    fill, rings = light_to_dark[0], light_to_dark[1:5]
    radii = [0.018, 0.04, 0.066, 0.096]
    sh = []
    for col, r in zip(rings, radii):
        for a in range(0, 360, 30):
            sh.append(f"{r * np.cos(np.radians(a)):.3f}em {r * np.sin(np.radians(a)):.3f}em 0 {col}")
    text_shadow = ",".join(sh)
    inset = ",".join(f"inset 0 0 0 {4 * (i + 1)}px {col}" for i, col in enumerate(seq[:5]))
    fringe3 = f"-.03em 0 {light_to_dark[1]}, .03em 0 {light_to_dark[-1]}, 0 0 .4em {fill}88"
    profile_grad = "linear-gradient(90deg," + ",".join(med) + ")"
    seq_grad = "linear-gradient(90deg," + ",".join(f"{c} {100 * i / len(seq):.1f}% {100 * (i + 1) / len(seq):.1f}%" for i, c in enumerate(seq)) + ")"

    # grain: the ramp at the feature's angle, a rule at its period, a section cut along it
    ang = g["css"]
    per = g["period"] / 2            # the feature is shown at 448 px, half the 896 px it was measured at
    ramp_at = c1["ramp"].replace("90deg", f"{ang:.1f}deg")
    stripes = f"repeating-linear-gradient({ang:.1f}deg,{c1['accent']} 0 {per * 0.18:.1f}px,transparent {per * 0.18:.1f}px {per:.1f}px)"
    # the section cut and its title follow the stripes: a horizontal-ish grain tilts the top and bottom edges and
    # rotates the title; a vertical-ish grain slants the side edges and leans the letters (skew), since text can't run up
    t = np.radians(g["theta"] + 90)  # stripe direction in the image frame (x right, y down)
    dx, dy = np.cos(t), np.sin(t)
    if abs(dx) >= abs(dy):
        drop = float(np.clip(40 * dy / dx, -40, 40))                    # % of height over a section 2.5× wider than tall
        poly = f"polygon(0 {max(0, -drop):.1f}%, 100% {max(0, drop):.1f}%, 100% {100 - max(0, -drop):.1f}%, 0 {100 - max(0, drop):.1f}%)"
        title_tf = f"rotate({np.degrees(np.arctan2(dy, dx)):.1f}deg)"
        how = "its top and bottom edges run with the stripes, and the title is turned to run with them"
    else:
        s = float(np.clip(100 * 0.4 * dx / dy / 2, -20, 20))            # % of width the side edges slant, for a 2.5:1 section
        # s > 0: the stripes lean "\" (top left), so each side edge's bottom sits s% to the right of its top
        poly = f"polygon(0 0, {100 - s:.1f}% 0, 100% 100%, {s:.1f}% 100%)" if s >= 0 else f"polygon({-s:.1f}% 0, 100% 0, {100 + s:.1f}% 100%, 0 100%)"
        title_tf = f"skewX({g['theta']:.1f}deg)"                          # CSS skewX(+a) leans tops to the right
        how = "its side edges slant with the stripes, and the letters lean with them"

    # iridescence: the ramp as a conic border, closed on itself, with a sheen
    ramp2 = c2["gradient"]
    conic = "conic-gradient(from 200deg," + ",".join(ramp2 + [ramp2[0]]) + ")"
    tube = "linear-gradient(180deg," + ",".join(ramp2[:4]) + f",{c2['highlight']} 52%," + ",".join(ramp2[4:]) + ")"

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Lab: four treatments</title>
<link rel="stylesheet" href="../neuron-library/neuron.css?v=lab">
<style>
body{{margin:0;font:17px/1.55 var(--n-text);background:#e7e4dd;color:#1b1a18}}
.wrap{{max-width:1240px;margin:0 auto;padding:30px 28px 80px}}
h1{{font:400 34px var(--n-text);margin:0 0 4px}} h2{{font:400 26px var(--n-text);margin:54px 0 6px}}
.lede{{max-width:760px;color:#3d3a35;margin:0 0 18px}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:stretch}}
.cell{{padding:28px 30px;min-height:260px;position:relative}}
.cap{{font:500 13px var(--n-label);text-transform:lowercase;letter-spacing:.08em;opacity:.75;margin-top:14px}}
.sw{{display:flex;height:34px;margin:10px 0}} .sw i{{flex:1}}
img.feat{{width:100%;display:block}}
code{{font:13px var(--n-mono);background:#0000000d;padding:1px 4px}}
/* 1 edge sequence */
.edge-type{{font:normal 150px/1 var(--n-display);color:{fill};text-shadow:{text_shadow};margin:10px 0}}
.edge-fringe{{font:normal 96px/1 var(--n-display);color:{fill};text-shadow:{fringe3}}}
.edge-card{{box-shadow:{inset};padding:40px}}
/* 2 ground glass */
.glass{{background:url(f10-c100-glass-coarse.jpg) center/448px;color:{c1['on_dark'] if glass['coarse']['L'][1] < 55 else c1['ink']}}}
.glass.fine{{background-image:url(f10-c100-glass-fine.jpg)}}
.glass h3{{font:normal 64px/1 var(--n-display);margin:0 0 12px}}
.glass p{{max-width:420px;margin:0}}
.glass .panel{{background:color-mix(in srgb,{c1['paper']} 82%,transparent);color:{c1['ink']};padding:22px 24px;max-width:460px;margin-top:20px;backdrop-filter:blur(2px)}}
/* 3 grain */
.grain-ramp{{background:{ramp_at};min-height:120px}}
.grain-stripes{{background:{stripes};height:60px}}
.grain-cut{{background:{c1['dark']};color:{c1['on_dark']};clip-path:{poly};padding:60px 80px;aspect-ratio:2.5}}
.grain-cut h3{{font:normal 56px/1 var(--n-display);margin:0;transform:{title_tf};transform-origin:left bottom;display:inline-block;color:{c1['dark_accent']}}}
.grain-line{{position:absolute;left:50%;top:50%;width:70%;height:3px;background:#fff;mix-blend-mode:difference;transform:translate(-50%,-50%) rotate({g['theta'] + 90:.1f}deg)}}
/* 4 iridescence */
.iris{{border:7px solid transparent;border-radius:18px;background:linear-gradient({c2['paper']},{c2['paper']}) padding-box,{conic} border-box;padding:26px 28px;position:relative;color:{c2['ink']}}}
.iris::after{{content:'';position:absolute;inset:-7px;border-radius:18px;pointer-events:none;background:linear-gradient(165deg,#ffffffb0 0,#ffffff00 28%,#ffffff00 70%,#ffffff50 100%);
  -webkit-mask:linear-gradient(#000,#000) content-box,linear-gradient(#000,#000);-webkit-mask-composite:xor;mask:linear-gradient(#000,#000) content-box,linear-gradient(#000,#000);mask-composite:exclude;padding:7px}}
.iris h3{{font:normal 44px/1 var(--n-display);margin:0 0 8px;color:{c2['accent']}}}
.pill{{display:inline-block;border:4px solid transparent;border-radius:99px;background:linear-gradient({c2['dark']},{c2['dark']}) padding-box,{conic} border-box;color:{c2['on_dark']};padding:12px 26px;font:500 16px var(--n-label);letter-spacing:.12em;text-transform:lowercase;box-shadow:inset 0 1px 0 #fff6,0 6px 18px -8px {c2['shadow']}}}
.bead{{width:72px;height:72px;border-radius:50%;background:radial-gradient(circle at 32% 28%,#fff 0 6%,#fff0 20%),{conic};display:inline-block;vertical-align:middle;margin-right:14px;box-shadow:0 0 0 2px {c2['dark']} inset}}
.tube{{height:22px;border-radius:11px;background:{tube};margin:22px 0;box-shadow:0 3px 8px -4px {c2['shadow']}}}
</style></head><body><div class="wrap">
<h1>Lab: four treatments from the features</h1>
<p class="lede">Each measured from one feature. Edge sequence, ground glass and grain from <b>{c1['short']}</b> (ConvNeXt {c1['layer']}, channel {c1['short'].split('-c')[1]}); iridescence from <b>{c2['short']}</b>'s ramp.</p>

<h2>1 · Edge sequence</h2>
<p class="lede">Colours met crossing the feature's strong edges, dark side to light side. Top strip: the median colour at each distance (a profile). Bottom strip: the most common colour at each distance, as a sequence of {len(seq)} real colours. Then the sequence stacked as outlines round display type, as a three-colour fringe, and as inset bands on a card.</p>
<div class="row">
  <div><img class="feat" src="f10-c100.jpg" alt=""><div class="cap">{c1['short']} · the feature</div></div>
  <div><div class="sw" style="background:{profile_grad}"></div><div class="cap">profile: median colour −16 px … +16 px from an edge</div>
       <div class="sw" style="background:{seq_grad}"></div><div class="cap">sequence: {" · ".join(seq)}</div>
       <div class="cell edge-card" style="background:{c1['paper']};color:{c1['ink']};min-height:0;margin-top:14px"><b>Inset bands</b><br>The sequence as the border of a card, innermost first.</div></div>
</div>
<div class="cell n-dark" style="background:{c1['dark']};margin-top:22px;text-align:center">
  <div class="edge-type">Lantern</div>
  <div class="cap" style="color:{c1['on_dark']}">type filled with the light end, outlined outward through the sequence</div>
  <div class="edge-fringe" style="margin-top:30px">Opponent</div>
  <div class="cap" style="color:{c1['on_dark']}">three-colour fringe: next-to-light on the left, dark end on the right, light halo</div>
</div>

<h2>2 · Ground glass</h2>
<p class="lede">The feature low-passed until only its large colour layout remains, {"from its seamless tile, so it still tiles" if glass['coarse']['tiles'] else "from the upscale"}. Fine (σ 14 px of 896) keeps the cells; coarse (σ 36) keeps only the drift of colour. Text set straight on it, and on a translucent paper panel.</p>
<div class="row">
  <div class="cell glass fine"><h3 class="n-display">Fine</h3><p>Lightness range {glass['fine']['L'][0]:.0f}–{glass['fine']['L'][2]:.0f} (mean {glass['fine']['L'][1]:.0f}). Text in the colorway's dark-side ink, nothing between it and the feature.</p>
    <div class="panel">On a panel of the colorway's paper at 82%, the ground shows through as a tint.</div></div>
  <div class="cell glass"><h3 class="n-display">Coarse</h3><p>Lightness range {glass['coarse']['L'][0]:.0f}–{glass['coarse']['L'][2]:.0f} (mean {glass['coarse']['L'][1]:.0f}). The same feature with only its colour layout left: where it is yellow and where it is purple.</p>
    <div class="panel">The panel again, over the coarse ground.</div></div>
</div>

<h2>3 · Grain</h2>
<p class="lede">The feature's dominant orientation is {g['theta']:.1f}° (gradient normal, image frame; coherence {g['coherence']:.2f} of 1) and its repeat period {g['period']:.0f} px of 896. Below: the feature with its grain marked; the ramp run at that angle; stripes at that period; a section cut along the grain with its title set on it.</p>
<div class="row">
  <div style="position:relative"><img class="feat" src="f10-c100.jpg" alt=""><div class="grain-line"></div><div class="cap">the grain, as measured</div></div>
  <div><div class="grain-ramp"></div><div class="cap">the colorway ramp at {ang:.0f}° (CSS)</div>
       <div class="grain-stripes" style="margin-top:16px"></div><div class="cap">accent stripes at the feature's period ({per:.0f} px at this scale)</div></div>
</div>
<div class="grain-cut" style="margin-top:22px"><h3 class="n-display">A section along the grain</h3><p style="max-width:520px">A section at the feature's grain: {how}.</p></div>

<h2>4 · Iridescence</h2>
<p class="lede">The colorway's ramp ({len(ramp2)} colours) closed into a conic gradient and used as a border, with a specular sheen across it; a pill, a bead and a tube in the same colours.</p>
<div class="row">
  <div><img class="feat" src="f10-c60.jpg" alt=""><div class="cap">{c2['short']} · the feature</div></div>
  <div><div class="iris"><h3 class="n-display">An iridescent card</h3><p style="margin:0">The border runs through the ramp once round the card and catches a highlight at the top left.</p>
       <div style="margin-top:18px"><span class="bead"></span><a class="pill" href="#">See the work</a></div></div>
       <div class="tube"></div><div class="cap">a tube: the ramp top to bottom with the highlight across its crown</div></div>
</div>
</div></body></html>"""
    (OUT / "index.html").write_text(html)
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    main()
