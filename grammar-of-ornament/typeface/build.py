"""
Neuron — build the font and its proofs.

  .venv/bin/python typeface/build.py

Writes typeface/fonts/Neuron-Regular.otf and .woff2, proof images in
typeface/proofs/ (set with the compiled font itself, through FreeType), and
typeface/proofs/specimen.html.

Spacing: each glyph's sidebearings follow the shape of its sides. A side
that is upright over much of the letter's height (the stem of H, n) gets
SB_STRAIGHT; a round side (O, o) SB_ROUND; an open or diagonal side (V, T's
bar) SB_OPEN, interpolated by how much of the height lies within 25 units of
the side's extreme.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brush as BR       # noqa: E402
import curated as CU     # noqa: E402
import ornate as OR      # noqa: E402
import formal as FM      # noqa: E402
import styles as ST      # noqa: E402
import dress as DR       # noqa: E402
import glyphs as GL      # noqa: E402
import outline as OL     # noqa: E402

FONTS = HERE / "fonts"
PROOFS = HERE / "proofs"
NAME = "Neuron"
# variants: the stroke model's settings (outline.py) and weights (glyphs.py)
# and which compound pieces the letters carry (dress.py)
_BASE = dict(W_CAP=122, W_LOW=112, END_JOIN=0.25, END_FREE=0.55, CONTRAST=0.38, SB=(50, 34, 12))
_FEATHER = {"foot": ("fork/1", 1.0, None), "head": ("fork/1", 1.0, None),
            "ascender": ("straight-feather/0", 1.5, 3), "descender": ("straight-feather/0", 1.5, 3),
            "arm": ("straight-feather/0", 1.7, 3), "terminal": ("curved-feather/15", 1.3, 3)}
# role → (piece, size, branches kept, max stroke thickness, cut behind first branch, squash along the stroke,
#         whole-piece length, dendrite generations)
_DENDRITE = ("fork/1", 2.0, None, 0.35, None, None, None, 2)
_ANTLER = {"foot": ("fork/1", 1.0), "head": ("fork/1", 1.0),
           "ascender": _DENDRITE, "descender": _DENDRITE, "arm": _DENDRITE,
           "terminal": ("fork/1", 1.2, None, 0.5, None, 0.6)}
VARIANTS = {   # name → stroke settings + the compound pieces fused in at the serif positions (fuse.py, dress.role)
    # other letter constructions on the Bare skeleton (styles.py)
    "Split":   dict(_BASE, **ST.PARAMS["split"], fuse=None, profiles=False, style="split"),
    "Reverse": dict(_BASE, **ST.PARAMS["reverse"], fuse=None, profiles=False, style="reverse"),
    # Curated's skeletons drawn with the manners of formal lettering (formal.py)
    "Uncial":  dict(_BASE, fuse=None, profiles=False, curated=True, formal="uncial", ref_bare=True, kern=True,
                    ref_sb=(66, 60), min_sb=0.32, line=(800, -300)),
    # Curated's letters carrying real fragments of the compounds (ornate.py)
    "Ornate":  dict(_BASE, fuse=None, profiles=False, curated=True, ornate=True, ref_bare=True, kern=True,
                    ref_sb=(66, 60), min_sb=0.32, line=(1000, -320)),
    # letters drawn from hand-chosen compound pieces (curated.py)
    "Curated": dict(_BASE, fuse=None, profiles=False, curated=True, ref_bare=True, kern=True,
                    ref_sb=(66, 60), min_sb=0.32, line=(800, -300)),
}
SB_STRAIGHT, SB_ROUND, SB_OPEN = 48, 32, 10
SPACE = 250
OVERHANG = 0.6      # share of the teeth's overhang that the spacing makes room for


def side_fraction(contours, side: str) -> float:
    """How much of the glyph's height has ink within 25 units of its leftmost (rightmost) point."""
    pts = np.vstack(contours)
    ys = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 60)
    ext = pts[:, 0].min() if side == "l" else pts[:, 0].max()
    hit = 0
    for y in ys:
        xs = _crossings(contours, y)
        if not len(xs):
            continue
        x = xs.min() if side == "l" else xs.max()
        hit += abs(x - ext) < 25
    return hit / len(ys)


def _crossings(contours, y):
    xs = []
    for C in contours:
        a, b = C, np.roll(C, -1, 0)
        m = (a[:, 1] - y) * (b[:, 1] - y) < 0
        t = (y - a[m, 1]) / (b[m, 1] - a[m, 1])
        xs.extend(a[m, 0] + t * (b[m, 0] - a[m, 0]))
    return np.array(xs)


DEPTH = 0.22       # how far into a letter's side the white counts, as a share of its zone height
MIN_SB = 0.14      # no side closer than this (in weights) to the edge of its advance
REF_SB = (54, 48)  # (caps, lowercase) sidebearing of the straight stem of H and n: every other side matches its white


def depths(contours, side: str, lo: float, hi: float, n: int = 80) -> tuple[np.ndarray, float]:
    """For rows across the zone lo..hi: how far in from the zone's outermost ink (on that side) the ink
    starts; and that outermost x. Parts of the letter outside the zone (an f's hook, a descender) don't
    count, so they may overhang the next letter."""
    rows = []
    for y in np.linspace(lo, hi, n):
        xs = _crossings(contours, y)
        rows.append((xs.min() if side == "l" else xs.max()) if len(xs) else None)
    have = [x for x in rows if x is not None]
    ext = min(have) if side == "l" else max(have)
    d = np.array([np.inf if x is None else (x - ext if side == "l" else ext - x) for x in rows])
    return d, ext


INTRUDE = 1.1      # how far (in weights) ornament may reach into the space between letters


def depths_eff(body, full, side: str, zone: float, intrude: float, n: int = 90):
    """Like depths(), on the body, but where the full glyph (with ornament) sticks out more than
    `intrude` past it, the ornament counts (less `intrude`). Rows span the ornament's height too."""
    P = np.vstack(full)
    ys = np.linspace(min(0.0, P[:, 1].min()), max(zone, P[:, 1].max()), n)
    rows = []
    for y in ys:
        xb, xf = _crossings(body, y), _crossings(full, y)
        if side == "l":
            cand = [x for x in ((xb.min() if len(xb) else None), (xf.min() + intrude if len(xf) else None)) if x is not None]
            rows.append(min(cand) if cand else None)
        else:
            cand = [x for x in ((xb.max() if len(xb) else None), (xf.max() - intrude if len(xf) else None)) if x is not None]
            rows.append(max(cand) if cand else None)
    have = [x for x in rows if x is not None]
    ext = min(have) if side == "l" else max(have)
    in_zone = (ys >= 0) & (ys <= zone)
    d = np.array([np.inf if x is None else (x - ext if side == "l" else ext - x) for x in rows])
    # rows outside the zone only matter where ornament sticks out
    d = np.where(in_zone, d, np.where(d < intrude, d, np.inf))
    return d, ext


def area_sb(d: np.ndarray, target: float, dmax: float) -> float:
    """The sidebearing that makes the side's white (per row, capped at dmax) average `target`."""
    lo, hi = -dmax, dmax
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.mean(np.minimum(mid + d, dmax)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sidebearing(f: float, scale: float) -> float:
    if f > 0.45:
        return SB_STRAIGHT * scale
    if f > 0.12:
        return (SB_ROUND + (SB_STRAIGHT - SB_ROUND) * (f - 0.12) / 0.33) * scale
    return (SB_OPEN + (SB_ROUND - SB_OPEN) * f / 0.12) * scale


def configure(variant: str) -> None:
    global SB_STRAIGHT, SB_ROUND, SB_OPEN, STYLE
    v = VARIANTS[variant]
    STYLE = variant
    GL.W_CAP, GL.W_LOW = v["W_CAP"], v["W_LOW"]
    for k in ("END_JOIN", "END_FREE", "CONTRAST"):
        setattr(OL, k, v[k])
    for k, default in (("BULB", 1.15), ("PROF_AMT", 0.45)):
        setattr(OL, k, v.get(k, default))
    SB_STRAIGHT, SB_ROUND, SB_OPEN = v["SB"]


REFS: dict = {}
EVEN = 0.75        # how much of a glyph's weight discrepancy (vs. H, n, 0) is corrected (1 = all)


def main_stroke(contours) -> float:
    """The thickness of a glyph's main strokes: the 75th percentile of twice the distance to the edge
    along its medial axis (ignoring thin ends and hairlines)."""
    from PIL import ImageDraw as _D
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    P = np.vstack(contours)
    lo = P.min(0) - 5
    Wd, Hd = (np.ceil(P.max(0) - lo) + 10).astype(int)
    m = np.zeros((Hd, Wd), bool)
    for C in contours:
        im = Image.new("1", (Wd, Hd), 0)
        _D.Draw(im).polygon([(x - lo[0], Hd - (y - lo[1])) for x, y in C], fill=1)
        m ^= np.asarray(im, bool)
    dt = ndi.distance_transform_edt(m)
    v = 2 * dt[skeletonize(m)]
    v = v[v > 0.35 * np.percentile(v, 90)]
    return float(np.percentile(v, 75))


_WEIGHT: dict = {}


def weight(ch: str) -> float:
    """The glyph's weight, corrected toward its reference letter's (H for capitals, n for lowercase,
    0 for figures), measured on the bare strokes."""
    if VARIANTS[STYLE].get("curated"):
        return CU.weight(ch, main_stroke)
    if ch not in _WEIGHT:
        base = GL.W_LOW if ch.islower() else GL.W_CAP
        refch = "n" if ch.islower() else ("0" if ch.isdigit() else "H")
        def bare(c):
            DR.dress(c, GL.GLYPHS[c], base, None, None)
            return main_stroke(OL.outline(GL.GLYPHS[c], base))
        f = bare(refch) / bare(ch)
        _WEIGHT[ch] = base * float(np.clip(f, 0.8, 1.35)) ** EVEN
    return _WEIGHT[ch]


BR_DEFAULTS = {k: getattr(BR, k) for k in ("REACH", "LAYERS", "MIN_BRANCHES", "TRIES", "ENVELOPE_SIDE",
                                             "ENVELOPE_VERT", "SPUR", "ISOLATED", "TENDRIL", "TENDRIL_MAX", "TENDRILS")}


def contours_for(ch: str, W: float, v: dict) -> list:
    strokes = GL.GLYPHS[ch]
    DR.dress(ch, strokes, W, None, None)
    if v.get("formal"):
        return FM.glyph(ch, W, v["formal"])
    if v.get("ornate"):
        return OR.glyph(ch, W)
    if v.get("curated"):
        return CU.glyph(ch, W)
    if v.get("style"):
        return OL.outline(ST.styled(ch, v["style"], W), W)
    if v.get("brush"):
        BR.CLEAN, BR.CONTOUR_SMOOTH = v.get("clean", 0.0), v.get("contour", 0.0)
        BR.RESTROKE = v.get("restroke", False)
        for k, val in {**BR_DEFAULTS, **v.get("params", {})}.items():
            setattr(BR, k, val)
        fam = {"effects": BR.EFFECTS, "wavy-rich": BR.WAVY_RICH, "thicket": BR.THICKET}.get(v["brush"], v["brush"])
        return BR.glyph(ch, fam, W)
    if v["fuse"]:
        return DR.fused(ch, strokes, W, v["fuse"], profiles=v["profiles"])
    return OL.outline(strokes, W)


def _ref_target(lower: bool, dmax: float) -> float:
    """The white beside the straight stem of n (or H) at its reference sidebearing."""
    ch = "n" if lower else "H"
    W = GL.W_LOW if lower else GL.W_CAP
    v = VARIANTS[STYLE]
    c = OL.outline(ST._base(ch, W), W) if v.get("space_bare") or v.get("ref_bare") else contours_for(ch, W, v)
    d, _ = depths(c, "l", 0, GL.XH if lower else GL.CAP)
    return float(np.mean(np.minimum(v.get("ref_sb", REF_SB)[lower] + d, dmax)))


KERN_BAND = 50      # rows this close count as level with each other when measuring a pair's gap
KERN_CLEAR = 0.32   # no pair's ink comes closer than this (in the lowercase weight)
KERN_WHITE = 1.2    # a pair with more white than this × a pair of stems is drawn together …
KERN_PULL = 0.65    # … by this share of the excess
KERN_MAX = 110


def _profile(contours, side: str, ys):
    """Each row's outermost ink on one side (nan where the glyph has none)."""
    out = np.full(len(ys), np.nan)
    for k, y in enumerate(ys):
        xs = _crossings(contours, y)
        if len(xs):
            out[k] = max(xs) if side == "r" else min(xs)
    return out


def kern_feature(outlines: dict, adv: dict, names: dict) -> str:
    """Kerning measured from the outlines: pairs that would nearly touch are opened, pairs with a hole
    between them (T o, V a) are closed, the rest are left to the sidebearings."""
    ys = np.arange(-340, 1060, 20.0)
    R = {c: _profile(o, "r", ys) for c, o in outlines.items()}
    L = {c: _profile(o, "l", ys) for c, o in outlines.items()}
    band = int(KERN_BAND / 20)

    def spread(p):
        q = p.copy()
        for k in range(1, band + 1):
            q = np.fmax(np.fmax(q, np.roll(p, k)), np.roll(p, -k))
        return q
    Rs = {c: spread(p) for c, p in R.items()}
    zone = (ys >= 0) & (ys <= GL.XH)
    cap = 420 if VARIANTS[STYLE].get("ornate") else 240  # white beyond this doesn't count (Ornate is spaced wider)

    def pair_white(a, b):
        g = ((adv[a] - R[a]) + L[b])[zone]
        return float(np.nanmean(np.minimum(g, cap))) if np.isfinite(g).any() else None
    ref = pair_white("n", "n")
    if VARIANTS[STYLE].get("ornate"):                   # a crest can widen one letter: take the typical pair
        low = [c for c in outlines if c.islower()]
        ref = float(np.median([w for a in low for b in low if (w := pair_white(a, b)) is not None]))
    clear = KERN_CLEAR * GL.W_LOW
    lines = []
    for a in outlines:
        for b in outlines:
            g = (adv[a] - Rs[a]) + L[b]
            g = g[np.isfinite(g)]
            if not len(g):
                continue
            k = 0.0
            if g.min() < clear:
                k = clear - g.min()
            else:
                w = pair_white(a, b)
                if w is not None and w > KERN_WHITE * ref:
                    k = -min(KERN_PULL * (w - KERN_WHITE * ref), g.min() - clear)
            k = int(round(np.clip(k, -KERN_MAX, KERN_MAX) / 5) * 5)
            if abs(k) >= 10:
                lines.append(f"  pos {names[a]} {names[b]} {k};")
    print(f"  kerning: {len(lines)} pairs", flush=True)
    return "feature kern {\n" + "\n".join(lines) + "\n} kern;\n"


def build() -> Path:
    t0 = time.time()
    order = [".notdef", "space"] + list(GL.GLYPHS)
    names = {ch: _gname(ch) for ch in GL.GLYPHS}
    charstrings, metrics, outlines, outlines_placed = {}, {}, {}, {}
    _WEIGHT.clear()
    for ch, strokes in GL.GLYPHS.items():
        W = weight(ch)
        contours = contours_for(ch, W, VARIANTS[STYLE])
        outlines[ch] = contours
        # optical spacing: each side gets the same white as the straight stem of H (caps, figures) or n
        zone = GL.XH if ch.islower() else GL.CAP
        dmax = DEPTH * zone
        if (STYLE, ch.islower()) not in REFS:
            REFS[(STYLE, ch.islower())] = _ref_target(ch.islower(), dmax)
        ref = REFS[(STYLE, ch.islower())]
        pts = np.vstack(contours)
        if VARIANTS[STYLE].get("space_bare"):
            # spaced by the plain body, but ornament reaching more than INTRUDE weights past it pushes
            # the neighbour away (measured over the ornament's whole height, not just the zone)
            body = OL.outline(ST._base(ch, W), W)
            dl, xl = depths_eff(body, contours, "l", zone, INTRUDE * W)
            dr, xr = depths_eff(body, contours, "r", zone, INTRUDE * W)
        elif VARIANTS[STYLE].get("curated"):
            # a spur hangs into the space, as a serif would; Ornate's fragments take up room
            v_ = VARIANTS[STYLE]
            body = contours if v_.get("ornate") or v_.get("formal") else CU.body(ch, W)
            dl, xl = depths(body, "l", 0, zone)
            dr, xr = depths(body, "r", 0, zone)
        else:
            dl, xl = depths(contours, "l", 0, zone)
            dr, xr = depths(contours, "r", 0, zone)
        floor = VARIANTS[STYLE].get("min_sb", MIN_SB) * W
        lsb = max(area_sb(dl, ref, dmax), floor)            # measured from the zone's ink; serif tips never touch
        rsb = max(area_sb(dr, ref, dmax), floor)
        dx = lsb - xl                                       # the zone's left ink sits lsb from the origin
        adv = int(round(xr - xl + lsb + rsb))
        lsb = lsb - (xl - pts[:, 0].min())                  # the font's lsb is from the glyph's leftmost point
        outlines_placed[ch] = [C + [dx, 0] for C in contours]
        pen = T2CharStringPen(adv, None)
        for C in contours:
            segs = OL.fit_contour(C + [dx, 0])
            if not segs:
                continue
            pen.moveTo(tuple(np.round(segs[0][0], 1)))
            for b in segs:
                pen.curveTo(tuple(np.round(b[1], 1)), tuple(np.round(b[2], 1)), tuple(np.round(b[3], 1)))
            pen.closePath()
        charstrings[names[ch]] = pen.getCharString()
        metrics[names[ch]] = (adv, int(round(lsb)))
    for g, w in ((".notdef", 500), ("space", SPACE)):
        pen = T2CharStringPen(w, None)
        charstrings[g] = pen.getCharString()
        metrics[g] = (w, 0)
    order = [".notdef", "space"] + [names[c] for c in GL.GLYPHS]
    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder(order)
    cmap = {ord(c): names[c] for c in GL.GLYPHS}
    cmap[32] = "space"
    fb.setupCharacterMap(cmap)
    fb.setupCFF(f"{NAME}-{STYLE}", {"FullName": f"{NAME} {STYLE}"}, charstrings, {})
    fb.setupHorizontalMetrics(metrics)
    asc, desc = VARIANTS[STYLE].get("line", (1250, -300))
    fb.setupHorizontalHeader(ascent=asc, descent=desc)
    fb.setupNameTable({"familyName": f"{NAME} {STYLE}", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=760, sTypoDescender=-240, sTypoLineGap=200, usWinAscent=asc, usWinDescent=-desc,
                sxHeight=GL.XH, sCapHeight=GL.CAP)
    fb.setupPost()
    FONTS.mkdir(parents=True, exist_ok=True)
    otf = FONTS / f"{NAME}-{STYLE}.otf"
    if VARIANTS[STYLE].get("kern"):
        adv = {ch: metrics[names[ch]][0] for ch in GL.GLYPHS}
        fb.addOpenTypeFeatures(kern_feature(outlines_placed, adv, names))
    fb.save(str(otf))
    f = TTFont(str(otf))
    f.flavor = "woff2"
    f.save(str(FONTS / f"{NAME}-{STYLE}.woff2"))
    print(f"built {otf.name}: {len(GL.GLYPHS)} glyphs in {time.time() - t0:.1f}s", flush=True)
    return otf


def _gname(ch: str) -> str:
    if ch.isdigit():
        return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][int(ch)]
    return ch


SAMPLES = ["Neuron", "HAMBURGEFONSTIV", "hamburgefonstiv", "ABCDEFGHIJKLM", "NOPQRSTUVWXYZ",
           "abcdefghijklm", "nopqrstuvwxyz", "0123456789", "HOHOHO nnonnoo", "Quick Wax Zephyr Jockey",
           "The Grammar of Ornament", "Sphinx of black quartz, judge my vow"]


def proofs(otf: Path) -> None:
    out = PROOFS / STYLE.lower()
    out.mkdir(parents=True, exist_ok=True)
    big = ImageFont.truetype(str(otf), 220)
    for i, text in enumerate(SAMPLES):
        text = text.replace(",", "")                    # no punctuation yet
        l, t, r, b = big.getbbox(text)
        im = Image.new("L", (r - l + 80, int(big.size * 1.8)), 255)      # room for ornament above the ascender
        ImageDraw.Draw(im).text((40 - l, int(big.size * 0.1)), text, font=big, fill=0)
        im.save(out / f"line{i:02d}.png")
    # glyph sheet
    f = ImageFont.truetype(str(otf), 150)
    chars = list(GL.GLYPHS)
    cw, ch_ = 200, 260
    sheet = Image.new("RGB", (cw * 10, ch_ * ((len(chars) + 9) // 10)), "white")
    d = ImageDraw.Draw(sheet)
    asc = f.getmetrics()[0]
    for k, c in enumerate(chars):
        x, y = (k % 10) * cw, (k // 10) * ch_
        adv = f.getlength(c)
        x0 = x + (cw - adv) / 2
        base = y + 40 + asc
        for yy in (0, GL.XH, GL.CAP):
            Y = base - yy * 150 / 1000
            d.line([(x + 8, Y), (x + cw - 8, Y)], fill=(170, 200, 235))
        for xx in (x0, x0 + adv):
            d.line([(xx, y + 20), (xx, y + ch_ - 30)], fill=(235, 130, 110))
        d.text((x0, base), c, font=f, fill="black", anchor="ls")
    sheet.save(out / "glyphs.png")


def specimen() -> None:
    rows = "".join(f'<p class="s" style="font-size:{s}px">{t}</p>' for t, s in
                   (("Neuron", 220), ("HAMBURGEFONSTIV", 110), ("hamburgefonstiv", 110),
                    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 72), ("abcdefghijklmnopqrstuvwxyz", 72), ("0123456789", 72),
                    ("The Grammar of Ornament", 96), ("Quick Wax Zephyr Jockey", 64),
                    ("Sphinx of black quartz judge my vow", 40), ("Sphinx of black quartz judge my vow", 20)))
    faces = "".join(f'@font-face{{font-family:"Neuron {v}";src:url(../fonts/{NAME}-{v}.woff2) format("woff2")}}' for v in VARIANTS)
    secs = "".join(f'<section style="font-family:\'Neuron {v}\'"><h2>Neuron {v}</h2>{rows}</section>' for v in VARIANTS)
    (PROOFS / "specimen.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Neuron Specimen</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{faces}
body{{margin:0;padding:24px 16px;background:#fff;color:#111;font-family:'Helvetica Neue',sans-serif}}
h2{{font:500 13px 'Helvetica Neue',sans-serif;color:#888;margin:40px 0 8px}}
.s{{margin:0 0 .15em;line-height:1.15;overflow-wrap:anywhere}}
@media (prefers-color-scheme:dark){{body{{background:#111;color:#eee}}}}</style></head><body>{secs}
<p style="font-size:12px;color:#888">Uppercase, lowercase and figures; no punctuation yet.</p></body></html>""")


if __name__ == "__main__":
    for variant in (sys.argv[1].split(",") if len(sys.argv) > 1 else VARIANTS):
        configure(variant)
        proofs(build())
    specimen()
    print(f"wrote {PROOFS}", flush=True)
