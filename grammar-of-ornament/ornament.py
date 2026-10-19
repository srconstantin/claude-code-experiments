#!/usr/bin/env python3
"""
Grammar of Ornament — ornament: vector ornaments generated from the measured
proportions of the strokes and compounds.

Nothing here is cut out of a feature. Each generator draws new marks whose
proportions come from the measurements in strokelib/ and compounds/ (tooth
spacing and angle of combs and feathers, hatch spacing and length, band
aspect, stroke tapers), so the results look like the features without copying
any of them, are seamless where they repeat, and are reproducible from a seed.

Building blocks
  stroke      a centerline (length, turn, waviness) drawn with a tapered
              width profile, as a filled outline
  Art         a set of filled shapes, positive (ink) or negative (cut out of
              the ink), with affine transforms and SVG output

Friezes (one seamless tile, repeated along a line; shapes that cross the tile
edge are duplicated a period away so the seam is invisible)
  comb_frieze       a spine with teeth on one side
  feather_frieze    a spine with barbs on both sides
  hatch_frieze      a row of short parallel tapered strokes
  wave_frieze       short tapered waves end to end
  band_frieze       a thick wavy band with gouges cut into its edges
  dash_frieze       hairline dashes and dots in glide reflection

Rosettes (a mark repeated round a centre, optionally mirrored)
  rosette           any element function, n-fold, with or without mirrors

Glyphs
  check, loop, caret, gouges, band_mark
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from linework import offset_rings


def f1(v: float) -> str:
    """Two decimals: ornament units are small (a tile is a few units high)."""
    return f"{v:.2f}".rstrip("0").rstrip(".")

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
#  Proportions measured from the feature set
# ─────────────────────────────────────────────────────────────────────────────

def measured(src: str = "vgg16features") -> dict:
    """Typical measurements (source px) of the compound types and stroke
    families, keyed by type or family label."""
    out = {}
    c = HERE / "compounds" / src / "compounds.json"
    if c.exists():
        for t in json.loads(c.read_text())["types"]:
            out[t["type"]] = t["typical"]
    s = HERE / "strokelib" / f"{src}-whole" / "families.json"
    if s.exists():
        for f in json.loads(s.read_text())["families"]:
            out.setdefault(f["label"], f["typical"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Strokes and shapes
# ─────────────────────────────────────────────────────────────────────────────

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def width_profile(n: int, w: float, start: float = 1.0, end: float = 1.0, ramp: float = 0.4) -> np.ndarray:
    """Width along a stroke: rises from start·w and falls to end·w (1 = blunt)."""
    t = np.linspace(0, 1, n)
    a = start + (1 - start) * smoothstep(t / ramp)
    b = end + (1 - end) * smoothstep((1 - t) / ramp)
    return w * np.minimum(a, b)


def centerline(length: float, angle: float = 0.0, turn: float = 0.0, wave: float = 0.0, waves: float = 1.0,
               phase: float = 0.0, start=(0.0, 0.0), n: int = 48, power: float = 1.0) -> np.ndarray:
    """Integrate a tangent angle that drifts by `turn` radians in total and
    oscillates with amplitude `wave` (radians), `waves` times along the length.
    With power > 1 the turning gathers toward the end, so the stroke curls into a spiral."""
    s = np.linspace(0, length, n)
    theta = angle + turn * ((s / length) ** power - (0.5 if power == 1.0 else 0.0)) + wave * np.sin(2 * np.pi * waves * s / length + phase)
    ds = length / (n - 1)
    x = start[0] + np.concatenate([[0.0], np.cumsum(np.cos(theta[:-1]) * ds)])
    y = start[1] + np.concatenate([[0.0], np.cumsum(np.sin(theta[:-1]) * ds)])
    return np.c_[x, y]


class Art:
    """Filled shapes. Each shape: rings (list of (N, 2) arrays), fill rule, and
    whether it is ink (+) or a cut (−)."""

    def __init__(self):
        self.shapes = []

    def stroke(self, pts: np.ndarray, w: np.ndarray, cut: bool = False, closed: bool = False):
        rings = offset_rings(pts[:, 0], pts[:, 1], np.asarray(w, float), closed)
        if rings:
            self.shapes.append((rings, "evenodd" if closed else "nonzero", cut))
        return self

    def poly(self, rings: list[np.ndarray], cut: bool = False, rule: str = "evenodd"):
        self.shapes.append(([np.asarray(r, float) for r in rings], rule, cut))
        return self

    def circle(self, cx: float, cy: float, r: float, cut: bool = False, n: int = 24):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return self.poly([np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]], cut)

    def extend(self, other: "Art"):
        self.shapes.extend(other.shapes)
        return self

    def transformed(self, M) -> "Art":
        """Apply a 2×3 affine matrix."""
        M = np.asarray(M, float)
        out = Art()
        for rings, rule, cut in self.shapes:
            out.shapes.append(([r @ M[:, :2].T + M[:, 2] for r in rings], rule, cut))
        return out

    def shifted(self, dx: float, dy: float) -> "Art":
        return self.transformed([[1, 0, dx], [0, 1, dy]])

    def bounds(self):
        allp = np.vstack([r for rings, _, cut in self.shapes if not cut for r in rings])
        return allp.min(0), allp.max(0)

    def svg(self, viewbox, ink: str = "#000", ground: str | None = None, width: float | None = None,
            height: float | None = None) -> str:
        x0, y0, W, H = viewbox
        size = f' width="{f1(width)}" height="{f1(height)}"' if width else ""

        def d(rings):
            return " ".join("M" + " L".join(f"{f1(x)},{f1(y)}" for x, y in r) + " Z" for r in rings)

        pos = "".join(f'<path fill-rule="{rule}" d="{d(r)}"/>' for r, rule, cut in self.shapes if not cut)
        neg = "".join(f'<path fill-rule="{rule}" d="{d(r)}"/>' for r, rule, cut in self.shapes if cut)
        bg = f'<rect x="{f1(x0)}" y="{f1(y0)}" width="{f1(W)}" height="{f1(H)}" fill="{ground}"/>' if ground else ""
        if neg:
            body = (f'<defs><mask id="m" maskUnits="userSpaceOnUse" x="{f1(x0 - W)}" y="{f1(y0 - H)}" width="{f1(3 * W)}" height="{f1(3 * H)}">'
                    f'<rect x="{f1(x0 - W)}" y="{f1(y0 - H)}" width="{f1(3 * W)}" height="{f1(3 * H)}" fill="#fff"/>'
                    f'<g fill="#000">{neg}</g></mask></defs><g fill="{ink}" mask="url(#m)">{pos}</g>')
        else:
            body = f'<g fill="{ink}">{pos}</g>'
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{f1(x0)} {f1(y0)} {f1(W)} {f1(H)}"{size} '
                f'preserveAspectRatio="none">{bg}{body}</svg>')


def data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def periodic(art: Art, period: float, axis: int = 0) -> Art:
    """Duplicate every shape one period either side, so a tile is seamless."""
    out = Art()
    for k in (-1, 0, 1):
        dx, dy = (k * period, 0) if axis == 0 else (0, k * period)
        out.extend(art.shifted(dx, dy))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Friezes
# ─────────────────────────────────────────────────────────────────────────────

def spine(art: Art, period: float, y: float, w: float, amp: float, waves: int, phase: float, cut: bool = False):
    """A constant-width wave across the tile and a little beyond, periodic in
    `period` (an integer number of waves), so it joins its neighbours exactly."""
    x = np.linspace(-0.25 * period, 1.25 * period, 160)
    yy = y + amp * np.sin(2 * np.pi * waves * x / period + phase)
    art.stroke(np.c_[x, yy], np.full(len(x), w), cut=cut)
    return lambda xq: (y + amp * np.sin(2 * np.pi * waves * xq / period + phase),
                       np.arctan(amp * 2 * np.pi * waves / period * np.cos(2 * np.pi * waves * xq / period + phase)))


def comb_frieze(seed: int = 0, n_teeth: int = 9, spacing: float = 3.4, spine_w: float = 1.7, tooth_len: float = 7.5,
                tooth_w: float = 1.25, angle_deg: float = 64.0, amp: float = 0.5, waves: int = 1,
                jitter: float = 0.2, len_jitter: float = 0.25, two_sided: bool = False, lean: float = 1.0,
                margin: float = 0.6, cut: bool = False) -> tuple[Art, tuple]:
    """Teeth hang below a wavy spine (and above it too if two_sided: a feather).
    Proportions default to the measured straight comb: spacing 3.4, spine 1.7,
    teeth at 64° to the spine. Returns the art and its viewBox."""
    rng = np.random.default_rng(seed)
    P = n_teeth * spacing
    top = (tooth_len * 1.25 if two_sided else 0) + spine_w / 2 + amp + margin
    art = Art()
    at = spine(art, P, top, spine_w, amp, waves, rng.uniform(0, 2 * np.pi), cut)
    teeth = Art()
    for i in range(n_teeth):
        for side in ((1, -1) if two_sided else (1,)):
            x0 = (i + 0.5 + (0.5 if side < 0 else 0)) * spacing + jitter * spacing * rng.uniform(-1, 1)
            y0, slope = at(x0)
            L = tooth_len * float(np.clip(1 + len_jitter * rng.normal(), 0.55, 1.5))
            a = slope + side * np.radians(angle_deg) * (1 if lean > 0 else -1)
            a = a if side > 0 else a  # barbs above lean the same way along the spine
            pts = centerline(L, angle=a, turn=rng.normal(0, 0.25), start=(x0, y0))
            teeth.stroke(pts, width_profile(len(pts), tooth_w, start=1.0, end=0.12, ramp=0.85), cut=cut)
    art.extend(periodic(teeth, P))
    H = top + spine_w / 2 + amp + tooth_len * 1.25 + margin
    return art, (0, 0, P, H)


def feather_frieze(seed: int = 0, **kw) -> tuple[Art, tuple]:
    """Barbs on both sides of the spine: the measured feather, two-sided, same spacing."""
    kw.setdefault("tooth_len", 5.0)
    kw.setdefault("angle_deg", 52.0)
    return comb_frieze(seed, two_sided=True, **kw)


def hatch_frieze(seed: int = 0, n: int = 12, spacing: float = 2.2, length: float = 4.6, w: float = 0.8,
                 angle_deg: float = 62.0, len_jitter: float = 0.18, curve: float = 0.15, margin: float = 1.0) -> tuple[Art, tuple]:
    """Short parallel tapered strokes, the measured fine hatching: spacing 2.2, length 4.6."""
    rng = np.random.default_rng(seed)
    P = n * spacing
    a = np.radians(angle_deg)
    h = length * np.sin(a)
    art = Art()
    for i in range(n):
        L = length * float(np.clip(1 + len_jitter * rng.normal(), 0.6, 1.4))
        x0 = (i + 0.5) * spacing - 0.5 * L * np.cos(a) + rng.uniform(-0.1, 0.1) * spacing
        y0 = margin + h - (h - L * np.sin(a)) / 2
        pts = centerline(L, angle=-a, turn=rng.normal(0, curve), start=(x0, y0))
        art.stroke(pts, width_profile(len(pts), w, start=0.35, end=0.2, ramp=0.5))
    return periodic(art, P), (0, 0, P, h + 2 * margin)


def wave_frieze(seed: int = 0, n: int = 4, length: float = 11.0, w: float = 1.2, waves: float = 1.5,
                wave: float = 0.35, overlap: float = 0.15, taper: float = 0.12, margin: float = 0.8) -> tuple[Art, tuple]:
    """Short tapered waves end to end, overlapping a little: the measured
    'short wave' family (length 11, about three inflections)."""
    rng = np.random.default_rng(seed)
    step = length * (1 - overlap)
    P = n * step
    amp_y = length * wave / (2 * np.pi * waves) * 1.6
    art = Art()
    for i in range(n):
        pts = centerline(length, angle=rng.normal(0, 0.05), wave=wave * float(rng.uniform(0.8, 1.2)), waves=waves,
                         phase=rng.uniform(-0.4, 0.4), start=(i * step, margin + amp_y + w + (0.45 * w if i % 2 else -0.45 * w)))
        art.stroke(pts, width_profile(len(pts), w, start=taper, end=taper, ramp=0.45))
    return periodic(art, P), (0, 0, P, 2 * (margin + amp_y + w))


def band_frieze(seed: int = 0, length: float = 25.0, width: float = 5.8, amp: float = 1.4, n_gouges: int = 9,
                gouge_len: float = 0.55, gouge_w: float = 0.95, margin: float = 0.5) -> tuple[Art, tuple]:
    """A thick wavy band (the measured band: width 5.8, aspect about 4.3 per
    wave) with relief gouges cut into both edges, as in a woodcut."""
    rng = np.random.default_rng(seed)
    P = length
    y = margin + amp + width / 2
    art = Art()
    at = spine(art, P, y, width, amp, 1, rng.uniform(0, 2 * np.pi))
    cuts = Art()
    for i in range(n_gouges):
        side = 1 if i % 2 else -1
        x0 = (i + 0.5) * P / n_gouges + rng.uniform(-0.15, 0.15) * P / n_gouges
        yc, slope = at(x0)
        y0 = yc + side * (width / 2 + 0.3)
        L = width * gouge_len * float(rng.uniform(0.7, 1.3))
        rad = np.radians(rng.uniform(25, 42))
        a = slope + (np.pi + side * rad if rng.uniform() < 0.5 else -side * rad)   # always pointing into the band
        pts = centerline(L, angle=a, turn=rng.normal(0, 0.3), start=(x0, y0))
        cuts.stroke(pts, width_profile(len(pts), gouge_w, start=1.0, end=0.1, ramp=0.9), cut=True)
    art.extend(periodic(cuts, P))
    return art, (0, 0, P, 2 * y)


def dash_frieze(seed: int = 0, period: float = 12.0, w: float = 0.55, dash: float = 4.5, dot: float = 0.75,
                tilt_deg: float = 10.0, margin: float = 0.6) -> tuple[Art, tuple]:
    """Hairline dashes and dots in glide reflection (frieze group p11g): each
    dash is mirrored across the midline and shifted half a period."""
    art = Art()
    y = margin + 2.0
    for k, sgn in ((0, 1), (1, -1)):
        x0 = k * period / 2 + period / 4 - dash / 2
        pts = centerline(dash, angle=sgn * np.radians(tilt_deg), turn=-sgn * 0.5, start=(x0, y - sgn * 0.6))
        art.stroke(pts, width_profile(len(pts), w, start=0.3, end=0.3))
        art.circle(k * period / 2 + period / 2 - 0.0, y + sgn * 0.9, dot / 2)
    return periodic(art, period), (0, 0, period, 2 * y)


# ─────────────────────────────────────────────────────────────────────────────
#  Rosettes and glyphs
# ─────────────────────────────────────────────────────────────────────────────

def rotation(theta: float, cx: float = 0.0, cy: float = 0.0):
    c, s = np.cos(theta), np.sin(theta)
    return [[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy]]


MIRROR_X = [[-1, 0, 0], [0, 1, 0]]


def rosette(element: Art, n: int, mirror: bool = False) -> Art:
    """n copies of an element (drawn pointing along +x from the origin) round
    the origin; with mirror, each copy also has its reflection (dihedral)."""
    out = Art()
    base = element
    if mirror:
        refl = element.transformed([[1, 0, 0], [0, -1, 0]])
        base = Art().extend(element).extend(refl)
    for k in range(n):
        out.extend(base.transformed(rotation(2 * np.pi * k / n)))
    return out


def petal(r0: float, length: float, w: float, turn: float = 0.6, taper=(0.2, 0.15)) -> Art:
    pts = centerline(length, angle=0.0, turn=turn, start=(r0, 0.0))
    return Art().stroke(pts, width_profile(len(pts), w, start=taper[0], end=taper[1], ramp=0.6))


def hook_element(r0: float, length: float, w: float, curl: float = 3.2) -> Art:
    """The measured 'short hook': a tapered stroke turning about 185°."""
    pts = centerline(length, angle=0.3, turn=curl, start=(r0, 0.0))
    return Art().stroke(pts, width_profile(len(pts), w, start=0.9, end=0.15, ramp=0.7))


def check(w: float = 1.4, seed: int = 0) -> Art:
    """A check mark drawn as one tapered stroke: short down-stroke, long up-stroke."""
    rng = np.random.default_rng(seed)
    a = np.array([[1.2, 5.4], [3.6, 8.2], [8.9, 1.4]]) + rng.normal(0, 0.08, (3, 2))
    t = np.linspace(0, 1, 40)
    # quadratic-ish blend through the corner, kept fairly sharp
    p = np.where(t[:, None] < 0.32, a[0] + (a[1] - a[0]) * (t[:, None] / 0.32),
                 a[1] + (a[2] - a[1]) * ((t[:, None] - 0.32) / 0.68))
    from scipy.ndimage import gaussian_filter1d
    p = gaussian_filter1d(p, 1.6, axis=0, mode="nearest")
    wp = width_profile(len(p), w, start=0.55, end=0.2, ramp=0.5)
    return Art().stroke(p, wp)


def loop(r: float = 4.0, w: float = 1.4, wobble: float = 0.06, seed: int = 0) -> Art:
    """A closed ring with a slightly uneven radius and width (the measured loop family)."""
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rr = r * (1 + wobble * np.sin(3 * a + rng.uniform(0, 6.28)) + 0.5 * wobble * np.sin(5 * a + rng.uniform(0, 6.28)))
    pts = np.c_[rr * np.cos(a), rr * np.sin(a)]
    ww = w * (1 + 0.25 * np.sin(a + rng.uniform(0, 6.28)))
    return Art().stroke(pts, ww, closed=True)


def caret(w: float = 1.2) -> Art:
    """A downward caret as one tapered hooked stroke."""
    a = np.array([[0.8, 2.0], [5.0, 6.2], [9.2, 2.0]])
    t = np.linspace(0, 1, 30)
    p = np.where(t[:, None] < 0.5, a[0] + (a[1] - a[0]) * (t[:, None] / 0.5), a[1] + (a[2] - a[1]) * ((t[:, None] - 0.5) / 0.5))
    from scipy.ndimage import gaussian_filter1d
    p = gaussian_filter1d(p, 1.2, axis=0, mode="nearest")
    return Art().stroke(p, width_profile(len(p), w, start=0.4, end=0.4, ramp=0.5))


def gouges(seed: int = 0, n: int = 4, height: float = 10.0, length: float = 4.0, w: float = 1.3, side: int = 1,
           cut: bool = False) -> Art:
    """A small fan of tapered gouge marks, as cut into a woodblock edge."""
    rng = np.random.default_rng(seed)
    art = Art()
    for i in range(n):
        y = (i + 0.5) * height / n + rng.uniform(-0.12, 0.12) * height / n
        L = length * float(rng.uniform(0.6, 1.15))
        a = (0 if side > 0 else np.pi) + rng.normal(0, 0.18)
        pts = centerline(L, angle=a, turn=rng.normal(0, 0.3), start=(0.0, y))
        art.stroke(pts, width_profile(len(pts), w, start=1.0, end=0.08, ramp=0.9), cut=cut)
    return art


def band_mark(seed: int = 0, length: float = 25.0, width: float = 5.8, waves: float = 1.0, amp: float = 0.35,
              vertical: bool = True) -> Art:
    """One band (the measured wavy band) as a free shape, tapered at both ends."""
    rng = np.random.default_rng(seed)
    pts = centerline(length, angle=np.pi / 2 if vertical else 0.0, wave=amp, waves=waves, phase=rng.uniform(0, 6.28))
    return Art().stroke(pts, width_profile(len(pts), width, start=0.45, end=0.35, ramp=0.35))


def fit(art: Art, pad: float = 0.5) -> tuple[Art, tuple]:
    lo, hi = art.bounds()
    return art, (lo[0] - pad, lo[1] - pad, hi[0] - lo[0] + 2 * pad, hi[1] - lo[1] + 2 * pad)


# ─────────────────────────────────────────────────────────────────────────────
#  More friezes and rosettes, and named recipes (for specs that must be data)
# ─────────────────────────────────────────────────────────────────────────────

def scroll_frieze(seed: int = 0, n: int = 4, step: float = 8.0, length: float = 16.0, w: float = 0.9,
                  curl: float = 7.5, power: float = 2.6, margin: float = 0.8) -> tuple[Art, tuple]:
    """A running scroll: tapered strokes that leave the baseline and curl into
    a spiral, alternately up and down (glide reflection). Each is the measured
    'short hook' (a stroke turning past 180°) with its turning gathered at the end."""
    rng = np.random.default_rng(seed)
    P = n * step
    art = Art()
    y = margin + length * 0.42
    for i in range(n):
        sgn = -1 if i % 2 == 0 else 1
        pts = centerline(length * float(rng.uniform(0.92, 1.08)), angle=0.0, turn=sgn * curl * float(rng.uniform(0.92, 1.08)),
                         power=power, start=(i * step, y))
        art.stroke(pts, width_profile(len(pts), w, start=0.9, end=0.18, ramp=0.9))
    base = np.linspace(-0.3 * P, 1.3 * P, 80)
    art.stroke(np.c_[base, np.full(len(base), y)], np.full(len(base), w * 0.45))
    return periodic(art, P), (0, 0, P, 2 * y)


def rosette_petal(n: int = 8, r0: float = 1.5, length: float = 4.5, w: float = 1.0, turn: float = 0.8,
                  taper0: float = 0.2, taper1: float = 0.15, mirror: bool = False) -> Art:
    return rosette(petal(r0, length, w, turn=turn, taper=(taper0, taper1)), n, mirror)


def rosette_hook(n: int = 6, r0: float = 2.0, length: float = 6.0, w: float = 0.9, curl: float = 3.2) -> Art:
    return rosette(hook_element(r0, length, w, curl), n)


def rosette_double(n: int = 8, r0: float = 1.2, length: float = 4.0, w: float = 0.7, turn: float = 0.9,
                   ring: bool = True) -> Art:
    """Two tiers of petals, the outer offset half a step, inside a hairline ring:
    the richer rosette for filigree."""
    inner = rosette(petal(r0, length * 0.6, w * 0.9, turn=-turn, taper=(0.3, 0.15)), n)
    outer = rosette(petal(r0 + length * 0.55, length, w, turn=turn, taper=(0.25, 0.12)), n).transformed(rotation(np.pi / n))
    art = Art().extend(inner).extend(outer)
    if ring:
        R = r0 + length * 1.55
        art.extend(loop(r=R, w=w * 0.5, wobble=0.0))
    art.circle(0, 0, r0 * 0.55)
    return art


RECIPES = {"rosette_petal": rosette_petal, "rosette_hook": rosette_hook, "rosette_double": rosette_double,
           "check": check, "loop": loop, "caret": caret, "gouges": gouges, "band_mark": band_mark,
           "comb_frieze": comb_frieze, "feather_frieze": feather_frieze, "hatch_frieze": hatch_frieze,
           "wave_frieze": wave_frieze, "band_frieze": band_frieze, "dash_frieze": dash_frieze, "scroll_frieze": scroll_frieze}


def make(recipe: str, kw: dict, seed: int | None = None, jitter: float = 0.0):
    """Call a recipe by name. With a seed, generators that take one get it, and
    with jitter every float parameter is scaled by up to ±jitter: a variant."""
    import inspect
    fn = RECIPES[recipe]
    kw = dict(kw)
    params = inspect.signature(fn).parameters
    rng = np.random.default_rng(seed)
    if seed is not None and "seed" in params:
        kw["seed"] = int(seed)
    if jitter:
        for k, v in list(kw.items()):
            if isinstance(v, float) and k not in ("taper0", "taper1", "margin"):
                kw[k] = v * float(rng.uniform(1 - jitter, 1 + jitter))
        if "n" in kw and recipe.startswith("rosette"):
            kw["n"] = int(max(3, kw["n"] + rng.integers(-1, 2)))
    return fn(**kw)
