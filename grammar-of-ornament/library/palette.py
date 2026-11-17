"""A colorway's colours, drawn from the image itself.

  base    the biggest patch of colour in the image: the real pixel nearest that patch's centre
  five    five more colours that are in the picture. The pixels are clustered into patches; patches that are too
          small, too grey or extreme in lightness are set aside (so the five sit at a similar saturation), as are
          patches that would not read against the ground (WCAG contrast under MIN_CONTRAST, or too close in
          lightness). From the rest, five are chosen as far apart from each other in Lab as possible (max-min
          dispersion, seeded by the biggest patch). Each is the real pixel nearest its patch's centre. Ordered by hue.
  ground  what the type sits on: the base itself when five of the image's colours read on it; otherwise (a
          mid-lightness base in an image with nothing light or dark) a light or dark tint of the base in its own hue,
          whichever lets more of the image's colours qualify. The ground is the only colour that may not be in the picture.

The filters relax in steps if they leave fewer than five patches; if a chosen colour still fails against the ground
its lightness is moved until it passes (counted as `adjusted`; only happens for images with no colour in them).
"""
import numpy as np
from skimage import color

MIN_CONTRAST = 2.2          # the five against the ground, so type in any of them reads on it
MIN_DL = 18                 # and at least this far in lightness
TINT_L = {"light": 93, "dark": 15}


def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in np.clip(rgb, 0, 1))


def lab_hex(L):
    return hexc(color.lab2rgb(np.asarray(L, float)[None, None])[0, 0])


def luminance(rgb):
    c = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def contrast(lab_a, lab_b):
    """WCAG contrast ratio between two Lab colours (arrays broadcast)."""
    la = luminance(color.lab2rgb(np.atleast_2d(lab_a)[None])[0])
    lb = luminance(color.lab2rgb(np.atleast_2d(lab_b)[None])[0])
    hi, lo = np.maximum(la, lb), np.minimum(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def kmeans(X, k, it=25, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return C


def tint(base, which):
    return np.array([TINT_L[which], base[1] * 0.35, base[2] * 0.35])


def palette(rgb, k=18, n=5):
    """rgb float HxWx3 → dict: base, ground, tint, five (hex), base_lab, ground_lab, five_lab, shares, adjusted, note."""
    lab = color.rgb2lab(rgb).reshape(-1, 3)
    rng = np.random.default_rng(1)
    C = kmeans(lab[rng.choice(len(lab), min(9000, len(lab)), replace=False)], k)
    assign = ((lab[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    share = np.bincount(assign, minlength=k) / len(lab)
    reps = np.array([lab[assign == j][((lab[assign == j] - C[j]) ** 2).sum(-1).argmin()] if share[j] else C[j] for j in range(k)])
    chroma = np.hypot(reps[:, 1], reps[:, 2])
    pop = share >= 0.015
    bi = int(np.argmax(share))
    base = reps[bi]
    med_c = np.median(chroma[pop]) if pop.any() else 0
    steps = ((max(14, 0.55 * med_c), 14, 90, MIN_CONTRAST, MIN_DL),
             (max(14, 0.55 * med_c), 14, 90, 1.6, 10),
             (max(8, 0.35 * med_c), 8, 95, 1.6, 10),
             (max(8, 0.35 * med_c), 8, 95, 1.0, 0),
             (0, 0, 100, 1.0, 0))

    def select(ground):
        cr = contrast(reps, ground)
        dL = np.abs(reps[:, 0] - ground[0])
        for si, (c_min, l_lo, l_hi, cr_min, dl_min) in enumerate(steps):
            ok = pop & (chroma >= c_min) & (reps[:, 0] >= l_lo) & (reps[:, 0] <= l_hi) & (cr >= cr_min) & (dL >= dl_min)
            ok[bi] = False
            if ok.sum() >= n:
                break
        cand = np.nonzero(ok)[0]
        if not len(cand):
            cand = np.array([j for j in np.argsort(-share) if j != bi][:n])
        chosen = [cand[np.argmax(share[cand])]]
        while len(chosen) < min(n, len(cand)):
            d = np.array([min(np.linalg.norm(reps[j] - reps[c]) for c in chosen) for j in cand])
            d[[list(cand).index(c) for c in chosen]] = -1
            chosen.append(cand[int(np.argmax(d))])
        while len(chosen) < n:                                   # an image with fewer than five patches
            chosen.append(chosen[len(chosen) % len(cand)])
        return chosen, si, cand

    ground, which = base, ""
    chosen, si, cand = select(base)
    if si >= 2 or contrast(reps[chosen], base).min() < MIN_CONTRAST:
        best = (si, -len(cand), chosen, cand, base, "")
        for name in ("light", "dark"):
            g = tint(base, name)
            ch, s, cd = select(g)
            best = min(best, (s, -len(cd), ch, cd, g, name), key=lambda t: t[:2])
        si, _, chosen, cand, ground, which = best
    five = reps[chosen].copy()
    adjusted = 0
    for i in range(len(five)):
        step = 2.0 if ground[0] < 50 else -2.0
        while contrast(five[i], ground)[0] < 1.6 and 2 < five[i, 0] < 98:
            five[i, 0] += step
            adjusted += (five[i, 0] - reps[chosen[i], 0]) == step
    order = np.argsort(np.degrees(np.arctan2(five[:, 2], five[:, 1])) % 360)
    five, shares = five[order], [float(share[chosen[i]]) for i in order]
    cr5 = contrast(five, ground)
    note = (f"chroma {chroma[chosen].min():.0f}–{chroma[chosen].max():.0f} · L {five[:, 0].min():.0f}–{five[:, 0].max():.0f} · "
            f"{len(cand)} cand · contrast {cr5.min():.1f}–{cr5.max():.1f}" + (f" · {which} tint" if which else "")
            + (f" · {adjusted} adjusted" if adjusted else ""))
    return {"base": lab_hex(base), "ground": lab_hex(ground), "tint": which, "five": [lab_hex(c) for c in five],
            "base_lab": [round(float(v), 2) for v in base], "ground_lab": [round(float(v), 2) for v in ground],
            "five_lab": [[round(float(v), 2) for v in c] for c in five], "shares": shares, "adjusted": int(adjusted), "note": note}


# ---------------------------------------------------------------- names

# Lab hue bins, calibrated on reference colours (red 40°, orange 60°, yellow 103°, lime 128°, green 136°, cyan 196°,
# azure 285°, blue 306°, violet 312°, magenta 328°, rose 3°). Lab packs blue to magenta into a narrow arc.
HUES = ((22, "rose"), (50, "red"), (78, "orange"), (115, "yellow"), (131, "lime"), (170, "green"), (225, "teal"),
        (309, "blue"), (321, "violet"), (345, "purple"), (360, "rose"))


def colour_name(h):
    """A plain name for a hex colour, from its Lab lightness, chroma and hue: the hue family, with the familiar
    name for its dark or pale forms (brown, olive, navy, pink, cream …) and 'dark' / 'pale' otherwise."""
    L, a, b = color.rgb2lab(np.array([[[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]]]))[0, 0]
    c = float(np.hypot(a, b))
    hue = float(np.degrees(np.arctan2(b, a)) % 360)
    if c < 9:
        return "black" if L < 18 else "charcoal" if L < 38 else "grey" if L < 72 else "white" if L > 92 else "pale grey"
    name = next(n for lim, n in HUES if hue < lim)
    dark, pale = L < 38, L > 76
    deep = L < 26 or (dark and c < 45)             # dark enough for the dark-form names (navy, plum …)
    if name in ("red", "rose") and pale:
        return "pink"
    if name in ("red", "rose") and dark:
        return "maroon"
    if name == "rose" and c < 30:
        return "dusty rose"
    if name == "red":
        if L < 50 and c < 45:
            return "brown"
        if c < 30:
            return "dusty rose"
    if name == "orange":
        if L < 50:
            return "brown"
        if pale and c < 40:
            return "tan"
    if name in ("yellow", "lime") and L < 62:
        return "olive"
    if name == "yellow" and pale and c < 35:
        return "cream"
    if name == "blue" and deep:
        return "navy"
    if name == "violet" and deep:
        return "indigo"
    if name == "purple" and deep:
        return "plum"
    if name == "purple" and pale:
        return "pink"
    if name == "blue" and pale:
        return "sky"
    return ("dark " if dark else "pale " if pale else "") + name
