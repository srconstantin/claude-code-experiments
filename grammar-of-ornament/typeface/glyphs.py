"""
Neuron — the letter skeletons.

Coordinates are font units on a 1000 em: baseline 0, x-height 500, cap
height 700, ascender 740, descender −220.

The skeleton is constructed, not drawn by hand:

  stems      crescents (outline.C): every straight stroke bows the same way,
             stems west and bars up, so a word repeats one asymmetric shape
  curves     superellipse arcs (exponent SQ, between an ellipse and a
             rectangle), so bowls are taut, with the same curvature family in
             every letter
  parts      the lowercase arch, bowl and descender hook are each built once
             (arch, bowl_r, bowl_l, hook) and shared, so n, h, m, u, b, d, p,
             q, g, j share the same construction
  joins      corners are separate crescents meeting at hairline joints; arms
             and bowls thin where they leave a stem (outline.joins)

The ornament (serifs and terminals) is not here: dress.py grafts it onto the
free ends.
"""

from __future__ import annotations

import numpy as np

from outline import C, Dot, Stroke

CAP, XH, ASC, DESC = 700, 500, 740, -250
W_CAP, W_LOW = 122, 112
SQ = 2.5            # superellipse exponent of every curve


def sup(cx, cy, rx, ry, a0, a1, n: int | None = None) -> np.ndarray:
    """Points along a superellipse arc from angle a0 to a1 (degrees, counter-clockwise)."""
    n = n or max(16, int(abs(a1 - a0) / 3))
    a = np.radians(np.linspace(a0, a1, n))
    e = 2 / SQ
    c, s = np.cos(a), np.sin(a)
    return np.stack([cx + rx * np.sign(c) * np.abs(c) ** e, cy + ry * np.sign(s) * np.abs(s) ** e], 1)


def K(*parts, **kw) -> Stroke:
    """A curve through the given point runs, joined end to end."""
    return Stroke(np.vstack([np.atleast_2d(np.asarray(p, float)) for p in parts]), **kw)


def ring(cx, cy, rx, ry) -> Stroke:
    return Stroke(sup(cx, cy, rx, ry, 90, 450)[:-1], closed=True)


# shared lowercase parts
def arch(x0, x1, top=XH):
    """The shoulder of n, h, m: leaves the stem at x0, rises, comes down as the stem at x1."""
    cx, rx = (x0 + x1) / 2, (x1 - x0) / 2
    return K(sup(cx, top - 175, rx, 182, 160, 0), [(x1, 0)])


def bowl_r(x, top=XH, bottom=0):
    """A bowl to the right of a stem at x (b, p)."""
    h = (top - bottom) / 2
    return K(sup(x + 182, bottom + h, 192, h + 6, 148, -148))


def bowl_l(x, top=XH, bottom=0):
    """A bowl to the left of a stem at x (d, q, a, g)."""
    h = (top - bottom) / 2
    return K(sup(x - 182, bottom + h, 192, h + 6, 32, 328))


def hook(x, y, r=170, a1=-150):
    """A descender (g, j): straight down from (x, y) to the descender zone, then curling left."""
    return K([(x, y)], sup(x - r, -70, r, -(DESC + 70) + 6, 0, a1))     # its bottom just past the descender line


G: dict[str, list] = {}

# ── uppercase ───────────────────────────────────────────────────────────────

G["A"] = [C((40, 0), (300, 712)), C((300, 712), (560, 0)), C((140, 240), (460, 240), w=0.8)]
G["B"] = [C((92, 0), (92, 700)),
          K([(92, 700), (285, 700)], sup(285, 538, 152, 162, 90, -90), [(285, 376), (92, 376)]),
          K([(92, 376), (305, 376)], sup(305, 188, 182, 188, 90, -90), [(305, 0), (92, 0)])]
G["C"] = [K(sup(312, 350, 258, 358, 38, 322))]
G["D"] = [C((92, 0), (92, 700)),
          K([(92, 700), (250, 700)], sup(250, 350, 312, 350, 90, -90), [(250, 0), (92, 0)])]
G["E"] = [C((92, 0), (92, 700)), C((92, 700), (460, 700), w=0.8), C((92, 362), (400, 362), w=0.8),
          C((92, 0), (480, 0), w=0.8)]
G["F"] = [C((92, 0), (92, 700)), C((92, 700), (470, 700), w=0.8), C((92, 355), (400, 355), w=0.8)]
G["G"] = [K(sup(312, 350, 258, 358, 38, 352)), C((360, 300), (568, 300), w=0.8)]
G["H"] = [C((92, 0), (92, 700)), C((508, 0), (508, 700)), C((92, 362), (508, 362), w=0.8)]
G["I"] = [C((120, 0), (120, 700))]
G["J"] = [K([(430, 700), (430, 250)], sup(275, 250, 155, 262, 0, -165))]
G["K"] = [C((92, 0), (92, 700)), C((92, 270), (500, 700), w=0.8), C((255, 440), (548, 0))]
G["L"] = [C((92, 0), (92, 700)), C((92, 0), (470, 0), w=0.8)]
G["M"] = [C((70, 0), (110, 700), w=0.8), C((110, 700), (330, 210)), C((330, 210), (550, 700), w=0.8),
          C((550, 700), (590, 0))]
G["N"] = [C((92, 0), (92, 700), w=0.8), C((92, 700), (508, 0)), C((508, 0), (508, 700), w=0.8)]
G["O"] = [ring(312, 350, 262, 360)]
G["P"] = [C((92, 0), (92, 700)),
          K([(92, 700), (300, 700)], sup(300, 520, 175, 180, 90, -90), [(300, 340), (92, 340)])]
G["Q"] = [ring(312, 350, 262, 360), C((340, 110), (640, -130))]
G["R"] = [C((92, 0), (92, 700)),
          K([(92, 700), (300, 700)], sup(300, 527, 172, 173, 90, -90), [(300, 354), (92, 354)]),
          C((280, 354), (552, 0))]
G["S"] = [K(sup(300, 530, 222, 176, 28, 270), sup(300, 178, 240, 178, 90, -152))]
G["T"] = [C((40, 700), (560, 700), w=0.8), C((300, 0), (300, 700))]
G["U"] = [K([(92, 700), (92, 262)], sup(305, 262, 213, 274, 180, 360), [(518, 262), (518, 700)])]
G["V"] = [C((40, 700), (300, 0)), C((300, 0), (560, 700), w=0.8)]
G["W"] = [C((25, 700), (190, 0)), C((190, 0), (310, 520), w=0.8), C((310, 520), (430, 0)), C((430, 0), (595, 700), w=0.8)]
G["X"] = [C((55, 0), (550, 700), w=0.8), C((55, 700), (550, 0))]
G["Y"] = [C((40, 700), (300, 340)), C((300, 340), (565, 700), w=0.8), C((300, 0), (300, 340))]
G["Z"] = [C((65, 700), (530, 700), w=0.8), C((65, 0), (530, 700)), C((65, 0), (545, 0), w=0.8)]

# ── lowercase ───────────────────────────────────────────────────────────────

G["a"] = [C((410, 0), (410, 335)), K(sup(262, 335, 148, 168, 0, 156)),
          K(sup(250, 150, 160, 150, 64, 344), end0="none")]
G["b"] = [C((88, 0), (88, 740)), bowl_r(88)]
G["c"] = [K(sup(258, 250, 192, 258, 42, 318))]
G["d"] = [C((428, 0), (428, 740)), bowl_l(428)]
G["e"] = [C((62, 262), (452, 262), w=0.78), K(sup(256, 250, 196, 258, 0, 320))]
G["f"] = [C((165, 0), (165, 560)), K(sup(312, 560, 147, 180, 180, 35)), C((48, 480), (312, 480), w=0.78)]
G["g"] = [ring(238, 318, 160, 190), hook(400, 505)]
G["h"] = [C((88, 0), (88, 740)), arch(88, 440)]
G["i"] = [C((115, 0), (115, 500)), Dot(115, 668)]
G["j"] = [hook(280, 500, r=118, a1=-160), Dot(280, 668)]
G["k"] = [C((88, 0), (88, 740)), C((88, 205), (405, 500), w=0.8), C((205, 315), (435, 0))]
G["l"] = [C((125, 0), (125, 740))]
G["m"] = [C((82, 0), (82, 500)), arch(82, 372), arch(372, 662)]
G["n"] = [C((88, 0), (88, 500)), arch(88, 440)]
G["o"] = [ring(255, 250, 196, 258)]
G["p"] = [C((88, DESC), (88, 500)), bowl_r(88)]
G["q"] = [C((428, DESC), (428, 500)), bowl_l(428)]
G["r"] = [C((88, 0), (88, 500)), K(sup(262, 330, 174, 172, 162, 62))]
G["s"] = [K(sup(242, 380, 158, 124, 26, 270), sup(242, 128, 176, 128, 90, -154))]
G["t"] = [K([(150, 650), (150, 125)], sup(252, 125, 102, 125, 180, 300)), C((40, 480), (305, 480), w=0.78)]
G["u"] = [K([(88, 500), (88, 175)], sup(264, 175, 176, 182, 180, 345)), C((440, 0), (440, 500))]
G["v"] = [C((28, 500), (232, 0)), C((232, 0), (436, 500), w=0.8)]
G["w"] = [C((20, 500), (160, 0)), C((160, 0), (255, 370), w=0.8), C((255, 370), (350, 0)), C((350, 0), (490, 500), w=0.8)]
G["x"] = [C((40, 0), (420, 500), w=0.8), C((40, 500), (420, 0))]
G["y"] = [C((30, 500), (245, 30)), C((455, 500), (150, DESC), w=0.9)]
G["z"] = [C((55, 500), (405, 500), w=0.78), C((55, 0), (405, 500)), C((55, 0), (420, 0), w=0.78)]

# ── figures (lining, cap height) ────────────────────────────────────────────

G["0"] = [ring(255, 350, 205, 362)]
G["1"] = [C((262, 0), (262, 712)), C((110, 560), (262, 712), w=0.8)]
G["2"] = [K(sup(255, 505, 190, 198, 158, -32), [(70, 0)]), C((70, 0), (470, 0), w=0.8)]
G["3"] = [K(sup(250, 535, 175, 168, 152, -90)), K(sup(255, 190, 196, 190, 90, -152))]
G["4"] = [C((390, 0), (390, 712)), C((50, 220), (390, 712), w=0.8), C((50, 220), (490, 220), w=0.8)]
G["5"] = [C((130, 700), (440, 700), w=0.8), C((112, 405), (130, 700), w=0.8),
          K(sup(262, 222, 198, 222, 130, -152))]
G["6"] = [ring(258, 218, 196, 222), K(sup(345, 340, 285, 360, 180, 68))]
G["7"] = [C((60, 700), (470, 700), w=0.8), C((200, 0), (470, 700))]
G["8"] = [ring(255, 535, 165, 165), ring(255, 190, 196, 192)]
G["9"] = [ring(242, 482, 196, 222), K(sup(155, 360, 285, 360, 0, -112))]

GLYPHS = G
