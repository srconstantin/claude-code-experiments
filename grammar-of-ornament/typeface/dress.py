"""
Neuron — serifs and terminals that are compound and strokelib shapes,
grafted where classical serifs and terminals go.

Each free end of a letter stroke gets an ornament, which is the actual
outline of one piece (silhouettes.py: a fork, sprig, branching line from
compounds/, or a branch, hook or arc stroke from strokelib/), smoothed only
at its own scale:

  serifs     ends of the straight strokes (stems, diagonals, arms)
             - an end on a guide line (baseline, x-height, cap height,
               ascender, descender) gets a serif lying along that line,
               centred on the stroke and flush with the line, as classical
               serifs do; diagonals too (A, V, W, X, Y feet and tops)
             - any other end (the tip of an arm of E, F, T, L, Z) gets a
               serif across the stroke, as on a classical arm
  terminals  free ends of curves (C, S, a, c, f, r, j, g, y, 2, 3 …) get the
             terminal piece, continuing the curve, where a classical letter
             has a ball or beak terminal

A font uses one serif piece and one terminal piece throughout. Ends where a
stroke grows out of another (joins) get nothing.
"""

from __future__ import annotations

import numpy as np

import glyphs as GL
import outline as OL
import silhouettes as SI

LINES = (0, GL.XH, GL.CAP, GL.ASC, GL.DESC)
ON_LINE = 45          # an end this close to a guide line sits on it


def _placed(ref: str, along: np.ndarray, into: np.ndarray, length: float, at: np.ndarray, flush: bool,
            flip: bool) -> list[np.ndarray]:
    """A piece's outline scaled so its extent along its spine is `length`, laid along `along`, its
    width toward `into`. flush: its outer edge sits on `at` (a serif sitting on its line); otherwise
    it is centred on `at` across, and starts at `at` along (a terminal)."""
    s = SI.silhouette(ref)
    C = [c.copy() for c in s["contours"]]
    if flip:
        for c in C:
            c[:, 1] *= -1
    P = np.vstack(C)
    (x0, y0), (x1, y1) = P.min(0), P.max(0)
    k = length / max(x1 - x0, 1e-6)
    out = []
    for c in C:
        x = (c[:, 0] - (x0 + x1) / 2) * k if flush else (c[:, 0] - x0) * k
        y = (c[:, 1] - y0) * k if flush else (c[:, 1] - (y0 + y1) / 2) * k
        out.append(at + np.outer(x, along) + np.outer(y, into))
    return out


def dress(ch: str, strokes: list, W: float, serif: str | None, term: str | None,
          span: float = 2.6, term_len: float = 1.5, flip: bool = False) -> None:
    S = [s for s in strokes if isinstance(s, OL.Stroke)]
    for s in S:
        s.prepare(W)
        s.teeth, s.grafts, s.graft_ends = [], [], set()
        s.taper0, s.taper1 = s.taper0_set, s.taper1_set
    if not serif and not term:
        return
    OL.joins(strokes, W)
    centre = np.vstack([s.c for s in S]).mean(0)
    for s in S:
        if s.closed:
            continue
        t = OL.tangents(s.c, False)
        for end, joined, mark in ((0, s.taper0, s.end0), (1, s.taper1, s.end1)):
            if joined or mark == "none":
                continue
            e = s.c[0] if end == 0 else s.c[-1]
            out = -t[0] if end == 0 else t[-1]                   # pointing out of the stroke
            line = min(LINES, key=lambda y: abs(e[1] - y))
            on_line = abs(e[1] - line) < ON_LINE and abs(out[1]) > 0.35
            if on_line and serif:                                # a classical serif, along its line
                along = np.array([1.0, 0.0])
                into = np.array([0.0, -np.sign(out[1])])         # from the line into the letter
                s.grafts.append(_placed(serif, along, into, span * W * s.w, np.array([e[0], line]), True, flip))
                s.graft_ends.add(end)
            elif s.kind == "crescent" and serif:                 # the tip of an arm: a serif across it
                along = np.array([-out[1], out[0]])
                s.grafts.append(_placed(serif, along, -out, span * W * s.w, e, True, flip))
                s.graft_ends.add(end)
            elif s.kind != "crescent" and term:
                nrm = np.array([-out[1], out[0]])
                into = nrm if nrm @ (centre - e) > 0 else -nrm   # curling toward the letter
                s.grafts.append(_placed(term, out, into, term_len * W, e - out * 0.35 * W, False, flip))
                s.graft_ends.add(end)


# ── fused serifs (fuse.py) ──────────────────────────────────────────────────

PROFILE_POOL = ["straight-band", "wavy-band", "curved-band", "fork", "straight-sprig", "curved-sprig",
                "straight-comb", "curved-comb", "straight-branching-line", "curved-branching-line"]


def _profiles() -> list:
    import pieces as PC
    if not hasattr(_profiles, "cache"):
        out = []
        for kind in PROFILE_POOL:
            for f in sorted((PC.ROOT / kind).glob("[0-9][0-9].svg"))[:16]:
                try:
                    s = SI.silhouette(f"{kind}/{int(f.stem)}")
                except Exception:
                    continue
                if s["aspect"] >= 4 and len(s["profile"]) > 20:
                    out.append(s["profile"])
        _profiles.cache = out
    return _profiles.cache


def role(e, out, kind) -> tuple[str, float | None]:
    """Where a free end sits, as a serif position: foot (baseline), head (cap height or x-height),
    ascender, descender, arm (the tip of a bar or arm, off the guide lines) or terminal (a curve's end);
    and the guide line it sits on, if any."""
    line = min(LINES, key=lambda y: abs(e[1] - y))
    if abs(e[1] - line) < ON_LINE and abs(out[1]) > 0.35:
        return {0: "foot", GL.XH: "head", GL.CAP: "head", GL.ASC: "ascender", GL.DESC: "descender"}[line], line
    return ("arm" if kind == "crescent" else "terminal"), None


def fused(ch: str, strokes: list, W: float, ref, mirror: bool = True, profiles: bool = True,
          term: bool = True) -> list[np.ndarray]:
    """The glyph's outline with real compound pieces fused in at its serif positions (fuse.py).
    ref: one piece for every position, or {role: (piece, size, branches)} (see role())."""
    import hashlib
    import fuse as FU
    S = [s for s in strokes if isinstance(s, OL.Stroke)]
    for s in S:
        s.prepare(W)
        s.teeth, s.grafts, s.graft_ends = [], [], set()
        s.taper0, s.taper1 = s.taper0_set, s.taper1_set
        s.prof = None
    OL.joins(strokes, W)
    if profiles:
        pool = _profiles()
        for i, s in enumerate(S):
            h = int(hashlib.md5(f"{ch}/{i}".encode()).hexdigest()[:8], 16)
            s.prof = pool[h % len(pool)] if not s.closed else None
    grafts = []
    if ref:
        for s in S:
            if s.closed:
                continue
            t = OL.tangents(s.c, False)
            for end, joined, mark in ((0, s.taper0, s.end0), (1, s.taper1, s.end1)):
                if joined or mark == "none":
                    continue
                e = s.c[0] if end == 0 else s.c[-1]
                out = -t[0] if end == 0 else t[-1]
                where, line = role(e, out, s.kind)
                if where == "terminal" and not term:
                    continue
                spec = (ref.get(where) or ref["foot"]) if isinstance(ref, dict) else (ref,)
                defaults = (None, 1.0, None, None, None, None, None, 0)   # piece, size, branches, body_max, behind, squash, whole, depth
                piece, size, nb, bmax, behind, sq, whole, depth = tuple(spec) + defaults[len(spec):]
                if depth:
                    line = None                                  # a dendritic crown rises free of the guide line
                if not piece:
                    continue
                s.graft_ends.add(end)
                w = OL.widths(s, W)
                we = float(w[0] if end == 0 else w[-1])
                grafts += FU.serif_piece(piece, e, out, we, line, mirror and where != "terminal", W,
                                         span=size, branches=nb, body_max=bmax, behind=behind,
                                         squash=sq, whole=whole, depth=depth)
    polys, rings, dots = [], [], []
    for s in strokes:
        if isinstance(s, OL.Dot):
            dots.append(OL.dot_polygon(s, W))
        elif s.closed:
            rings.append(OL.polygons(s, W))
        else:
            polys += OL.polygons(s, W)
    return FU.raster_glyph(polys, rings, grafts, dots, W)
