#!/usr/bin/env python3
"""
Grammar of Ornament — compounds: structures made of several marks.

strokelib.py cuts every mark at its junctions and files the pieces. This
script puts the pieces back together into three kinds of compound, from the
per-stroke tables linework.py writes (<name>.strokes.json):

  trees     connected groups of strokes (ink or relief) joined at junctions.
            The spine is the longest path through the group; the rest are side
            branches, measured by how many there are per length of spine, how
            long they are relative to it, their angle to it, whether they hang
            off one side, how regularly they are spaced, and whether the group
            has cycles. Types: comb (many short, regular, one-sided teeth),
            feather (two-sided), branching line (a few long branches), sprig,
            tree (bushy), fork and burst (one junction), loop and mesh (cycles).
  bundles   groups of strokes that run parallel a short distance apart:
            hatching. Two bundles crossing at a steep angle in the same place
            are crosshatching. Measured by spacing and its regularity,
            orientation, stroke length and straightness.
  bands     masses (the thick black) with an elongated spine, measured like
            strokes at the thick scale: length, width, aspect, curvature,
            waviness, taper, arms; with the relief hatching that sits on them
            attached, giving plain, hatched and crosshatched bands.

Each compound is rendered whole as an SVG (masses filled, strokes as
variable-width outlines, relief marks white over their mass), and each type
gets a contact sheet showing its most typical and its largest examples.

Written into <dst>/:
  <type>/NN.svg        the representatives of each type
  <type>/sheet.png     contact sheet for the type
  compounds.png        one row per type
  compounds.json       every type: count, typical measurements, by layer, the
                       representatives with their measurements
  by_feature.csv       per feature, how many compounds of each type
  and a table of types by layer printed at the end

Run:  python3 compounds.py                                   # linework/vgg16features → compounds/vgg16features
      python3 compounds.py --per-type 24 --gap 3 --min-teeth 5
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from linework import offset_rings, rings_path, f1

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def signed_area(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def end_dir(pts: np.ndarray, end: int) -> np.ndarray:
    """Unit direction pointing from the given end into the stroke."""
    n = len(pts)
    m = min(4, n - 1)
    v = (pts[m] - pts[0]) if end == 0 else (pts[-1 - m] - pts[-1])
    return v / (np.linalg.norm(v) + 1e-9)


def circ_mean_deg(angles_deg: np.ndarray, weights=None) -> tuple[float, float]:
    """Mean orientation (mod 180) and spread (0 = all parallel, 1 = uniform)."""
    z = np.exp(2j * np.radians(angles_deg))
    m = np.average(z, weights=weights) if weights is not None else z.mean()
    return float(np.degrees(np.angle(m)) / 2 % 180), float(1 - abs(m))


def ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


# ─────────────────────────────────────────────────────────────────────────────
#  Trees
# ─────────────────────────────────────────────────────────────────────────────

class Graph:
    """How a feature's strokes connect: each stroke end is a node, which is a
    junction ("j", id), a mass the stroke runs into ("m", id; ink strokes only,
    since the thin mask is cut around masses), or a free end ("leaf", i, end)."""

    def __init__(self, d: dict, attach: float):
        self.strokes, self.k = d["strokes"], d["k"]
        self.J = d["junctions"]
        self.masses = {m["id"]: m for m in d["masses"]}
        S, k = self.strokes, self.k
        self.nodes = [None] * len(S)
        # mass ring vertices, for finding the mass an ink stroke ends at
        verts, owner = [], []
        for m in d["masses"]:
            for r in m["rings"]:
                verts.extend(r)
                owner.extend([m["id"]] * len(r))
        tree = cKDTree(np.asarray(verts, float)) if verts else None
        for i, s in enumerate(S):
            ends = []
            for e, jid in ((0, s["j0id"]), (1, s["j1id"])):
                if jid >= 0:
                    ends.append(("j", jid))
                    continue
                node = ("leaf", i, e)
                if tree is not None and s["pol"] == "ink":
                    p = s["pts"][0] if e == 0 else s["pts"][-1]
                    dist, vi = tree.query(p, distance_upper_bound=attach * k)
                    if np.isfinite(dist):
                        node = ("m", owner[vi])
                ends.append(node)
            self.nodes[i] = tuple(ends)

    def pos(self, node) -> np.ndarray:
        """Node position in source px."""
        if node[0] == "j":
            return np.asarray(self.J[self._pol][node[1]][:2], float) / self.k
        if node[0] == "m":
            m = self.masses[node[1]]
            return np.array([m["cx"], m["cy"]]) / self.k
        s = self.strokes[node[1]]
        return np.asarray(s["pts"][0] if node[2] == 0 else s["pts"][-1], float) / self.k

    def use(self, pol: str):
        self._pol = pol
        return self


def components(G: Graph, idx: list[int]) -> list[list[int]]:
    uf = UnionFind()
    for i in idx:
        uf.union(*G.nodes[i])
    groups = {}
    for i in idx:
        groups.setdefault(uf.find(G.nodes[i][0]), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def longest_path(G: Graph, members: list[int]) -> tuple[list[tuple[int, int]], float]:
    """Double-sweep Dijkstra: the heaviest path through the group (length times
    width, so a fat spine wins over a zigzag through thin teeth), as a list of
    (stroke index, direction) with direction 0 = pts forward, and its length."""
    strokes = G.strokes
    weight = lambda i: strokes[i]["length"] * max(strokes[i]["width"], 0.3)
    adj = {}
    for i in members:
        a, b = G.nodes[i]
        adj.setdefault(a, []).append((b, i, 0))
        adj.setdefault(b, []).append((a, i, 1))

    def sweep(start):
        dist, prev = {start: 0.0}, {}
        pq = [(0.0, 0, start)]
        n = 0
        while pq:
            d, _, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf):
                continue
            for v, i, direction in adj[u]:
                nd = d + weight(i)
                if nd < dist.get(v, np.inf) - 1e-9:
                    dist[v] = nd
                    prev[v] = (u, i, direction)
                    n += 1
                    heapq.heappush(pq, (nd, n, v))
        far = max(dist, key=dist.get)
        return far, dist[far], prev

    leaves = [n for n in adj if len(adj[n]) == 1]
    u, _, _ = sweep(leaves[0] if leaves else next(iter(adj)))
    v, L, prev = sweep(u)
    path = []
    while v in prev:
        pu, i, direction = prev[v]
        path.append((i, direction))
        v = pu
    path = path[::-1]
    return path, sum(strokes[i]["length"] for i, _ in path)


def decompose(G: Graph, members: list[int], max_simple: int = 80, min_spine: float = 10.0, max_peels: int = 200) -> list[list[int]]:
    """A large connected network is peeled into pieces: its longest path with
    the branches hanging directly off it, then the same again on what is left,
    so a comb or a branching line inside a mesh comes out as one piece."""
    work, out, peels = [list(members)], [], 0
    while work:
        m = work.pop()
        if len(m) <= max_simple or peels >= max_peels:
            out.append(m)
            continue
        path, L = longest_path(G, m)
        spine = {i for i, _ in path}
        if L < min_spine or not spine:
            out.append(m)
            continue
        nodes = {nd for i in spine for nd in G.nodes[i]}
        peel = spine | {i for i in m if i not in spine and (G.nodes[i][0] in nodes or G.nodes[i][1] in nodes)}
        peels += 1
        out.append(sorted(peel))
        rest = [i for i in m if i not in peel]
        work.extend(components(G, rest))
    return out


def tree_descriptors(G: Graph, members: list[int]) -> dict:
    strokes, k = G.strokes, G.k
    all_nodes = {nd for i in members for nd in G.nodes[i]}
    n_nodes = len(all_nodes)
    n_j = sum(1 for nd in all_nodes if nd[0] in ("j", "m"))
    cycles = len(members) - n_nodes + 1
    total = sum(strokes[i]["length"] for i in members)
    path, spine_len = longest_path(G, members)
    on_spine = {i for i, _ in path}
    # spine polyline (source px) and arc length
    pts = []
    for i, direction in path:
        p = np.asarray(strokes[i]["pts"], float) / k
        pts.append(p if direction == 0 else p[::-1])
    spine = np.vstack(pts) if pts else np.zeros((0, 2))
    seg = np.hypot(*np.diff(spine, axis=0).T) if len(spine) > 1 else np.zeros(0)
    s_at = np.concatenate([[0.0], np.cumsum(seg)])
    chord = float(np.linalg.norm(spine[-1] - spine[0])) if len(spine) > 1 else 0.0
    spine_nodes = {nd for i, _ in path for nd in G.nodes[i]}
    spine_infl = sum(strokes[i]["inflections"] for i in on_spine)
    spine_w = float(np.median([strokes[i]["width"] for i in on_spine])) if on_spine else 0.0
    # side branches
    sides, rungs, deep = [], 0, 0
    for i in members:
        if i in on_spine:
            continue
        a, b = G.nodes[i]
        on = [e for e, nd in ((0, a), (1, b)) if nd in spine_nodes]
        if len(on) == 2:
            rungs += 1
            continue
        if not on:
            deep += 1
            continue
        e = on[0]
        jx, jy = G.pos(a if e == 0 else b)
        p = np.asarray(strokes[i]["pts"], float) / k
        d = end_dir(p, e)
        at = int(np.argmin(np.hypot(spine[:, 0] - jx, spine[:, 1] - jy))) if len(spine) else 0
        lo, hi = max(0, at - 3), min(len(spine) - 1, at + 3)
        t = spine[hi] - spine[lo]
        t = t / (np.linalg.norm(t) + 1e-9)
        phi = np.degrees(np.arctan2(t[0] * d[1] - t[1] * d[0], t[0] * d[0] + t[1] * d[1]))
        sides.append({"i": i, "s": float(s_at[at]), "angle": float(min(abs(phi), 180 - abs(phi))), "side": int(np.sign(phi) or 1),
                      "length": strokes[i]["length"], "dir": d})
    d = {"n_branches": len(members), "n_junctions": n_j, "cycles": int(cycles), "total_length": total,
         "spine_length": spine_len, "spine_straightness": chord / (spine_len + 1e-9),
         "spine_inflections_per_len": spine_infl / (spine_len + 1e-9), "spine_width": spine_w,
         "n_side": len(sides), "n_rungs": rungs, "n_deep": deep,
         "side_density": len(sides) / (spine_len + 1e-9),
         "side_ratio": float(np.median([s["length"] for s in sides]) / (spine_len + 1e-9)) if sides else 0.0,
         "side_len_cv": float(np.std([s["length"] for s in sides]) / (np.mean([s["length"] for s in sides]) + 1e-9)) if sides else 0.0,
         "side_angle": float(np.median([s["angle"] for s in sides])) if sides else 0.0,
         "side_angle_spread": float(np.subtract(*np.percentile([s["angle"] for s in sides], [75, 25]))) if len(sides) > 1 else 0.0,
         "one_sidedness": float(abs(sum(s["side"] for s in sides)) / len(sides)) if sides else 0.0,
         # how parallel the side branches are to each other in the image (1 = all pointing the same way)
         "side_alignment": float(np.linalg.norm(np.mean([s["dir"] for s in sides], axis=0))) if sides else 0.0}
    if len(sides) >= 3:
        gaps = np.diff(sorted(s["s"] for s in sides))
        d["spacing"] = float(np.median(gaps))
        d["spacing_cv"] = float(gaps.std() / (gaps.mean() + 1e-9))
    else:
        d["spacing"], d["spacing_cv"] = 0.0, 1.0
    d["_spine"] = on_spine
    d["_masses"] = sorted({nd[1] for nd in all_nodes if nd[0] == "m"})
    return d


def spine_word(d: dict) -> str:
    if d["spine_inflections_per_len"] > 0.12 and d["spine_straightness"] < 0.95:
        return "wavy"
    return "curved" if d["spine_straightness"] < 0.8 else "straight"


def tree_type(d: dict, min_teeth: int) -> str | None:
    # a few stray cycles in a big structure are skeleton artefacts of a wide mark; a mesh has many
    if d["cycles"] >= 3 and d["cycles"] / d["n_branches"] > 0.12:
        return "mesh"
    if d["cycles"] >= 1 and d["n_branches"] <= 8:
        return "loop"
    if d["n_junctions"] == 1:
        return "burst" if d["n_branches"] >= 5 else "fork"
    if d["n_branches"] < 4:
        return None
    if d["n_side"] >= min_teeth and d["side_ratio"] < 0.35:
        return ("comb" if max(d["one_sidedness"], d["side_alignment"]) > 0.6 else "feather")
    if d["n_side"] <= 4 and d["side_ratio"] >= 0.35:
        return "branching line"
    if d["n_side"] >= 5:
        return "tree"
    return "sprig"


# ─────────────────────────────────────────────────────────────────────────────
#  Bundles (hatching)
# ─────────────────────────────────────────────────────────────────────────────

def bundles(strokes: list[dict], idx: list[int], k: float, gap: float, max_angle: float = 20.0, min_n: int = 4) -> list[dict]:
    """Groups of near-parallel strokes within `gap` source px of each other
    (perpendicular), overlapping along their length."""
    idx = [i for i in idx if not strokes[i]["closed"] and strokes[i]["length"] >= 1.5]
    if len(idx) < min_n:
        return []
    P = np.array([np.asarray(strokes[i]["pts"], float) / k for i in idx], dtype=object)
    mid = np.array([p.mean(0) for p in P])
    L = np.array([strokes[i]["length"] for i in idx])
    ang = np.array([strokes[i]["angle"] for i in idx])
    u = np.c_[np.cos(np.radians(ang)), np.sin(np.radians(ang))]
    tree = cKDTree(mid)
    uf = UnionFind()
    nearest = np.full(len(idx), np.inf)
    for a, b in tree.query_pairs(float(L.max()) / 2 + gap):
        if ang_diff(ang[a], ang[b]) > max_angle:
            continue
        v = mid[b] - mid[a]
        perp = abs(u[a, 0] * v[1] - u[a, 1] * v[0])
        along = abs(u[a, 0] * v[0] + u[a, 1] * v[1])
        if perp <= gap and perp > 0.2 and along <= 0.7 * (L[a] + L[b]) / 2:
            uf.union(a, b)
            nearest[a], nearest[b] = min(nearest[a], perp), min(nearest[b], perp)
    groups = {}
    for a in range(len(idx)):
        groups.setdefault(uf.find(a), []).append(a)
    out = []
    for g in groups.values():
        if len(g) < min_n:
            continue
        g = np.array(g)
        o, spread = circ_mean_deg(ang[g], L[g])
        sp = nearest[g][np.isfinite(nearest[g])]
        allp = np.vstack([P[a] for a in g])
        out.append({"members": [idx[a] for a in g], "n": len(g), "orientation": o, "spread": spread,
                    "spacing": float(np.median(sp)) if len(sp) else 0.0, "spacing_cv": float(sp.std() / (sp.mean() + 1e-9)) if len(sp) else 0.0,
                    "length": float(np.median(L[g])), "straightness": float(np.median([strokes[idx[a]]["straightness"] for a in g])),
                    "on_mass": float(np.mean([strokes[idx[a]]["mass"] >= 0 for a in g])),
                    "bbox": [float(v) for v in (*allp.min(0), *allp.max(0))]})
    return out


def overlap(b1, b2) -> float:
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    a1, a2 = (b1[2] - b1[0]) * (b1[3] - b1[1]), (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (min(a1, a2) + 1e-9)


def pair_crosshatch(bs: list[dict]) -> list[dict]:
    """Merge bundles that overlap in place and cross at 45° or more."""
    used, out = set(), []
    for i, b in enumerate(bs):
        if i in used:
            continue
        best = None
        for j in range(i + 1, len(bs)):
            if j in used:
                continue
            if ang_diff(b["orientation"], bs[j]["orientation"]) >= 45 and overlap(b["bbox"], bs[j]["bbox"]) >= 0.3:
                best = j
                break
        if best is None:
            out.append({**b, "kind": "hatching"})
        else:
            c = bs[best]
            used.add(best)
            out.append({"members": b["members"] + c["members"], "n": b["n"] + c["n"], "kind": "crosshatching",
                        "orientation": b["orientation"], "cross_angle": ang_diff(b["orientation"], c["orientation"]),
                        "spread": max(b["spread"], c["spread"]), "spacing": (b["spacing"] + c["spacing"]) / 2,
                        "spacing_cv": max(b["spacing_cv"], c["spacing_cv"]), "length": (b["length"] + c["length"]) / 2,
                        "straightness": min(b["straightness"], c["straightness"]), "on_mass": (b["on_mass"] + c["on_mass"]) / 2,
                        "bbox": [min(b["bbox"][0], c["bbox"][0]), min(b["bbox"][1], c["bbox"][1]), max(b["bbox"][2], c["bbox"][2]), max(b["bbox"][3], c["bbox"][3])]})
    return out


def bundle_type(b: dict) -> str:
    word = "crosshatching" if b["kind"] == "crosshatching" else ("wavy hatching" if b["straightness"] < 0.85 else "hatching")
    return ("fine " if b["spacing"] < 2.5 else "") + word


# ─────────────────────────────────────────────────────────────────────────────
#  Bands
# ─────────────────────────────────────────────────────────────────────────────

def band_descriptors(G: Graph, spine_idx: list[int], mass: dict) -> dict | None:
    strokes, k = G.strokes, G.k
    if not spine_idx:
        return None
    if len(spine_idx) == 1:
        s = strokes[spine_idx[0]]
        path, L = [(spine_idx[0], 0)], s["length"]
        on_spine, arms = {spine_idx[0]}, 0
        straight, infl = s["straightness"], s["inflections"]
    else:
        d = tree_descriptors(G, spine_idx)
        on_spine, L = d["_spine"], d["spine_length"]
        if not on_spine:   # the mass's skeleton pieces do not connect: take the longest piece as its spine
            best = max(spine_idx, key=lambda i: strokes[i]["length"])
            on_spine, L = {best}, strokes[best]["length"]
            d["spine_straightness"], d["spine_inflections_per_len"] = strokes[best]["straightness"], strokes[best]["inflections"] / max(L, 1e-9)
        straight, infl = d["spine_straightness"], d["spine_inflections_per_len"] * L
        arms = sum(1 for i in spine_idx if i not in on_spine and strokes[i]["length"] > 0.0)
    ws = np.concatenate([np.asarray(strokes[i]["w"], float) / k for i in on_spine])
    width = float(np.median(ws))
    if L < 1e-6:
        return None
    ends = [t for i in on_spine for t in (strokes[i]["taper_start"], strokes[i]["taper_end"]) if t is not None]
    return {"length": L, "width": width, "aspect": L / (width + 1e-9), "straightness": straight,
            "inflections_per_len": infl / L, "taper": float(np.mean(ends)) if ends else 1.0,
            "arms": int(sum(1 for i in spine_idx if i not in on_spine and strokes[i]["length"] > width)),
            "area": mass["area"], "_spine": on_spine}


# ─────────────────────────────────────────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────────────────────────────────────────

class Scene:
    """Things to draw, in source px: filled rings (masses), strokes, discs."""

    def __init__(self):
        self.items = []           # (kind, data, black: bool)

    def mass(self, rings, k):
        self.items.append(("rings", [np.asarray(r, float) / k for r in rings], True))

    def stroke(self, s, k, black=True):
        pts = np.asarray(s["pts"], float) / k
        w = np.asarray(s["w"], float) / k
        self.items.append(("stroke", (pts, w, s["closed"]), black))

    def disc(self, x, y, r, black=True):
        self.items.append(("disc", (x, y, r), black))

    def bounds(self):
        lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
        for kind, data, _ in self.items:
            if kind == "rings":
                for r in data:
                    lo, hi = np.minimum(lo, r.min(0)), np.maximum(hi, r.max(0))
            elif kind == "stroke":
                pts, w, _ = data
                lo, hi = np.minimum(lo, pts.min(0) - w.max() / 2), np.maximum(hi, pts.max(0) + w.max() / 2)
            else:
                x, y, r = data
                lo, hi = np.minimum(lo, (x - r, y - r)), np.maximum(hi, (x + r, y + r))
        return lo, hi

    def svg(self, margin: float = 1.0, px_per_unit: float = 8.0) -> str:
        lo, hi = self.bounds()
        lo, hi = lo - margin, hi + margin
        W, H = (hi - lo) * px_per_unit
        parts = []
        for kind, data, black in self.items:
            fill = "#000" if black else "#fff"
            if kind == "rings":
                parts.append(f'<path fill="{fill}" fill-rule="evenodd" d="{rings_path([(r - lo) * px_per_unit for r in data])}"/>')
            elif kind == "stroke":
                pts, w, closed = data
                rings = offset_rings(pts[:, 0], pts[:, 1], w, closed)
                if rings:
                    parts.append(f'<path fill="{fill}" fill-rule="nonzero" d="{rings_path([(r - lo) * px_per_unit for r in rings])}"/>')
            else:
                x, y, r = data
                parts.append(f'<circle cx="{f1((x - lo[0]) * px_per_unit)}" cy="{f1((y - lo[1]) * px_per_unit)}" r="{f1(r * px_per_unit)}" fill="{fill}"/>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {f1(W)} {f1(H)}" width="{f1(W)}" height="{f1(H)}">'
                + "".join(parts) + "</svg>")

    def raster(self, cell: int, max_scale: float, pad: int = 8, ss: int = 2) -> Image.Image:
        """The compound fitted to the cell, but never enlarged beyond max_scale px
        per source px; a tick in the corner shows 10 source px at the cell's scale."""
        lo, hi = self.bounds()
        span = hi - lo
        scale = min(max_scale, (cell - 2 * pad) / max(float(span.max()), 1e-6))
        off = (np.array([cell, cell]) - span * scale) / 2 - lo * scale
        im = Image.new("L", (cell * ss, cell * ss), 255)
        dr = ImageDraw.Draw(im)
        dr.line([(4 * ss, (cell - 4) * ss), ((4 + 10 * scale) * ss, (cell - 4) * ss)], fill=150, width=ss)
        for kind, data, black in self.items:
            col = 0 if black else 255
            if kind == "rings":
                rings = sorted(data, key=lambda r: -abs(signed_area(r)))
                outer_sign = np.sign(signed_area(rings[0])) if rings else 1
                for r in rings:
                    q = r * scale + off
                    hole = np.sign(signed_area(r)) != outer_sign
                    dr.polygon([tuple(p) for p in q * ss], fill=(255 - col) if hole else col)
            elif kind == "stroke":
                pts, w, closed = data
                P = (pts * scale + off) * ss
                R = w * scale * ss / 2
                if len(P) < 2:
                    continue
                th = np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0]))
                nx, ny = np.sin(th), -np.cos(th)
                pairs = list(zip(range(len(P) - 1), range(1, len(P)))) + ([(len(P) - 1, 0)] if closed else [])
                for i, j in pairs:
                    dr.polygon([(P[i, 0] + nx[i] * R[i], P[i, 1] + ny[i] * R[i]), (P[j, 0] + nx[j] * R[j], P[j, 1] + ny[j] * R[j]),
                                (P[j, 0] - nx[j] * R[j], P[j, 1] - ny[j] * R[j]), (P[i, 0] - nx[i] * R[i], P[i, 1] - ny[i] * R[i])], fill=col)
                for (x, y), r in zip(P, R):
                    dr.ellipse([x - r, y - r, x + r, y + r], fill=col)
            else:
                x, y, r = data
                cx, cy = (np.array([x, y]) * scale + off) * ss
                rr = r * scale * ss
                dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col)
        return im.resize((cell, cell), Image.LANCZOS)


def sheet(rows: list[tuple[str, list[Scene]]], cell: int, max_scale: float, label_w: int = 300) -> Image.Image:
    font = ImageFont.load_default(size=13)
    n = max(len(r[1]) for r in rows) if rows else 1
    W, H = label_w + n * cell, len(rows) * cell + 30
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    for i, (text, scenes) in enumerate(rows):
        y = i * cell
        if i:
            dr.line([(0, y), (W, y)], fill=(225, 225, 225))
        for j, line in enumerate(text.split("\n")):
            dr.text((10, y + 10 + 16 * j), line, fill=(0, 0, 0) if j == 0 else (110, 110, 110), font=font)
        for j, sc in enumerate(scenes):
            im.paste(sc.raster(cell, max_scale).convert("RGB"), (label_w + j * cell, y))
    dr.text((10, H - 22), "each cell is fitted to its compound; the grey tick in a cell is 10 source px at that cell's scale", fill=(0, 0, 0), font=font)
    return im


# ─────────────────────────────────────────────────────────────────────────────
#  Per feature
# ─────────────────────────────────────────────────────────────────────────────

def find_compounds(d: dict, gap: float, min_teeth: int, min_band_aspect: float, attach: float) -> list[dict]:
    strokes, k, J = d["strokes"], d["k"], d["junctions"]
    G = Graph(d, attach)
    out = []
    by_pol = {p: [i for i, s in enumerate(strokes) if s["pol"] == p] for p in ("ink", "relief", "mass")}
    masses = G.masses

    # trees
    for pol in ("ink", "relief"):
        G.use(pol)
        for members in (piece for comp in components(G, by_pol[pol]) for piece in decompose(G, comp)):
            if len(members) < 3 or len(members) > 400:
                continue
            td = tree_descriptors(G, members)
            t = tree_type(td, min_teeth)
            if t is None or td["spine_length"] < 6:
                continue
            label = t if t in ("mesh", "loop", "burst", "fork") else f"{spine_word(td)} {t}"
            sc = Scene()
            for mid in td["_masses"]:
                sc.mass(masses[mid]["rings"], k)
            for i in members:
                sc.stroke(strokes[i], k)
            jids = {nd[1] for i in members for nd in G.nodes[i] if nd[0] == "j"}
            for j in jids:
                x, y, r = J[pol][j]
                sc.disc(x / k, y / k, r / k)
            out.append({"kind": "tree", "type": label, "pol": pol, "size": td["spine_length"], "n": len(members),
                        "desc": {kk: v for kk, v in td.items() if not kk.startswith("_")} | {"n_masses": len(td["_masses"])}, "scene": sc})

    # bundles
    for pol in ("ink", "relief"):
        for b in pair_crosshatch(bundles(strokes, by_pol[pol], k, gap)):
            sc = Scene()
            for i in b["members"]:
                sc.stroke(strokes[i], k)
            desc = {kk: v for kk, v in b.items() if kk not in ("members", "bbox", "kind")}
            out.append({"kind": "bundle", "type": bundle_type(b), "pol": pol,
                        "size": b["n"], "n": b["n"], "desc": desc, "scene": sc})

    # bands
    spines_by_mass = {}
    for i in by_pol["mass"]:
        spines_by_mass.setdefault(strokes[i]["mass"], []).append(i)
    relief_by_mass = {}
    for i in by_pol["relief"]:
        if strokes[i]["mass"] >= 0:
            relief_by_mass.setdefault(strokes[i]["mass"], []).append(i)
    G.use("mass")
    for mid, m in masses.items():
        bd = band_descriptors(G, spines_by_mass.get(mid, []), m)
        if bd is None:
            continue
        hatch = pair_crosshatch(bundles(strokes, relief_by_mass.get(mid, []), k, gap)) if len(relief_by_mass.get(mid, [])) >= 4 else []
        hatched = "crosshatched" if any(h["kind"] == "crosshatching" for h in hatch) else ("hatched" if hatch else "")
        is_band = bd["aspect"] >= min_band_aspect and bd["length"] >= 8
        if not is_band and not hatched:
            continue
        shape = ("lobed " if bd["arms"] >= 2 else "") + (f"{spine_word({'spine_inflections_per_len': bd['inflections_per_len'], 'spine_straightness': bd['straightness']})} band" if is_band else "blob")
        label = f"{hatched} {shape}".strip()
        sc = Scene()
        sc.mass(m["rings"], k)
        for h in hatch:
            for i in h["members"]:
                sc.stroke(strokes[i], k, black=False)
        out.append({"kind": "band", "type": label, "pol": "mass", "size": bd["length"], "n": 1 + sum(h["n"] for h in hatch),
                    "desc": {kk: v for kk, v in bd.items() if not kk.startswith("_")} | {"hatch_bundles": len(hatch)}, "scene": sc})
    return out


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "linework" / "vgg16features"))
    ap.add_argument("--dst", default="", help="default: compounds/<src folder name>")
    ap.add_argument("--per-type", type=int, default=16, help="representatives per type: half most typical, half largest")
    ap.add_argument("--gap", type=float, default=4.0, help="max spacing between strokes of a bundle, source px")
    ap.add_argument("--min-teeth", type=int, default=4, help="side branches needed for a comb or feather")
    ap.add_argument("--min-band-aspect", type=float, default=3.0, help="length / width for a mass to count as a band")
    ap.add_argument("--attach", type=float, default=2.5, help="an ink stroke ending within this of a mass (source px) connects through it")
    ap.add_argument("--min-count", type=int, default=5, help="types with fewer compounds than this are not shown")
    ap.add_argument("--cell", type=int, default=150)
    ap.add_argument("--max-scale", type=float, default=5.0, help="never draw a compound larger than this many px per source px")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = Path(a.src)
    dst = Path(a.dst) if a.dst else HERE / "compounds" / src.name
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feats, found = [], []
    for p in sorted(src.glob("*.strokes.json")):
        d = json.loads(p.read_text())
        if "junctions" not in d:
            print(f"  {p.name}: no junction data; rerun linework.py --force first")
            sys.exit(1)
        fi = len(feats)
        feats.append({"name": d["name"], "layer": d["layer"]})
        cs = find_compounds(d, a.gap, a.min_teeth, a.min_band_aspect, a.attach)
        for c in cs:
            c["f"] = fi
            lo, hi = c["scene"].bounds()
            c["extent"] = float((hi - lo).max()) / d["source_px"]
        found.extend(cs)
        print(f"  {fi + 1}  {d['name'][:48]}  {len(cs)} compounds  ({time.time() - t0:.0f}s)", flush=True)

    layers = sorted({f["layer"] for f in feats}, key=lambda l: [int(p) if p.isdigit() else p for p in l.split(".")])
    types = {}
    for c in found:
        types.setdefault(c["type"], []).append(c)
    order = sorted(types, key=lambda t: -len(types[t]))
    rows, report, picked_all = [], [], []
    for t in order:
        cs = types[t]
        if len(cs) < a.min_count:
            continue
        keys = sorted({kk for c in cs for kk, v in c["desc"].items() if isinstance(v, (int, float))})
        X = np.array([[c["desc"].get(kk, 0.0) for kk in keys] for c in cs], float)
        Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        dist = np.linalg.norm(Z - np.median(Z, 0), axis=1)
        picked, per_feat = [], {}
        half = a.per_type // 2
        for pool in (np.argsort(dist), np.argsort([-c["size"] for c in cs])):
            got = 0
            for i in pool:
                c = cs[i]
                if c in picked or per_feat.get(c["f"], 0) >= 2 or c["extent"] > 0.6:   # a structure spanning most of the image is the texture, not a motif
                    continue
                per_feat[c["f"]] = per_feat.get(c["f"], 0) + 1
                picked.append(c)
                got += 1
                if got >= half:
                    break
        tdir = dst / slug(t)
        tdir.mkdir(exist_ok=True)
        reps = []
        for j, c in enumerate(picked):
            (tdir / f"{j:02d}.svg").write_text(c["scene"].svg())
            reps.append({"svg": f"{tdir.name}/{j:02d}.svg", "feature": feats[c["f"]]["name"], "layer": feats[c["f"]]["layer"],
                         "pol": c["pol"], "n_strokes": c["n"], **{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in c["desc"].items()}})
        typical = {kk: round(float(np.median(X[:, i])), 3) for i, kk in enumerate(keys)}
        by_layer = {ly: sum(1 for c in cs if feats[c["f"]]["layer"] == ly) for ly in layers}
        report.append({"type": t, "kind": cs[0]["kind"], "n": len(cs), "share": len(cs) / len(found), "typical": typical,
                       "by_layer": by_layer, "representatives": reps, "dir": tdir.name})
        rows.append((f"{t}\n{len(cs)} found  ·  typical {'spine ' if cs[0]['kind'] == 'tree' else ''}L {typical.get('spine_length', typical.get('length', 0)):.0f}",
                     [c["scene"] for c in picked]))
        picked_all.append((tdir, rows[-1]))
        print(f"  {t:36s} {len(cs):6d}   {', '.join(f'{kk} {v:.2f}' for kk, v in list(typical.items())[:5])}", flush=True)

    if rows:
        sheet(rows, a.cell, a.max_scale).save(dst / "compounds.png", optimize=True)
        for tdir, row in picked_all:
            sheet([row], a.cell * 2, a.max_scale * 2, 360).save(tdir / "sheet.png", optimize=True)
    (dst / "compounds.json").write_text(json.dumps({"src": str(src), "n_compounds": len(found), "layers": layers, "types": report}, indent=1))
    with open(dst / "by_feature.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "layer"] + order)
        for fi, f in enumerate(feats):
            w.writerow([f["name"], f["layer"]] + [sum(1 for c in types[t] if c["f"] == fi) for t in order])
    print(f"\n  compounds per layer (count, averaged over the layer's features):")
    shown = [r["type"] for r in report]
    print("  " + "layer".ljust(14) + "".join(t[:12].rjust(13) for t in shown))
    for ly in layers:
        nf = sum(1 for f in feats if f["layer"] == ly)
        print("  " + ly.ljust(14) + "".join(f"{r['by_layer'][ly] / nf:13.1f}" for r in report))
    print(f"\n  {len(found)} compounds, {time.time() - t0:.0f}s → {dst}")


if __name__ == "__main__":
    main()
