/* Grammar of Ornament — shapes drawn to fit each element.
   Any element with data-shape="<kind>" gets an SVG behind its content whose
   outline is computed from the element's actual size, so curves and points
   never stretch. Options: data-seed (irregular shapes), data-r, data-top,
   data-inner (an inner rule this many px in), data-clip (also clip the
   element's content, e.g. a card's picture, to the shape). Colours come from
   CSS: --shape-fill, --shape-line, --shape-stroke. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  function rng(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const f = (v) => (Math.round(v * 10) / 10).toString();
  const poly = (pts) => "M" + pts.map((p) => f(p[0]) + " " + f(p[1])).join("L") + "Z";

  // a woodblock edge: a few vertices with small irregularities, and now and then a chip gouged out
  function edge(r, x0, x1, y, inward, chips) {
    const pts = [];
    const n = Math.max(2, Math.round(Math.abs(x1 - x0) / 34));
    const dir = x1 > x0 ? 1 : -1;
    for (let i = 1; i < n; i++) {
      const x = x0 + ((x1 - x0) * i) / n + (r() - 0.5) * 10;
      pts.push([x, y + inward * r() * 2.2]);
      if (r() < chips) {
        pts.push([x + dir * (2 + r() * 2), y + inward * (3 + r() * 4)]);
        pts.push([x + dir * (6 + r() * 3), y + inward * r() * 1.2]);
      }
    }
    return pts;
  }

  const SHAPES = {
    rect: (w, h) => `M0 0H${f(w)}V${f(h)}H0Z`,

    // woodcut: an irregular slab with chipped corners and gouged edges
    slab(w, h, o) {
      const r = rng(o.seed), c = Math.min(6, h * 0.18);
      const pts = [[r() * c, c * 0.6 + r() * 2], [c * 0.5 + r() * 3, r() * 2]];
      pts.push(...edge(r, 0, w, 0, 1, 0.35));
      pts.push([w - c * 0.5 - r() * 3, r() * 2], [w - r() * c * 0.6, c * 0.7 + r() * 2]);
      pts.push([w - r() * 3, h * 0.5 + (r() - 0.5) * 4]);
      pts.push([w - r() * c * 0.6, h - c * 0.7 - r() * 2], [w - c * 0.5 - r() * 3, h - r() * 2]);
      pts.push(...edge(r, w, 0, h, -1, 0.3));
      pts.push([c * 0.5 + r() * 3, h - r() * 2], [r() * c * 0.6, h - c * 0.7 - r() * 2]);
      pts.push([r() * 3, h * 0.5 + (r() - 0.5) * 4]);
      return poly(pts);
    },
    // woodcut card: slab with a torn bottom edge
    slabTorn(w, h, o) {
      const r = rng(o.seed), t = 9;
      const pts = [[r() * 4, 6 + r() * 3], [5 + r() * 3, r() * 2]];
      pts.push(...edge(r, 0, w, 0, 1, 0.25));
      pts.push([w - 5 - r() * 3, r() * 2], [w - r() * 4, 6 + r() * 3]);
      pts.push([w - r() * 3, h * 0.5]);
      pts.push([w - r() * 2, h - t]);
      const n = Math.round(w / 9);
      for (let i = n - 1; i > 0; i--) pts.push([(w * i) / n + (r() - 0.5) * 5, h - t + (i % 2 ? r() * t : -r() * 3)]);
      pts.push([r() * 2, h - t * 0.5], [r() * 3, h * 0.5]);
      return poly(pts);
    },

    // engraving: a cartouche, the ends scooped inward
    cartouche(w, h) {
      const b = Math.min(h * 0.34, w * 0.09);
      return `M0 0H${f(w)}Q${f(w - 2 * b)} ${f(h / 2)} ${f(w)} ${f(h)}H0Q${f(2 * b)} ${f(h / 2)} 0 0Z`;
    },
    // engraving card: concave quarter-circle corners
    cartoucheCard(w, h, o) {
      const r = Math.min(o.r || 20, w / 4, h / 4);
      return `M${f(r)} 0H${f(w - r)}A${f(r)} ${f(r)} 0 0 0 ${f(w)} ${f(r)}V${f(h - r)}A${f(r)} ${f(r)} 0 0 0 ${f(w - r)} ${f(h)}` +
             `H${f(r)}A${f(r)} ${f(r)} 0 0 0 0 ${f(h - r)}V${f(r)}A${f(r)} ${f(r)} 0 0 0 ${f(r)} 0Z`;
    },
    // engraving: a ribbon with swallowtail ends
    ribbon(w, h) {
      const n = Math.min(h * 0.42, 12);
      return `M0 0H${f(w)}L${f(w - n)} ${f(h / 2)}L${f(w)} ${f(h)}H0L${f(n)} ${f(h / 2)}Z`;
    },
    // engraving alert: a flag, swallowtail on the right only
    flag(w, h) {
      const n = Math.min(h * 0.3, 16);
      return `M0 0H${f(w)}L${f(w - n)} ${f(h / 2)}L${f(w)} ${f(h)}H0Z`;
    },

    // filigree: an ogee leaf, pointed at both ends with S-curved shoulders
    leaf(w, h) {
      const a = Math.min(h * 1.05, w * 0.24);
      return `M0 ${f(h / 2)}C${f(a * 0.5)} ${f(h / 2)} ${f(a * 0.3)} 0 ${f(a)} 0H${f(w - a)}` +
             `C${f(w - a * 0.3)} 0 ${f(w - a * 0.5)} ${f(h / 2)} ${f(w)} ${f(h / 2)}` +
             `C${f(w - a * 0.5)} ${f(h / 2)} ${f(w - a * 0.3)} ${f(h)} ${f(w - a)} ${f(h)}H${f(a)}` +
             `C${f(a * 0.3)} ${f(h)} ${f(a * 0.5)} ${f(h / 2)} 0 ${f(h / 2)}Z`;
    },
    // a real piece from the atlas (window.GOAtlas[group]), its outline scaled to the element; the seed picks the piece
    atlas(w, h, o, el) {
      const g = (window.GOAtlas || {})[el.dataset.group] || [];
      if (!g.length) return SHAPES.rect(w, h);
      const p = g[Math.abs(o.seed) % g.length];
      const sx = w / p.w, sy = h / p.h;
      return p.d.replace(/(-?\d+\.?\d*),(-?\d+\.?\d*)/g, (m, x, y) => f(x * sx) + "," + f(y * sy));
    },

    // a box whose sides are drawn by four real strokes (see EXTRAS.frame); the fill is the box itself
    frame: (w, h) => `M0 0H${f(w)}V${f(h)}H0Z`,

    // filigree card: a pointed ogee arch over straight sides
    arch(w, h, o) {
      const t = Math.min(o.top || w * 0.4, h * 0.6);
      return `M0 ${f(h)}V${f(t)}C0 ${f(t * 0.45)} ${f(w * 0.32)} ${f(t * 0.52)} ${f(w / 2)} 0` +
             `C${f(w * 0.68)} ${f(t * 0.52)} ${f(w)} ${f(t * 0.45)} ${f(w)} ${f(t)}V${f(h)}Z`;
    },
  };

  // extra marks drawn with a shape, in the line colour
  const EXTRAS = {
    frame(w, h, o, el) {
      const g = (window.GOAtlas || {}).frame || [];
      if (!g.length) return "";
      const r = rng(o.seed * 7 + 3), T = +(el.dataset.t || 5);
      let out = "";
      for (let i = 0; i < 4; i++) {
        const p = g[Math.floor(r() * g.length)];
        const side = i < 2 ? w : h;
        const L = side * (1.05 + r() * 0.05), off = -(L - side) / 2 + (r() - 0.5) * 4;
        const sx = L / p.w, sy = (T / p.h) * (0.85 + r() * 0.35), j = (r() - 0.5) * 1.5;
        const tf = [`translate(${f(off)},${f(-T / 2 + j)})`, `translate(${f(off)},${f(h - T / 2 + j)})`,
                    `translate(${f(T / 2 + j)},${f(off)}) rotate(90)`, `translate(${f(w + T / 2 + j)},${f(off)}) rotate(90)`][i];
        out += `<path class="stroke" transform="${tf} scale(${sx.toFixed(4)},${sy.toFixed(4)})" d="${p.d}"/>`;
      }
      return out;
    },
  };

  function draw(el) {
    const kind = el.dataset.shape, fn = SHAPES[kind];
    const w = el.offsetWidth, h = el.offsetHeight;
    if (!fn || !w || !h) return;
    const o = { seed: +(el.dataset.seed || 1), r: +(el.dataset.r || 0), top: +(el.dataset.top || 0) };
    let svg = el.querySelector(":scope > svg.shape");
    if (!svg) {
      svg = document.createElementNS(NS, "svg");
      svg.setAttribute("class", "shape");
      svg.setAttribute("aria-hidden", "true");
      el.prepend(svg);
    }
    // data-contain: size the shape so the largest rectangle inside it covers the element's box,
    // so the text (the element's content) always lies within the outline
    let ox = 0, oy = 0, W = w, H = h;
    if (kind === "atlas" && el.hasAttribute("data-contain")) {
      const g = (window.GOAtlas || {})[el.dataset.group] || [];
      const p = g.length ? g[Math.abs(o.seed) % g.length] : null;
      if (p && p.inset) {
        const [x0, y0, x1, y1] = p.inset;
        W = w / (x1 - x0); H = h / (y1 - y0); ox = -x0 * W; oy = -y0 * H;
      }
    }
    const sw = parseFloat(getComputedStyle(el).getPropertyValue("--shape-stroke")) || 0;
    const k = sw / 2, ins = +(el.dataset.inner || 0);
    const at = (d, dx) => (dx ? `transform="translate(${f(dx)},${f(dx)})" ` : "") + `d="${d}"`;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    svg.style.left = ox + "px";
    svg.style.top = oy + "px";
    const d = fn(W, H, o, el);
    svg.innerHTML =
      `<path class="fill" ${at(d, 0)}/>` +
      (sw ? `<path class="line" ${at(fn(W - 2 * k, H - 2 * k, o, el), k)}/>` : "") +
      (ins ? `<path class="inner" ${at(fn(W - 2 * ins, H - 2 * ins, o, el), ins)}/>` : "") +
      (EXTRAS[kind] ? EXTRAS[kind](W, H, o, el) : "") +
      `<path class="focus" ${at(fn(W + 10, H + 10, o, el), -5)}/>`;
    if (el.hasAttribute("data-clip")) el.style.clipPath = `path('${d}')`;
  }

  const ro = new ResizeObserver((entries) => entries.forEach((e) => draw(e.target)));
  function drawAll(root) {
    (root || document).querySelectorAll("[data-shape]").forEach((el) => { draw(el); ro.observe(el); });
  }
  window.GOShapes = { draw, drawAll, SHAPES };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => drawAll());
  else drawAll();
  if (document.fonts) document.fonts.ready.then(() => drawAll());
})();
