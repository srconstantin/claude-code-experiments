/* Neuron elements — optional web components for the Neuron library.
 *
 * Pieces are the library's own compound strokes and single strokes (see PIECES below), drawn as SVG sized to
 * the element, so they fit any width without stretching out of shape. All are filled with currentColor.
 *
 *   <neuron-divider piece="straight-band-13" mode="repeat|single|mirror" centre="curved-comb-05" height="28">
 *   <neuron-frieze piece="wavy-band-00" height="40" gap="0.3">
 *   <neuron-frame piece="loop-00" edge="f07-short-wave-branch-fine-15" size="72"> …content… </neuron-frame>
 *   <neuron-fleuron piece="fork-04" fold="6" mirror size="96">
 *   <neuron-box weight="3"> …content… </neuron-box>        a box drawn with four single strokes
 *   <neuron-medallion motif="f24-c27-3655" form="filled|duo|silhouette" fold="6" size="480">
 *   <neuron-palette colorway="f10-c100"> …content… </neuron-palette>
 *
 * List the available pieces in the console with NeuronElements.pieces(); each has roles (rules, compact,
 * curved, petals, tiles, lines) saying what it was picked for, but any piece works in any element.
 */
(() => {
  const PIECES = /*PIECES*/{};
  const MOTIFS = /*MOTIFS*/{};        // id → {axis, aspect, filled}: how each motif lies, for medallions
  const NS = "http://www.w3.org/2000/svg";
  const keys = Object.keys(PIECES);
  const byRole = role => keys.filter(k => PIECES[k].roles.includes(role));
  const get = (name, role) => PIECES[name] || PIECES[byRole(role)[0]] || PIECES[keys[0]];

  function svg(w, h) {
    const s = document.createElementNS(NS, "svg");
    s.setAttribute("viewBox", `0 0 ${w} ${h}`);
    s.setAttribute("width", w);
    s.setAttribute("height", h);
    s.setAttribute("fill", "currentColor");
    s.setAttribute("aria-hidden", "true");
    s.style.display = "block";
    s.style.overflow = "visible";
    return s;
  }

  // the piece drawn into box (x, y, w, h); flipX puts its heavy end right, flipY turns it over
  function place(parent, p, x, y, w, h, flipX = false, flipY = false, extra = "") {
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", p.d);
    path.setAttribute("fill-rule", "evenodd");
    const sx = w / p.w, sy = h / p.h;
    path.setAttribute("transform",
      `${extra} translate(${x + (flipX ? w : 0)} ${y + (flipY ? h : 0)}) scale(${flipX ? -sx : sx} ${flipY ? -sy : sy})`);
    parent.appendChild(path);
    return path;
  }

  class Base extends HTMLElement {
    connectedCallback() {
      if (!this.style.display && getComputedStyle(this).display === "inline") this.style.display = "block";   // inline has no width
      if (!this._ro) {
        this._ro = new ResizeObserver(() => this.draw());
        this._ro.observe(this);
      }
      this.draw();
    }
    disconnectedCallback() { this._ro && this._ro.disconnect(); this._ro = null; }
    attributeChangedCallback() { this.isConnected && this.draw(); }
    num(name, d) { const v = parseFloat(this.getAttribute(name)); return isNaN(v) ? d : v; }
  }

  // a row of copies, each stretched by at most a little so a whole number fills the width exactly
  function row(g, p, x0, width, h, gap, glide, flipX) {
    const unit = p.w * h / p.h;
    const n = Math.max(1, Math.round(width / (unit * (1 + gap))));
    const step = width / n, w = step / (1 + gap);
    for (let i = 0; i < n; i++) place(g, p, x0 + i * step + (step - w) / 2, 0, w, h, flipX, glide && i % 2 === 1);
  }

  class Divider extends Base {
    static get observedAttributes() { return ["piece", "mode", "centre", "height", "gap"]; }
    draw() {
      const W = this.clientWidth; if (!W) return;
      const h = this.num("height", 28), mode = this.getAttribute("mode") || "repeat";
      const p = get(this.getAttribute("piece"), "rules");
      const s = svg(W, h);
      if (mode === "single") {
        const w = Math.min(W, p.w * h / p.h), hh = w * p.h / p.w;
        place(s, p, (W - w) / 2, (h - hh) / 2, w, hh);
      } else if (mode === "mirror") {
        const c = get(this.getAttribute("centre"), "compact");
        const cw = Math.min(h * 1.1, h * c.w / c.h), ch = cw * c.h / c.w;
        const gap = h * 0.4, arm = (W - cw) / 2 - gap, ah = h * 0.62;
        const left = document.createElementNS(NS, "g"), right = document.createElementNS(NS, "g");
        left.setAttribute("transform", `translate(0 ${(h - ah) / 2})`);
        right.setAttribute("transform", `translate(${W - arm} ${(h - ah) / 2})`);
        row(left, p, 0, arm, ah, 0, false, true);        // heavy ends toward the centre
        row(right, p, 0, arm, ah, 0, false, false);
        s.append(left, right);
        place(s, c, (W - cw) / 2, (h - ch) / 2, cw, ch);
      } else {
        row(s, p, 0, W, h, this.num("gap", 0.08), true, false);
      }
      this.replaceChildren(s);
    }
  }

  class Frieze extends Base {
    static get observedAttributes() { return ["piece", "height", "gap"]; }
    draw() {
      const W = this.clientWidth; if (!W) return;
      const h = this.num("height", 40), p = get(this.getAttribute("piece"), "tiles");
      const s = svg(W, h);
      row(s, p, 0, W, h * 0.8, this.num("gap", 0.3), false, false);
      s.firstChild && [...s.children].forEach(c => c.setAttribute("transform", `translate(0 ${h * 0.1}) ` + c.getAttribute("transform")));
      this.replaceChildren(s);
    }
  }

  class Frame extends Base {
    static get observedAttributes() { return ["piece", "edge", "size", "weight"]; }
    constructor() {
      super();
      const r = this.attachShadow({ mode: "open" });
      r.innerHTML = `<style>:host{display:block;position:relative}
        .o{position:absolute;inset:0;pointer-events:none}
        .c{position:relative;padding:var(--n-pad, calc(var(--s) * .62))}</style>
        <div class="o"></div><div class="c"><slot></slot></div>`;
    }
    draw() {
      const W = this.clientWidth, H = this.clientHeight; if (!W || !H) return;
      const size = Math.min(this.num("size", 72), W / 2.2, H / 2.2), t = this.num("weight", size * 0.1);
      this.style.setProperty("--s", size + "px");
      const p = get(this.getAttribute("piece"), "curved"), e = get(this.getAttribute("edge"), "lines");
      const s = svg(W, H), ah = size * 0.3;
      const corner = t => {
        const g = document.createElementNS(NS, "g");
        g.setAttribute("transform", t);
        place(g, p, 0, 0, size, ah);                                   // along the top, heavy end in the corner
        place(g, p, 0, 0, size, ah, false, false, "matrix(0 1 1 0 0 0)");   // and down the side
        s.appendChild(g);
      };
      corner("");
      corner(`translate(${W} 0) scale(-1 1)`);
      corner(`translate(0 ${H}) scale(1 -1)`);
      corner(`translate(${W} ${H}) scale(-1 -1)`);
      // the edges: one stroke stretched along each side between the corners, its taper kept, the heavy ends
      // with a half-turn symmetry (left on top, right on the bottom, bottom on the left side)
      const ins = size * 0.16, L = W - 2 * size, V = H - 2 * size;
      const edge = (transform, len, flip) => {
        if (len <= 0) return;
        const g = document.createElementNS(NS, "g");
        g.setAttribute("transform", transform);
        place(g, e, 0, -t / 2, len, t, flip);
        s.appendChild(g);
      };
      edge(`translate(${size} ${ins})`, L, false);
      edge(`translate(${size} ${H - ins})`, L, true);
      edge(`translate(${ins} ${H - size}) rotate(-90)`, V, false);
      edge(`translate(${W - ins} ${size}) rotate(90)`, V, false);
      this.shadowRoot.querySelector(".o").replaceChildren(s);
    }
  }

  class Fleuron extends Base {
    static get observedAttributes() { return ["piece", "fold", "mirror", "size"]; }
    draw() {
      const size = this.num("size", 96), n = Math.max(2, Math.round(this.num("fold", 6)));
      const mirror = this.hasAttribute("mirror"), p = get(this.getAttribute("piece"), "petals");
      this.style.display = "inline-block";
      const s = svg(size, size), c = size / 2, r0 = size * 0.07, len = size / 2 - r0, w = size * 0.18;
      for (let i = 0; i < n; i++) {
        const rot = `rotate(${360 * i / n} ${c} ${c})`;
        place(s, p, c + r0, c - w / 2, len, w, false, false, rot);
        if (mirror) place(s, p, c + r0, c - w / 2, len, w, false, true, rot);
      }
      this.replaceChildren(s);
    }
  }

  // a box drawn with four strokes fitted to the element's real size, their ends crossing a little at the corners
  class Box extends Base {
    static get observedAttributes() { return ["piece", "weight", "overshoot"]; }
    constructor() {
      super();
      const r = this.attachShadow({ mode: "open" });
      r.innerHTML = `<style>:host{display:block;position:relative}
        .o{position:absolute;inset:0;pointer-events:none;overflow:visible}
        .c{position:relative;padding:var(--n-pad, 1.2em 1.4em)}</style><div class="o"></div><div class="c"><slot></slot></div>`;
    }
    draw() {
      const W = this.clientWidth, H = this.clientHeight; if (!W || !H) return;
      const w = this.num("weight", 3), over = this.num("overshoot", 0.9) * w;
      const names = byRole("lines"), want = this.getAttribute("piece");
      const pick = i => PIECES[want] || PIECES[names[i % names.length]] || PIECES[keys[0]];
      const s = svg(W, H), h = w * 1.2;
      // each side: the stroke laid along it (its own taper kept), stretched to the side's length plus the overshoot
      const side = (p, x0, y0, x1, y1, i) => {
        const L = Math.hypot(x1 - x0, y1 - y0) + 2 * over, a = Math.atan2(y1 - y0, x1 - x0) * 180 / Math.PI;
        const g = document.createElementNS(NS, "g");
        g.setAttribute("transform", `translate(${x0} ${y0}) rotate(${a}) translate(${-over} ${-h / 2})`);
        place(g, p, 0, 0, L, h, i % 2 === 1, i >= 2);
        s.appendChild(g);
      };
      const m = w / 2;
      side(pick(0), m, m, W - m, m, 0); side(pick(1), W - m, m, W - m, H - m, 1);
      side(pick(2), W - m, H - m, m, H - m, 2); side(pick(3), m, H - m, m, m, 3);
      this.shadowRoot.querySelector(".o").replaceChildren(s);
    }
  }

  // a motif turned round a centre, in any of its three forms (the emblem classes from neuron.css do the drawing)
  class Medallion extends HTMLElement {
    static get observedAttributes() { return ["motif", "form", "fold", "size", "mirror"]; }
    connectedCallback() { this.draw(); }
    attributeChangedCallback() { this.isConnected && this.draw(); }
    draw() {
      const id = this.getAttribute("motif") || Object.keys(MOTIFS)[0], m = MOTIFS[id];
      if (!m) return;
      const form = this.getAttribute("form") || (m.filled ? "filled" : "silhouette");
      const n = Math.max(2, Math.round(parseFloat(this.getAttribute("fold")) || (m.aspect >= 2.6 ? 8 : m.aspect >= 1.9 ? 6 : 4)));
      const D = parseFloat(this.getAttribute("size")) || 480, r0 = D * 0.06, s = D / 2 - r0;   // each copy spans r0..D/2
      const cls = "n-emblem-" + id + (form === "silhouette" ? "" : "-" + form);
      Object.assign(this.style, { display: "inline-block", position: "relative", width: D + "px", height: D + "px" });
      const mirror = this.hasAttribute("mirror");
      this.replaceChildren(...Array.from({ length: n }, (_, i) => {
        const a = 360 * i / n, rad = a * Math.PI / 180, cx = D / 2 + (r0 + s / 2) * Math.cos(rad), cy = D / 2 + (r0 + s / 2) * Math.sin(rad);
        const d = document.createElement("div");
        d.className = cls;
        // the copy is turned so its long axis lies on the radius with the heavy end inward
        d.style.cssText = `position:absolute;left:${cx - s / 2}px;top:${cy - s / 2}px;width:${s}px;height:${s}px;transform:rotate(${m.axis + a}deg)${mirror && i % 2 ? " scaleY(-1)" : ""}`;
        return d;
      }));
    }
  }

  class Palette extends HTMLElement {
    static get observedAttributes() { return ["colorway"]; }
    connectedCallback() { this.apply(); }
    attributeChangedCallback() { this.apply(); }
    apply() {
      [...this.classList].filter(c => c.startsWith("n-cw-")).forEach(c => this.classList.remove(c));
      const cw = this.getAttribute("colorway");
      if (cw) this.classList.add("n-cw-" + cw);
      if (!this.style.display) this.style.display = "block";
    }
  }

  const def = (n, c) => customElements.get(n) || customElements.define(n, c);
  def("neuron-divider", Divider);
  def("neuron-frieze", Frieze);
  def("neuron-frame", Frame);
  def("neuron-fleuron", Fleuron);
  def("neuron-box", Box);
  def("neuron-medallion", Medallion);
  def("neuron-palette", Palette);
  window.NeuronElements = { pieces: () => Object.fromEntries(keys.map(k => [k, PIECES[k].roles])), byRole, motifs: () => Object.keys(MOTIFS) };
})();
