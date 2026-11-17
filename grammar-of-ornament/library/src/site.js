/* Neuron Library site: one page, a tab per hash. Everything listed comes from catalog.js (window.NEURON). */
(() => {
  const N = window.NEURON;
  const view = document.getElementById("view");
  const pick = document.getElementById("cw");
  const dk = document.getElementById("dk");
  try { dk.checked = localStorage.getItem("neuron-dark") === "1"; } catch (e) {}
  dk.addEventListener("change", () => { try { localStorage.setItem("neuron-dark", dk.checked ? "1" : "0"); } catch (e) {} render(); });
  const CW = N.colorways;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  document.getElementById("built").textContent = N.built;

  // the colorway a card is shown in: the chosen one, or spread over all of them
  const cwFor = i => pick.value || CW[(i * 37 + 11) % CW.length].short;
  const cwName = s => CW.find(c => c.short === s);
  for (const c of CW) pick.insertAdjacentHTML("beforeend", `<option value="${c.short}">${c.short} · ${c.base_name} · ${c.name_a} / ${c.name_b}</option>`);
  try { pick.value = localStorage.getItem("neuron-cw") || ""; } catch (e) {}
  pick.addEventListener("change", () => { try { localStorage.setItem("neuron-cw", pick.value); } catch (e) {} render(); });

  const ink = (src, extra = "") => `<div class="ink" style="--src:url('${src}');${extra}"></div>`;
  const tile = (i, inner, ar = 1, dark = false) =>
    `<div class="tile n-cw-${cwFor(i)}${dark || dk.checked ? " n-dark" : ""}" style="--ar:${ar}" title="colorway ${cwFor(i)}">${inner}</div>`;
  const copyable = t => `<code class="copy" title="click to copy">${esc(t)}</code>`;
  // a piece's source, in words: "family-05-short-wave-branch-fine/stroke-06" → "single stroke short-wave-branch-fine 06"
  const from = s => s.replace(/(^|\s)r-/g, "$1resnet50 ").replace(/(^|\s)c-/g, "$1convnext ")
    .replace(/family-\d+-([\w-]+)\/stroke-(\d+)/g, "single stroke $1 $2").replace(/([a-z][\w-]*)\/(\d+)/g, "compound $1 $2");

  // ------------------------------------------------------------ tabs
  const tabs = {
    font() {
      const AZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?'\"-–—/()&";
      const panels = [0, 1, 2, 3].map(i => {
        const c = cwFor(i + 3);
        return `<div class="panel n-cw-${c} ${i % 2 ? "n-dark" : "n-page"}">
          <div class="n-display" style="font-size:64px;color:${i % 2 ? "var(--n-on-dark)" : "var(--n-accent)"}">${["Sapphire Moon", "Plate 119", "Ornament", "Made of itself"][i]}</div>
          <div class="cap" style="color:inherit;opacity:.75">Neuron Leaf · colorway ${c}</div></div>`;
      }).join("");
      return `<h1>Neuron Leaf</h1>
        <p class="lede">A display face drawn with a leaf-shaped pen, swelling in the middle and fine at the ends, the way the
        networks' own strokes do, with long ascenders and descenders. Use it for titles at 32px and up, with the text face below for everything else.</p>
        <p><a class="dl" href="neuron-library/fonts/Neuron-Leaf.woff2" download>Neuron-Leaf.woff2</a>
        <a class="dl" href="neuron-library/fonts/Neuron-Leaf.otf" download>.otf</a></p>
        <div class="spec">
          <p class="n-display" style="font-size:132px">Grammar of Ornament</p>
          <p class="n-display" style="font-size:72px">The Quick Brown Fox Jumps</p>
          <p class="n-display" style="font-size:44px">hamburgefonstiv · SAPPHIRE MOON 1984</p>
          <p class="n-display" style="font-size:28px">A face for titles, numbers and short headings.</p>
        </div>
        <h2>Text face</h2>
        <p class="lede">Body text is set in <b>Alegreya Sans</b>, a humanist sans serif built on the same broad-pen logic as
        Neuron Leaf's tapering strokes. It is bundled, with its small caps (<code>n-label</code>, for labels, navigation and
        metadata) and DM Mono for anything that is literally code. <code>font-family: var(--n-text)</code> or class <code>n-text</code>.</p>
        <div class="panels">${[0, 1].map(i => `<div class="panel ${i ? "n-dark" : "n-page"} n-cw-${cwFor(i + 9)} n-text">
            <div><div class="n-display" style="font-size:44px;color:var(--n-accent)">Twenty-two cuts</div>
            <p style="font-size:18px;margin:8px 0">A network trained on photographs has a channel that responds to an arch: ribs
            that spring from a base, bend over and come down again. <b>Ask it to draw what it looks for</b>, and <i>this</i> is what it draws.</p>
            <p class="n-label" style="margin:0;opacity:.8">Alegreya Sans SC · labels, navigation, metadata · 1984</p></div>
            <div class="cap" style="color:inherit">Alegreya Sans 400, 400 italic, 500, 700 · small caps 400, 500, 700${i ? " · n-dark" : ""}</div></div>`).join("")}</div>
        <h2>Light and edges</h2>
        <p class="lede">The features glow, and at every edge their colours come in a fixed order: a dark outline, the body
        colour, a bright halo, its opponent. <code>n-glow</code> gives type (or, as <code>n-glow-box</code>, an emblem) a
        halo in the colorway's highlight; <code>n-fringe</code> splits an edge into the opponent pair. <code>n-outline</code>
        fills type with the light end of the feature's measured edge sequence and outlines it outward through the sequence (<code>--n-outline</code> sets the rings' width);
        <code>n-fringe-edge</code> is the fringe in the sequence's colours; <code>n-outline-in</code> is the outline the
        other way round, for paper. All but the last are made for dark grounds: <code>n-dark</code> on any container swaps the whole palette.</p>
        <div class="panels">${[0, 1, 2, 3, 4, 5].map(i => {
          const c = cwFor(i + 20), cls = ["n-glow", "n-fringe", "n-fringe n-glow", "n-outline", "n-fringe-edge", "n-outline-in"][i];
          return `<div class="panel n-cw-${c} ${i === 5 ? "n-page" : "n-dark"}">
            <div class="n-display ${cls}" style="font-size:64px;${i === 0 ? "color:var(--n-ink)" : i === 1 || i === 2 || i === 4 ? "color:var(--n-accent)" : ""}">${["Lantern", "Opponent", "Both", "Outline", "Sequence", "On paper"][i]}</div>
            <div class="cap" style="color:inherit;opacity:.75">${cls} · ${i === 5 ? "n-page" : "n-dark"} · colorway ${c}</div></div>`;
        }).join("")}</div>
        <h2>Character set</h2><div class="glyphs">${[...AZ].map(c => `<span>${esc(c)}</span>`).join("")}</div>
        <h2>In the colorways</h2><div class="panels">${panels}</div>
        <h2>CSS</h2><p>${copyable('<h1 class="n-display">…</h1>')} or
        ${copyable("font-family: 'Neuron Leaf'")}</p>`;
    },

    features() {
      const models = Object.keys(N.features);
      const sel = state.model || "all";
      const items = models.filter(m => sel === "all" || m === sel).flatMap(m => N.features[m]);
      return `<h1>Features</h1>
        <p class="lede">What one channel of a trained image network responds to, drawn by optimizing an image until that
        channel fires as strongly as it can. ${models.map(m => `${N.features[m].length} from ${m}`).join(", ")}. Use them
        large, as surfaces and covers, one feature to a composition. Each was made at 224 px and upscaled twice, to
        1792 px (click one to see it) and 3584 px (the download), so it holds as a page background. The ConvNeXt
        features also come as seamless tiles (on the Colorways tab), for backgrounds that repeat.</p>
        <div class="filters">${["all", ...models].map(m => `<button data-model="${m}" class="${m === sel ? "on" : ""}">${m}</button>`).join("")}</div>
        <div class="grid" style="--min:170px">${items.map(f => `<figure class="feat">
          <img src="${f.src}" loading="lazy" alt="${f.model} ${f.layer} channel ${f.channel}" data-big="${f.big || f.src}">
          <figcaption>${f.model} · ${f.layer} · c${f.channel}${f.huge ? ` · <a href="${f.huge}" download>3584 px</a>` : f.big ? ` · <a href="${f.big}" download>big</a>` : ""}</figcaption></figure>`).join("")}</div>`;
    },

    linework() {
      const L = N.linework;
      const models = [...new Set(L.map(l => l.model))];
      const sel = state.model || "all";
      const items = L.filter(l => sel === "all" || l.model === sel);
      const hero = L[(7 * 37 + 11) % L.length], tiled = L[(23 * 37 + 11) % L.length];
      return `<h1>Linework</h1>
        <p class="lede">Each feature's print taken apart into its strokes and drawn again, every stroke at its measured
        width, the masses filled and the relief marks carved: the line of the feature without its colour. ${models.map(m => `${L.filter(l => l.model === m).length} from ${m}`).join(", ")}.
        Each is one SVG, a mask painted in <code>currentColor</code>, so it takes any colour or colorway: meant large, as a
        background, a header band or a panel of pattern, one to a composition. Put <code>n-lw-&lt;id&gt;</code> on a
        sized element; it covers it by default, or tiles at <code>--n-lw-size</code> with <code>--n-lw-repeat: repeat</code>.
        Below, each is shown in a colorway chosen for it (the picker above changes that); the colour is the page's, not the drawing's.
        The drawings are large files, so they come as their own download, to unpack into the library folder.</p>
        <p><a class="dl" href="neuron-linework.zip" download>neuron-linework.zip</a></p>
        <div class="lwhero n-cw-${cwFor(7)}${dk.checked ? " n-dark" : ""}"><div class="${hero.css}" style="color:var(--n-accent)"></div>
          <div class="cap">as a header band · ${copyable(hero.css)} · colorway ${cwFor(7)}</div></div>
        <div class="lwhero n-cw-${cwFor(23)} n-dark" style="--h:150px"><div class="${tiled.css}" style="color:var(--n-accent2);--n-lw-size:25%;--n-lw-repeat:repeat"></div>
          <div class="cap">tiled at a quarter width · ${copyable(tiled.css)} · colorway ${cwFor(23)} · dark side</div></div>
        <div class="filters">${["all", ...models].map(m => `<button data-model="${m}" class="${m === sel ? "on" : ""}">${m}</button>`).join("")}</div>
        <div class="grid" style="--min:300px">${items.map((l, i) => {
          const c = cwFor(i + 2), dark = dk.checked || i % 3 === 1;
          return `<figure class="lw n-cw-${c}${dark ? " n-dark" : ""}">
            <div class="pat"><div style="--src:url('${l.preview}');color:var(--n-accent)"></div>${l.feature ? `<img src="${l.feature}" alt="" loading="lazy">` : ""}</div>
            <figcaption>${l.model} · ${l.layer} · c${l.channel} · ${copyable(l.css)} · <a href="${l.src}" download>svg</a><br>
              <span style="opacity:.7">colorway ${c}${dark ? " · dark side" : ""} · ink ${Math.round(l.ink * 100)}%${l.masses ? ` · ${l.masses} masses` : ""}</span></figcaption></figure>`;
        }).join("")}</div>`;
    },

    strokes() {
      let i = 0;
      const all = Object.keys(N.strokes), M = N.stroke_models || {};
      const models = [...new Set(all.map(f => M[f] || "vgg16"))];
      const sel = state.model && models.includes(state.model) ? state.model : models[0];
      const fams = all.filter(f => (M[f] || "vgg16") === sel).sort();
      const bare = f => f.replace(/^[rc]-/, "");
      return `<h1>Strokes</h1>
        <p class="lede">The compound strokes: families of marks (combs, feathers, sprigs, bands, branching lines,
        hatching, meshes, loops, forks) cut from the drawings of the features of three networks, ${all.reduce((a, f) => a + N.strokes[f].length, 0)}
        in all: ${models.map(m => `${all.filter(f => (M[f] || "vgg16") === m).reduce((a, f) => a + N.strokes[f].length, 0)} from ${m}`).join(", ")}.
        Each is in the library as a clean, recolourable SVG: <code>elements/strokes/&lt;family&gt;-&lt;n&gt;.svg</code>, where the
        family carries <code>r-</code> for resnet50 and <code>c-</code> for convnext.</p>
        <div class="filters">${models.map(m => `<button data-model="${m}" class="${m === sel ? "on" : ""}">${m}</button>`).join("")}</div>
        <div class="filters">${fams.map(f => `<button data-jump="fam-${f}">${bare(f)}</button>`).join("")}</div>
        ${fams.map(f => `<h2 id="fam-${f}">${bare(f)} <span style="font-size:14px;opacity:.6">${sel}</span></h2><div class="grid" style="--min:120px">${N.strokes[f].map(n => {
          const src = `neuron-library/elements/strokes/${f}-${n}.svg`;
          return `<figure>${tile(i++, ink(src))}<figcaption><a href="${src}" download>${f}/${n}</a></figcaption></figure>`;
        }).join("")}</div>`).join("")}`;
    },

    motifs() {
      const mode = state.motif || "filled";
      const btn = (v, l) => `<button data-motif="${v}" class="${mode === v ? "on" : ""}">${l}</button>`;
      const show = m => mode === "silhouette" ? ink(m.svg)
        : mode === "filled" ? `<div class="fill" style="background:url('${m.filled}') center/contain no-repeat"></div>`
        : `<div class="fill n-emblem-${m.id}-duo"></div>`;
      return `<h1>Motifs</h1>
        <div class="filters">${btn("filled", "filled: the print's own colours")}${btn("duo", "duotone")}${btn("silhouette", "silhouette")}</div>
        <p class="lede">${N.motifs.length} motifs cut by hand from vgg16 features: each feature upscaled, its detail restored,
        thresholded to pure black and white, then lassoed. In the library each is an emblem in three forms: a silhouette
        (one recolourable path), <b>filled</b> with the interior of its own print (the feature's colours inside the cut),
        and a <b>duotone</b> of that interior in any two colours (the colorway's shadow and highlight by default). For
        heroes, covers and big panels, not small UI. Hover a card to see the motif as cut.</p>
        <div class="grid" style="--min:220px">${N.motifs.map((m, i) => `<figure class="motif">
          <div style="position:relative">${tile(i, show(m))}
          <img src="${m.png}" alt="" loading="lazy" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#fff;opacity:0;transition:opacity .2s"
            onmouseenter="this.style.opacity=1" onmouseleave="this.style.opacity=0"></div>
          <figcaption>${copyable("n-emblem-" + m.id + (mode === "silhouette" ? "" : "-" + (mode === "duo" ? "duo" : "filled")))} · <a href="${m.svg}" download>svg</a> · <a href="${m.filled}" download>filled</a> · <a href="${m.png}" download>png</a></figcaption></figure>`).join("")}</div>`;
    },

    friezes() {
      const F = N.friezes, f = state.fz || (state.fz = { kind: "all", arr: "all", made: false });
      const arrs = [...new Set(F.map(z => z.arrangement))].sort();
      const items = F.filter(z => (f.kind === "all" || z.kind === f.kind) && (f.arr === "all" || z.arrangement === f.arr));
      const btn = (key, val, label) => `<button data-fz="${key}:${val}" class="${String(f[key]) === String(val) ? "on" : ""}">${label}</button>`;
      return `<h1>Friezes</h1>
        <p class="lede">${F.length} bands. Each repeats one unit along a band, taking the next member of
        a family at each step, the way the networks' own repeats are imperfect. Units are compound strokes (with
        their nearest neighbour of the same type) or a motif with its siblings from the same print. They come in eight
        arrangements: shift, mirror, glide, turn, stack, stack2, drift and pulse. Some have rails, or a counterband
        of tiny copies. In the library each frieze is recolourable: the ground is dropped and the ink is
        <code>currentColor</code>. Use <code>class="n-frieze-f000"</code> for a band that scales to its width.</p>
        <div class="filters">${btn("made", false, "In colorways")}${btn("made", true, "As made")}
          <span style="width:16px"></span>${btn("kind", "all", "all units")}${btn("kind", "stroke", "strokes")}${btn("kind", "motif", "motifs")}
          <span style="width:16px"></span>${btn("arr", "all", "every arrangement")}${arrs.map(a => btn("arr", a, a)).join("")}</div>
        <div class="grid" style="--min:520px">${items.map((z, i) => {
          const inner = f.made ? `<img src="${z.orig}" alt="" loading="lazy" style="width:100%;display:block">`
            : `<div class="tile n-cw-${cwFor(i)}${z.reverse ? " dark" : ""}" style="--ar:auto;padding:12px 10px">
                 <div class="ink" style="--src:url('neuron-library/${z.file}');position:static;width:100%;aspect-ratio:1600/140;-webkit-mask-size:100% 100%;mask-size:100% 100%"></div></div>`;
          return `<figure>${inner}<figcaption>${copyable("n-frieze-" + z.id)} · <a href="neuron-library/${z.file}" download>svg</a>
            · <a href="${z.orig}" download>as made</a><br>${esc(z.unit)} · ${z.arrangement}${z.rails !== "none" ? " · rails: " + z.rails : ""}${z.two ? " · two inks" : ""}</figcaption></figure>`;
        }).join("")}</div>`;
    },

    colorways() {
      return `<h1>Colorways</h1>
        <p class="lede">${CW.length} palettes, one for each ConvNeXt feature, in the feature's own colours. The <b>base</b> is the
        biggest patch of colour in the picture; the <b>five</b> (<code>--n-c1</code> … <code>--n-c5</code>) are five more
        colours present in it, one per patch, as far apart as the picture allows and each chosen to read against the base.
        The base is the page: paper when it is light, the dark ground when it is dark (the other side is a tint of it).
        When a base is mid-toned and the picture has nothing light or dark, the ground is a tint of the base and the
        base is kept as a panel colour (<code>--n-base</code>, the square on those cards). <code>--n-accent</code> and
        <code>--n-accent2</code> are the most saturated of the five and the one farthest from it, the picture's own
        opposition; each card is named for its base and those two. Put
        <code>n-cw-&lt;id&gt;</code> on any element to use one, and <code>n-dark</code> inside it for the dark side. Each card
        also carries the feature's <b>edge sequence</b>: the colours met crossing its edges, dark side to light side.
        It is the strip above the ramp, and the outlined Aa on the dark row (<code>n-outline</code>).</p>
        <div class="grid" style="--min:250px">${CW.map(c => `<figure class="cw n-cw-${c.short}">
          <div class="pic">${c.feature ? `<img src="${c.feature}" alt="" loading="lazy" data-big="${c.big || c.feature}">` : ""}</div>
          <div class="sw"><i style="background:${c.base};flex:1.6" title="base · ${c.base_name} ${c.base}"></i>${c.five.map((h, i) => `<i style="background:${h}" title="${c.names[i]} ${h}"></i>`).join("")}</div>
          <div class="body ${c.ground === c.dark ? "n-dark" : ""}"><b>Aa</b> ${[1, 2, 3, 4, 5].map(i => `<b style="font-size:22px;color:var(--n-c${i})">Aa</b>`).join(" ")} <span title="base · accent / accent2" style="display:block;margin-top:4px">${esc(c.base_name)} · ${esc(c.name_a)} / ${esc(c.name_b)}${c.ground !== c.base ? ` · <i class="n-base" style="display:inline-block;width:1em;height:1em;vertical-align:-.15em" title="the base, as a panel colour"></i> base` : ""}</span>
          <div class="cap" style="color:inherit">${copyable("n-cw-" + c.short)} · ${c.layer}${c.tile ? ` · <a href="${c.tile}" download>tile</a>` : ""}</div></div>
          <div class="body ${c.ground === c.dark ? "" : "n-dark"}"><b class="n-glow" style="color:var(--n-accent)">Aa</b> <b style="color:var(--n-accent2);font-size:22px">Aa</b> ${c.edge ? '<b class="n-outline" style="font-size:40px">Aa</b>' : ""} <span style="font-size:13px">${c.ground === c.dark ? "n-page" : "n-dark"} · ${c.ground === c.dark ? "light" : "dark"} tint</span></div>
          ${c.edge ? '<div class="n-edge-line" style="--n-h:8px;--n-gap:0"></div>' : ""}<div class="n-ramp" style="height:8px"></div></figure>`).join("")}</div>`;
    },

    elements() {
      const E = N.elements;
      const group = (kind, title, note, opts = {}) => {
        const items = E.filter(e => e.kind === kind);
        if (!items.length) return "";
        return `<h2>${title}</h2><p class="lede">${note}</p><div class="grid" style="--min:${opts.min || 200}px">${items.map((e, i) => {
          let inner;
          if (kind === "tiles") inner = `<div class="ink" style="--src:url('neuron-library/${e.file}');-webkit-mask:var(--src) left center/auto ${e.style === "strip" ? 44 : 22}px repeat-x;mask:var(--src) left center/auto ${e.style === "strip" ? 44 : 22}px repeat-x;height:60px"></div>`;
          else if (kind === "frames") inner = `<div class="${e.css}" style="--n-size:58px;color:inherit;min-height:190px;width:100%;display:flex;align-items:center;justify-content:center"><span class="n-display" style="font-size:30px;color:var(--n-ink)">Framed</span></div>`;
          else if (kind === "lines") inner = `<div style="position:absolute;inset:14px;display:flex;align-items:center;gap:18px"><div class="${e.css}" style="--n-h:6px;--n-gap:0;flex:1"></div><div class="n-line-v-${e.n}" style="--n-h:6px;--n-len:100%;height:100%"></div></div>`;
          else inner = ink("neuron-library/" + e.file);
          const ar = kind === "dividers" || kind === "lines" ? 3.2 : kind === "icons" ? 1 : kind === "tiles" ? 4 : kind === "frames" ? 1.4 : 1;
          return `<figure class="el ${opts.wide ? "wide" : ""}">${tile(i + kind.length * 5, inner, ar)}
            <figcaption>${copyable(e.css || e.id)}${kind === "lines" ? " · " + copyable("n-line-v-" + e.n) : ""} · <a href="neuron-library/${e.file}" download>svg</a><br>from ${esc(from(e.source))}</figcaption></figure>`;
        }).join("")}</div>`;
      };
      const c = cwFor(1), pieces = N.pieces, byRole = r => Object.keys(pieces).filter(k => pieces[k].roles.includes(r));
      const [r0, r1] = byRole("rules"), [c0] = byRole("compact"), [k0] = byRole("curved"), [p0, p1] = byRole("petals"), [t0] = byRole("tiles"), [l0] = byRole("lines");
      const med = N.motifs.filter(m => m.axis !== undefined && m.filled).sort((a, b) => b.aspect - a.aspect);
      return `<h1>Elements</h1>
        <p class="lede">Ornaments for web pages, drawn from the compound strokes, the single strokes they are built from, and the
        motifs, and chosen automatically by shape (long pieces for rules, curved ones for corners, compact branching ones
        for marks, long single strokes for lines and the edges of frames and boxes). Each
        is one SVG path in <code>currentColor</code>, shown here in the colorways. Click a class name to copy it.</p>
        <h2>Adaptive (neuron-elements.js)</h2>
        <p class="lede">The same pieces as web components that fit themselves to the space: try resizing the window.</p>
        <div class="demo n-cw-${c}">
          <neuron-divider piece="${r0}" mode="repeat" height="22"></neuron-divider>
          <p><code>&lt;neuron-divider piece="${r0}" mode="repeat"&gt;</code></p>
          <neuron-divider piece="${r1}" mode="mirror" centre="${c0}" height="34"></neuron-divider>
          <p><code>&lt;neuron-divider piece="${r1}" mode="mirror" centre="${c0}"&gt;</code></p>
          <neuron-frieze piece="${t0}" height="40"></neuron-frieze>
          <p><code>&lt;neuron-frieze piece="${t0}"&gt;</code></p>
          <div style="display:grid;grid-template-columns:2fr 1fr;gap:24px;align-items:center;margin-top:20px">
            <neuron-frame piece="${k0}" edge="${l0}" size="64"><p class="n-display" style="font-size:40px;color:var(--n-ink);margin:0">A frame that fits its box</p>
              <p>Corners sit at the real corners however the box's proportions change, and each side is one stroke stretched between them. <code>&lt;neuron-frame piece="${k0}" edge="${l0}"&gt;</code></p></neuron-frame>
            <div style="text-align:center"><neuron-fleuron piece="${p0}" fold="8" mirror size="150"></neuron-fleuron>
              <neuron-fleuron piece="${p1}" fold="5" size="110"></neuron-fleuron>
              <p><code>&lt;neuron-fleuron piece="${p0}" fold="8" mirror&gt;</code></p></div>
          </div>
        </div>
        ${group("dividers", "Dividers", "Rules (<code>rule-</code>): one long compound laid flat. Mirrored (<code>mirror-</code>): two copies meeting at a small piece from the same family.", { min: 380, wide: false })}
        ${group("tiles", "Repeats", "Small seamless tiles that repeat sideways to any length: compounds (<code>strip-</code>) and single strokes in glide reflection (<code>dash-</code>). For whole bands see the <a href=\"#friezes\">Friezes</a> tab.", { min: 380 })}
        ${group("lines", "Lines", "A single stroke stretched to the length of a rule, its taper kept, instead of a flat CSS border: <code>n-line-k</code> runs across (<code>--n-h</code> is its weight), <code>n-line-v-k</code> runs up (<code>--n-len</code> its length).", { min: 380 })}
        ${group("corners", "Corners", "A curved compound along one edge and mirrored down the other. Each comes in four turns (<code>-tl -tr -bl -br</code>).", { min: 170 })}
        ${group("frames", "Frames", "Four corners joined by a single stroke along each side, stretched between them. As a class, the frame fits any box.", { min: 260 })}
        <h2>Drawn boxes</h2>
        <p class="lede">Panels that are nearly rectangles: each edge wavers the way a single stroke does, and the corners
        sit a little off true, as in a box drawn by hand. <code>n-box-k</code> is a filled panel with that edge;
        <code>n-drawn-k</code> draws it as a border (wrap the content in one child). Any size or proportion.</p>
        <div class="grid" style="--min:220px">${N.boxes.map((b, i) => `<figure>
          ${tile(i + 80, `<div class="fill n-box-${b.id.slice(4)}" style="background:var(--n-accent);color:var(--n-paper);inset:12px;display:flex;align-items:center;justify-content:center"><span class="n-display" style="font-size:30px">Aa</span></div>`, 1.5)}
          <figcaption>${copyable("n-box-" + b.id.slice(4))} · ${copyable("n-drawn-" + b.id.slice(4))}<br>edges from ${esc(b.strokes.map(s => s.split("/")[0].replace("family-", "")).join(", "))}</figcaption></figure>`).join("")}</div>
        <div class="demo n-cw-${cwFor(3)}" style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
          <div class="n-drawn-2" style="color:var(--n-accent);--n-line:3px"><div><p class="n-display" style="font-size:34px;margin:0 0 6px">A drawn border</p><p style="margin:0">Two nested boxes clipped to the same drawn edge, the outer in the line colour: <code>&lt;div class="n-drawn-2"&gt;&lt;div&gt;…&lt;/div&gt;&lt;/div&gt;</code></p></div></div>
          <neuron-box piece="${Object.keys(N.pieces).find(k => N.pieces[k].roles.includes("lines")) || ""}" weight="4" style="color:var(--n-accent2)"><p class="n-display" style="font-size:34px;margin:0 0 6px">Four strokes</p><p style="margin:0">The adaptive version draws four tapered single strokes to the box's real size, their ends crossing at the corners: <code>&lt;neuron-box&gt;</code></p></neuron-box>
        </div>
        ${group("marks", "Marks", "Compact, branching compounds for bullets and section marks.", { min: 130 })}
        ${group("fleurons", "Fleurons", "One piece turned round a centre, 4- to 8-fold, some mirrored.", { min: 150 })}
        ${group("medallions", "Medallions", "A whole motif turned round a centre, its long axis on the radius: rosettes for large spots (300px and up), never small UI. The elongated motifs are chosen automatically.", { min: 220 })}
        <p class="lede">As a web component the medallion is built from any motif in any of its forms, including the motif filled with its own print:</p>
        <div class="demo n-cw-${cwFor(5)} n-dark" style="display:flex;flex-wrap:wrap;gap:30px;align-items:center;justify-content:center">
          ${med.slice(0, 3).map((m, i) => `<div style="text-align:center"><neuron-medallion motif="${m.id}" form="${["filled", "duo", "silhouette"][i]}" size="300" ${i === 1 ? "mirror" : ""} class="${i === 0 ? "n-glow-box" : ""}" style="color:var(--n-accent)"></neuron-medallion>
            <p style="margin:6px 0 0"><code>&lt;neuron-medallion motif="${m.id}" form="${["filled", "duo", "silhouette"][i]}"${i === 1 ? " mirror" : ""}&gt;</code></p></div>`).join("")}
        </div>
        ${group("icons", "Icons", "Interface icons drawn with single strokes, in two sets. <code>&lt;span class=\"n-icon-menu\"&gt;</code>, or from sprite.svg.", { min: 90 })}
        <h2>Emblems</h2><p class="lede">The ${N.motifs.length} motifs: see the <a href="#motifs">Motifs</a> tab.</p>`;
    },

    examples() {
      const ex = [
        ["plate.html", "Plate", "An article page laid out like a plate in a pattern book: one motif as the hero, its details magnified at the side, two columns of text with rules and mark bullets."],
        ["emblem.html", "Emblem", "An event page, everything centred on a dark ground: a fleuron, a large glowing number, a framed programme."],
        ["ground.html", "Ground", "A gallery page with one ConvNeXt feature, upscaled and made seamless, as the page's background, and that feature's colorway for everything on it."],
        ["lantern.html", "Lantern", "A dark page: a filled emblem glowing, fringed display type, duotones and ramps."],
        ["invaded.html", "Invaded", "A portfolio in black and white halves, giant type broken into by a motif, one feature as the only image."],
        ["specimen.html", "Specimen", "A type specimen on a dark ground: every letter outlined through the feature's edge sequence, the sequence as rules, bands and a fringe."],
        ["made.html", "Made of itself", "A motif drawn out of hundreds of copies of one small mark, and the same motif turned into a medallion."],
        ["bands.html", "Bands", "Friezes as the page's structure: full-width bands of one family in six arrangements, the type in the gutters, the title and number running off the edges."],
        ["column.html", "Column", "A feature as a full-height column down the left, the title in vertical type beside it, the text narrow at the right. Nothing centred."],
        ["dossier.html", "Dossier", "One filled emblem in the middle of a sheet and everything written round it: a big date, codes, numbered markers with leader lines, a magnified detail in a drawn box."],
      ];
      return `<h1>Examples</h1>
        <p class="lede">Web pages made from the library: the font, one colorway each, a single feature or motif family, and the
        elements. Each page is plain HTML and CSS, so view source to see how it's put together.</p>
        <div class="grid ex" style="--min:300px">${ex.map(([f, t, d]) => `<figure><a href="examples/${f}">
          <img src="examples/${f.replace(".html", ".png")}" alt="${t}" loading="lazy" onerror="this.style.visibility='hidden'">
          <h3>${t}</h3></a><figcaption>${d}</figcaption></figure>`).join("")}</div>`;
    },

    readme() {
      return `<a class="dl" href="neuron-library.zip" download>Download neuron-library.zip</a>
        <article class="readme">${md(N.readme)}</article>`;
    },
  };
  const state = {};

  // ------------------------------------------------------------ markdown (enough for the README)
  function md(src) {
    const out = [];
    const inline = s => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/\*([^*]+)\*/g, "<i>$1</i>").replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
    const lines = src.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i];
      if (l.startsWith("```")) {
        const buf = [];
        while (++i < lines.length && !lines[i].startsWith("```")) buf.push(lines[i]);
        out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
      } else if (/^#{1,3} /.test(l)) {
        const n = l.match(/^#+/)[0].length;
        out.push(`<h${n}>${inline(l.slice(n + 1))}</h${n}>`);
      } else if (l.startsWith("|")) {
        const rows = [];
        while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
        i--;
        out.push("<table>" + rows.filter(r => !/^\|[-| ]+\|$/.test(r)).map((r, k) => "<tr>" + r.slice(1, -1).split("|")
          .map(c => `<${k ? "td" : "th"}>${inline(c.trim())}</${k ? "td" : "th"}>`).join("") + "</tr>").join("") + "</table>");
      } else if (l.startsWith("- ")) {
        const items = [];
        while (i < lines.length && (lines[i].startsWith("- ") || lines[i].startsWith("  "))) {
          if (lines[i].startsWith("- ")) items.push(lines[i].slice(2)); else items[items.length - 1] += " " + lines[i].trim();
          i++;
        }
        i--;
        out.push("<ul>" + items.map(t => `<li>${inline(t)}</li>`).join("") + "</ul>");
      } else if (l.trim()) {
        const buf = [l];
        while (i + 1 < lines.length && lines[i + 1].trim() && !/^(#|```|- |\|)/.test(lines[i + 1])) buf.push(lines[++i]);
        out.push(`<p>${inline(buf.join(" "))}</p>`);
      }
    }
    return out.join("\n");
  }

  // ------------------------------------------------------------ routing and events
  function render() {
    const tab = (location.hash.slice(1).split("/")[0]) || "font";
    const fn = tabs[tab] || tabs.font;
    view.innerHTML = fn();
    document.querySelectorAll("#tabs a").forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + tab));
  }
  window.addEventListener("hashchange", () => { render(); scrollTo(0, 0); });
  view.addEventListener("click", ev => {
    const t = ev.target;
    if (t.dataset.model) { state.model = t.dataset.model; render(); }
    else if (t.dataset.motif) { state.motif = t.dataset.motif; render(); }
    else if (t.dataset.fz) { const [k, v] = t.dataset.fz.split(":"); state.fz[k] = v === "true" ? true : v === "false" ? false : v; render(); }
    else if (t.dataset.jump) document.getElementById(t.dataset.jump)?.scrollIntoView({ behavior: "smooth" });
    else if (t.dataset.big) {
      const lb = document.createElement("div");
      lb.className = "lightbox";
      lb.innerHTML = `<img src="${t.dataset.big}"><p>${esc(t.alt || t.dataset.big)}</p>`;
      lb.onclick = () => lb.remove();
      document.body.appendChild(lb);
    } else if (t.classList.contains("copy")) {
      navigator.clipboard?.writeText(t.textContent);
      const was = t.textContent;
      t.textContent = "copied";
      setTimeout(() => (t.textContent = was), 700);
    }
  });
  render();
})();
