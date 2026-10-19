/* Grammar of Ornament — the View panel on the source tiles.
   Colorway: recolours the page (inks, ground, accent and every ornament) from
   one convnext feature's opponent pair, fetched from the server.
   Background: puts a picture behind the whole page, or behind one or more
   full-width bands (the hero, the cards, or every other section). Text and
   controls keep a solid panel of the page's ground, so words never sit on
   the picture.
   Picture: the feature in its own colours (the colorway's feature, or the
   page's); the feature remapped to the page's colours; or a linework drawing in
   the page's colours. The page-colour versions
   are made here, in a canvas, from whatever colorway is showing. */
(function () {
  const V = window.GOView;
  const panel = document.querySelector(".view");
  if (!V || !panel) return;
  const root = document.documentElement, body = document.body;
  const served = location.protocol.startsWith("http");
  const sel = (n) => panel.querySelector(`[name=${n}]`);
  let applied = [], token = 0;

  const css = (name) => getComputedStyle(root).getPropertyValue("--" + name).trim();
  const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const lum = (h) => { const [r, g, b] = rgb(h); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };

  function featureUrl() {
    const cw = V.options.find((o) => o.name === sel("colorway").value);
    return cw ? cw.fill : V.fill;
  }
  // map an image's luminance onto two colours: t=0 (dark) → c0, t=1 (light) → c1
  async function twoTone(src, c0, c1, stretch) {
    const img = new Image();
    img.src = src;
    await img.decode();
    const w = Math.min(1100, img.naturalWidth || 1100), h = Math.round(w * ((img.naturalHeight || w) / (img.naturalWidth || w)));
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h), px = d.data;
    let lo = 0, hi = 255;
    if (stretch) {   // stretch between the 2nd and 98th percentiles
      const hist = new Uint32Array(256);
      for (let i = 0; i < px.length; i += 4) hist[(0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) | 0]++;
      let acc = 0; const n = px.length / 4;
      for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc < n * 0.02) lo = v; if (acc < n * 0.98) hi = v; }
    }
    const a = rgb(c0), b = rgb(c1);
    for (let i = 0; i < px.length; i += 4) {
      const t = Math.min(1, Math.max(0, ((0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) - lo) / Math.max(1, hi - lo)));
      px[i] = a[0] + (b[0] - a[0]) * t; px[i + 1] = a[1] + (b[1] - a[1]) * t; px[i + 2] = a[2] + (b[2] - a[2]) * t;
    }
    ctx.putImageData(d, 0, 0);
    return cv.toDataURL("image/jpeg", 0.88);
  }
  async function picture() {
    const pic = sel("pic").value;
    const ink = css("ink"), paper = css("paper"), accent = css("accent");
    const darkPage = lum(paper) < lum(ink);
    if (pic === "feature") return featureUrl();
    if (pic === "duotone") {
      const [d, l] = lum(ink) < lum(paper) ? [ink, paper] : [paper, ink];
      return twoTone(featureUrl(), d, l, true);
    }
    // a drawing: its marks in the ink (the accent on dark pages), its ground in the page's ground
    return twoTone(sel("drawing").value, darkPage ? accent : ink, paper, false);
  }
  async function apply() {
    const my = ++token, where = sel("where").value;
    panel.querySelector(".drawsrc").hidden = sel("pic").value !== "drawing";
    body.classList.remove("bg-page");
    document.querySelectorAll(".bg-seg").forEach((e) => e.classList.remove("bg-seg"));
    if (where === "none") return;
    panel.classList.add("busy");
    try {
      const u = await picture();
      if (my !== token) return;
      root.style.setProperty("--bg-img", `url("${u}")`);
      const sections = [...document.querySelectorAll(".wrap > section")];
      if (where === "page") body.classList.add("bg-page");
      if (where === "hero") document.querySelector(".hero").classList.add("bg-seg");
      if (where === "cards") sections.filter((s) => s.querySelector(".cards")).forEach((s) => s.classList.add("bg-seg"));
      if (where === "alt") sections.forEach((s, i) => i % 2 === 1 && s.classList.add("bg-seg"));
    } catch (e) { console.error(e); }
    panel.classList.remove("busy");
  }
  async function setColorway() {
    const name = sel("colorway").value;
    applied.forEach((k) => root.style.removeProperty("--" + k));
    applied = [];
    if (name !== "default") {
      if (!served) { alert("Colorways need the local server: http://127.0.0.1:8790/designlib/tiles/"); sel("colorway").value = "default"; return; }
      panel.classList.add("busy");
      try {
        const r = await fetch("/api/designlib/palette", { method: "POST", headers: { "Content-Type": "application/json" },
                                                         body: JSON.stringify({ theme: body.dataset.theme, colorway: name }) });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || r.statusText);
        for (const [k, v] of Object.entries(j.vars)) { root.style.setProperty("--" + k, v); applied.push(k); }
      } catch (e) { alert("Could not load the colorway: " + e.message); }
      panel.classList.remove("busy");
    }
    const o = V.options.find((x) => x.name === name);
    panel.querySelector(".swatch").style.background = o ? `linear-gradient(90deg, ${o.a} 50%, ${o.b} 50%)` : "transparent";
    if (window.GOShapes) GOShapes.drawAll();
    apply();
  }
  sel("colorway").addEventListener("change", setColorway);
  ["where", "pic", "drawing"].forEach((n) => sel(n).addEventListener("change", apply));
})();
