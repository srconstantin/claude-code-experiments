/* Grammar of Ornament — style tile behaviour: working tabs and buttons, and
   for every specimen a toolbar to make a variant, save it to designlib/saved/
   (when the page is served by server.py), or download its ornament as SVG. */
(function () {
  const theme = document.body.dataset.theme;
  const served = location.protocol.startsWith("http");

  function toast(msg) {
    let t = document.querySelector(".toast");
    if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
    t.textContent = msg;
    t.classList.add("on");
    clearTimeout(t._h);
    t._h = setTimeout(() => t.classList.remove("on"), 3200);
  }
  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  }
  function svgOf(el, name) {
    const v = getComputedStyle(el).getPropertyValue("--" + name).trim();
    const m = v.match(/base64,([^")]+)/);
    return m ? atob(m[1]) : null;
  }
  function download(text, filename) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ── specimens ──
  async function variant(sp) {
    const seed = Math.floor(Math.random() * 1e6);
    sp.dataset.variantSeed = seed;
    sp.querySelectorAll("[data-shape]").forEach((el) => { el.dataset.seed = seed; GOShapes.draw(el); });
    const vars = (sp.dataset.vars || "").split(/\s+/).filter(Boolean);
    if (!vars.length) return;
    if (!served) { toast("Open this page through the local server to make ornament variants."); return; }
    try {
      const j = await post("/api/designlib/variant", { theme, vars, seed });
      for (const [k, v] of Object.entries(j.vars)) sp.style.setProperty("--" + k, v);
      sp.querySelectorAll("[data-shape]").forEach((el) => GOShapes.draw(el));
    } catch (e) { toast("Variant failed: " + e.message); }
  }
  async function save(sp) {
    if (!served) { toast("Open this page through the local server (http://127.0.0.1:8790/designlib/tiles/) to save."); return; }
    const clone = sp.cloneNode(true);
    clone.querySelectorAll(".tools, svg.shape").forEach((n) => n.remove());
    clone.querySelectorAll("[data-shape]").forEach((n) => n.style.removeProperty("clip-path"));
    try {
      const j = await post("/api/designlib/save", {
        theme, kind: sp.dataset.kind || "element", label: sp.dataset.label || "", html: clone.outerHTML,
        vars: (sp.dataset.vars || "").split(/\s+/).filter(Boolean).map((n) => [n, getComputedStyle(sp).getPropertyValue("--" + n).trim()]),
      });
      toast("Saved to designlib/saved/" + j.id);
    } catch (e) { toast("Save failed: " + e.message); }
  }
  document.querySelectorAll(".specimen").forEach((sp) => {
    const bar = document.createElement("div");
    bar.className = "tools";
    bar.innerHTML = '<button type="button" data-act="variant" title="A new variant of this ornament and shape">↻ variant</button>' +
      '<button type="button" data-act="save" title="Save to designlib/saved/">Save</button>' +
      (sp.dataset.svg ? '<button type="button" data-act="svg" title="Download the ornament as SVG">SVG</button>' : "");
    bar.addEventListener("click", (e) => {
      const act = e.target.dataset.act;
      e.stopPropagation();
      if (act === "variant") variant(sp);
      if (act === "save") save(sp);
      if (act === "svg") {
        const s = svgOf(sp, sp.dataset.svg);
        if (s) download(s, `${theme}-${sp.dataset.svg}-${sp.dataset.variantSeed || "base"}.svg`);
      }
    });
    sp.appendChild(bar);
  });

  // ── the page's own controls ──
  document.querySelectorAll("[role=tablist]").forEach((list) => {
    const tabs = [...list.querySelectorAll("[role=tab]")];
    tabs.forEach((tab) => tab.addEventListener("click", () => {
      tabs.forEach((t) => {
        const on = t === tab;
        t.setAttribute("aria-selected", on);
        const panel = document.getElementById(t.getAttribute("aria-controls"));
        if (panel) panel.hidden = !on;
      });
    }));
  });
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-action]");
    if (!b) return;
    const act = b.dataset.action, sp = b.closest(".specimen");
    if (act === "save-self" && sp) save(sp);
    if (act === "export") {
      const s = svgOf(document.documentElement, b.dataset.var || "divider");
      if (s) download(s, `${theme}-${b.dataset.var || "divider"}.svg`);
    }
    if (act === "layers") {
      document.body.classList.toggle("show-layers");
      toast(document.body.classList.contains("show-layers") ? "Showing the drawn shapes." : "Shapes hidden.");
    }
    if (act === "scroll") document.querySelector(b.dataset.target).scrollIntoView({ behavior: "smooth" });
    if (act === "toast") toast(b.dataset.msg);
  });
  if (!served) document.body.classList.add("offline");
})();
