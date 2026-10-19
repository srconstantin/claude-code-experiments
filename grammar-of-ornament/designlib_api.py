"""
Grammar of Ornament — the design library's routes in server.py.

  GET    /designlib/<path>               the style tiles and their assets (designlib/)
  GET    /designlib/saved/               a gallery of saved specimens
  GET    /api/designlib/saved            the saved specimens as JSON
  POST   /api/designlib/variant          {theme, vars, seed} → {vars: {name: css url()}}
  POST   /api/designlib/palette          {theme, colorway} → {vars}: a source tile recoloured from a colorway
  POST   /api/designlib/save             {theme, kind, label, html, vars} → {id}
  DELETE /api/designlib/saved/<id>      moves it to designlib/saved/.trash/<id> (recoverable)

A saved specimen is a folder designlib/saved/<id>/ with specimen.html (a
standalone page: the theme's styles and fonts, the shapes script, and the
element exactly as it was, variants included), meta.json, and each of its
ornaments as an SVG file.
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "designlib"
SAVED = ROOT / "saved"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json", ".jpg": "image/jpeg"}


def _send(h, body: bytes, ctype: str, status: int = 200) -> None:
    h.send_response(status)
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(len(body)))
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    h.wfile.write(body)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "element"


def _items() -> list[dict]:
    out = []
    if SAVED.exists():
        for p in sorted(SAVED.glob("*/meta.json")):
            try:
                out.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                continue
    out.sort(key=lambda m: m["saved"], reverse=True)
    return out


def _gallery() -> str:
    cards = "".join(
        f'<figure data-id="{m["id"]}"><iframe src="/designlib/saved/{m["id"]}/specimen.html" loading="lazy" title="{html.escape(m["label"])}"></iframe>'
        f'<figcaption><b>{html.escape(m["theme"])}</b> · {html.escape(m["kind"])} · {html.escape(m["label"])}<br><small>{m["saved"]} · '
        f'<a href="/designlib/saved/{m["id"]}/specimen.html" target="_blank">open</a> · '
        + " ".join(f'<a href="/designlib/saved/{m["id"]}/{s}" download>{s}</a>' for s in m.get("svgs", []))
        + f' · <button data-del="{m["id"]}">delete</button></small></figcaption></figure>' for m in _items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Saved specimens</title><style>
body{{font:14px/1.45 system-ui,sans-serif;margin:28px;background:#f4f2ee;color:#222}}
h1{{font-size:20px}} nav a{{margin-right:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px;margin-top:18px}}
figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden}}
iframe{{width:100%;height:280px;border:0;display:block}}
figcaption{{padding:10px 12px;border-top:1px solid #eee}} small{{color:#666}}
button{{font:inherit;font-size:12px;border:1px solid #ccc;background:#fff;border-radius:4px;cursor:pointer}}
</style></head><body><h1>Saved specimens</h1>
<nav><a href="/designlib/tiles/woodcut.html">Woodcut</a><a href="/designlib/tiles/engraving.html">Engraving</a><a href="/designlib/tiles/filigree.html">Filigree</a></nav>
<p>Each is a standalone page in <code>designlib/saved/&lt;id&gt;/</code> with its ornaments as SVG files.</p>
<div class="grid">{cards or "<p>Nothing saved yet. Hover a component on a style tile and press Save.</p>"}</div>
<script>document.addEventListener("click",async e=>{{const id=e.target.dataset.del;if(!id||!confirm("Delete this specimen?"))return;
await fetch("/api/designlib/saved/"+id,{{method:"DELETE"}});e.target.closest("figure").remove();}});</script></body></html>"""


def _save(body: dict) -> dict:
    theme = str(body.get("theme") or "")
    page = ROOT / "tiles" / f"{theme}.html"
    if not page.exists():
        raise ValueError(f"unknown theme {theme!r}")
    src = page.read_text()
    head = src.split("<body", 1)[0]
    shapes_js = (ROOT / "tiles" / "shapes.js").read_text()
    kind, label = str(body.get("kind") or "element"), str(body.get("label") or "")
    sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{theme}-{_slug(kind + ' ' + label)}"
    out = SAVED / sid
    out.mkdir(parents=True, exist_ok=True)
    element = str(body.get("html") or "")
    # images referenced relative to the tiles folder
    element = element.replace("url(assets/", "url(/designlib/tiles/assets/").replace('url("assets/', 'url("/designlib/tiles/assets/')
    head = head.replace("url(assets/", "url(/designlib/tiles/assets/")
    (out / "specimen.html").write_text(
        head + f'<body data-theme="{theme}" style="padding:44px 48px"><div class="wrap" style="padding:0">{element}</div>'
        f'<style>.specimen{{margin:0}}.specimen:hover{{box-shadow:none}}</style><script>{shapes_js}</script></body></html>')
    svgs = []
    for name, value in body.get("vars") or []:
        m = re.search(r"base64,([^\")]+)", value or "")
        if m:
            fn = f"{_slug(name)}.svg"
            (out / fn).write_bytes(base64.b64decode(m.group(1)))
            svgs.append(fn)
    meta = {"id": sid, "theme": theme, "kind": kind, "label": label, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "svgs": svgs}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def handle(h, method: str, u) -> bool:
    """Serve a design-library request; returns False if the path is not ours."""
    path = u.path
    try:
        if method == "GET" and path in ("/designlib/saved", "/designlib/saved/"):
            _send(h, _gallery().encode(), TYPES[".html"])
            return True
        if method == "GET" and path in ("/designlib/tiles", "/designlib/tiles/"):
            path = "/designlib/tiles/index.html"
        if method == "GET" and path.startswith("/designlib/"):
            p = (ROOT / path[len("/designlib/"):]).resolve()
            if not str(p).startswith(str(ROOT.resolve())) or not p.is_file() or p.suffix not in TYPES:
                h._json({"error": "not found"}, 404)
                return True
            _send(h, p.read_bytes(), TYPES[p.suffix])
            return True
        if method == "GET" and path == "/api/designlib/saved":
            h._json({"items": _items()})
            return True
        if method == "POST" and path == "/api/designlib/variant":
            import atlastiles
            import sourcetiles
            import styletiles
            b = h._body()
            theme = str(b.get("theme"))
            mod = next((m for m in (sourcetiles, styletiles, atlastiles) if theme in m.SPECS), None)
            if mod is None:
                h._json({"error": f"unknown theme {theme!r}"}, 400)
                return True
            h._json({"vars": mod.variant_vars(theme, [str(n) for n in b.get("vars") or []], int(b.get("seed") or 0))})
            return True
        if method == "POST" and path == "/api/designlib/palette":
            import sourcetiles
            b = h._body()
            theme, cw = str(b.get("theme")), str(b.get("colorway") or "default")
            if theme not in sourcetiles.SPECS:
                h._json({"error": f"colorways are only on the source tiles, not {theme!r}"}, 400)
                return True
            if cw != "default" and not (HERE / "colorways" / "convnext_basefeatures" / f"{cw}.json").exists():
                h._json({"error": f"no colorway {cw!r}"}, 400)
                return True
            h._json({"vars": sourcetiles.palette_vars(theme, cw)})
            return True
        if method == "POST" and path == "/api/designlib/save":
            h._json(_save(h._body()))
            return True
        if method == "DELETE" and path.startswith("/api/designlib/saved/"):
            sid = path.rsplit("/", 1)[1]
            d = SAVED / sid
            if "/" in sid or ".." in sid or not (d / "meta.json").exists():
                h._json({"error": "no such specimen"}, 404)
                return True
            trash = SAVED / ".trash"          # deletes are reversible: the folder moves here
            trash.mkdir(exist_ok=True)
            shutil.move(str(d), str(trash / sid))
            h._json({"ok": True, "trashed": f"designlib/saved/.trash/{sid}"})
            return True
    except ValueError as e:
        h._json({"error": str(e)}, 400)
        return True
    return False
