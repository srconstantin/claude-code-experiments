#!/usr/bin/env python3
"""
Grammar of Ornament — local server for the feature-visualization browser.

Serves webapp.html on 127.0.0.1, renders activation-maximization images in a
single background worker (one GPU job at a time), caches every render in
./cache/, and copies the ones you like into ./features/.

Run:  python3 server.py
"""

from __future__ import annotations

import hashlib
import json
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import vis

HERE = Path(__file__).resolve().parent
PAGE = HERE / "webapp.html"
CACHE = HERE / "cache"
SAVED = HERE / "features"
INDEX = SAVED / "index.json"
PORTS = range(8790, 8801)
BATCH = 16  # images optimized together in one job chunk

MODEL_LOCK = threading.Lock()  # the GPU does one thing at a time
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
QUEUE: "queue.Queue[str]" = queue.Queue()


class Problem(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def say(msg: str) -> None:
    print(f"  {time.strftime('%H:%M:%S')}  {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Render requests, cache paths
# ─────────────────────────────────────────────────────────────────────────────

def parse_target(d: dict) -> vis.Target:
    try:
        c = int(d["c"])
    except (KeyError, TypeError, ValueError):
        raise Problem("each target needs an integer channel 'c'")
    y, x = d.get("y"), d.get("x")
    if (y is None) != (x is None):
        raise Problem("a neuron target needs both y and x")
    return vis.Target(c, None if y is None else int(y), None if x is None else int(x), bool(d.get("negate")))


def parse_settings(body: dict) -> dict:
    preset = vis.PRESETS.get(body.get("preset", "full"), vis.PRESETS["full"])
    s = {
        "model": str(body.get("model") or "convnext_base"),
        "layer": str(body.get("layer") or ""),
        "size": int(body.get("size") or preset["size"]),
        "steps": int(body.get("steps") or preset["steps"]),
        "seed": int(body.get("seed") or 0),
        "decay": float(body.get("decay") if body.get("decay") is not None else preset["decay"]),
        "tv": float(body.get("tv") if body.get("tv") is not None else preset["tv"]),
    }
    if not s["layer"]:
        raise Problem("layer is required")
    if not 32 <= s["size"] <= 1024 or not 1 <= s["steps"] <= 4096:
        raise Problem("size must be 32–1024 and steps 1–4096")
    return s


def cache_path(s: dict, t: vis.Target) -> Path:
    key = json.dumps([s["model"], s["layer"], asdict(t), s["size"], s["steps"], s["seed"], s["decay"], s["tv"]])
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    safe_model = s["model"].replace("/", "_").replace(":", "_")
    return CACHE / safe_model / s["layer"] / f"{t.label()}__{s['size']}px__{h}.png"


def rel(p: Path) -> str:
    return "/" + p.relative_to(HERE).as_posix()


def describe(s: dict, t: vis.Target) -> dict:
    return {**s, "target": asdict(t), "url": rel(cache_path(s, t))}


# ─────────────────────────────────────────────────────────────────────────────
#  Background worker
# ─────────────────────────────────────────────────────────────────────────────

class Cancelled(Exception):
    pass


def worker() -> None:
    while True:
        job_id = QUEUE.get()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None or job["cancelled"]:
            continue
        s = job["settings"]
        try:
            with MODEL_LOCK:
                lm = vis.load_model(s["model"])
                chunks = job["chunks"]
                for ci, chunk in enumerate(chunks):
                    if job["cancelled"]:
                        raise Cancelled()
                    t0 = time.time()

                    def progress(step: int, steps: int, ci=ci) -> None:
                        if job["cancelled"]:
                            raise Cancelled()
                        job["progress"] = (ci + step / steps) / len(chunks)

                    imgs = vis.render(lm, s["layer"], chunk, size=s["size"], steps=s["steps"], seed=s["seed"],
                                      decay_power=s["decay"], tv_weight=s["tv"], progress=progress)
                    for t, im in zip(chunk, imgs):
                        p = cache_path(s, t)
                        vis.save_png(im, p)
                        job["images"].append(describe(s, t))
                    say(f"rendered {len(chunk)} × {s['layer']} @ {s['size']}px/{s['steps']} in {time.time() - t0:.1f}s")
            job["progress"] = 1.0
        except Cancelled:
            say(f"cancelled job {job_id[:8]}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            job["error"] = f"{type(e).__name__}: {e}"
        finally:
            job["done"] = True


def submit(s: dict, targets: list[vis.Target]) -> dict:
    hits, misses = [], []
    for t in targets:
        (hits if cache_path(s, t).exists() else misses).append(t)
    images = [describe(s, t) for t in hits]
    if not misses:
        return {"done": True, "progress": 1.0, "images": images}
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id, "settings": s, "created": time.time(),
        "chunks": [misses[i:i + BATCH] for i in range(0, len(misses), BATCH)],
        "images": images, "progress": 0.0, "done": False, "cancelled": False, "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        # forget finished jobs older than an hour
        for k in [k for k, j in JOBS.items() if j["done"] and time.time() - j["created"] > 3600]:
            del JOBS[k]
    QUEUE.put(job_id)
    return job_view(job)


def job_view(job: dict) -> dict:
    return {"job_id": job["id"], "done": job["done"], "progress": job["progress"],
            "images": list(job["images"]), "error": job["error"], "cancelled": job["cancelled"]}


# ─────────────────────────────────────────────────────────────────────────────
#  Saving
# ─────────────────────────────────────────────────────────────────────────────

SAVE_LOCK = threading.Lock()


def read_index() -> list[dict]:
    if not INDEX.exists():
        return []
    items = json.loads(INDEX.read_text())
    for it in items:  # the url is derived from the name, so a folder rename can't strand it
        it["url"] = f"/features/{it['name']}.png"
    return items


def write_index(items: list[dict]) -> None:
    SAVED.mkdir(exist_ok=True)
    INDEX.write_text(json.dumps(items, indent=1))


def save_image(s: dict, t: vis.Target, note: str) -> dict:
    src = cache_path(s, t)
    if not src.exists():
        raise Problem("That image has not been rendered yet.", 404)
    safe_model = s["model"].replace("/", "_").replace(":", "_")
    name = f"{safe_model}__{s['layer']}__{t.label()}__{s['size']}px__s{s['seed']}__{src.stem.rsplit('__', 1)[1][:6]}"
    with SAVE_LOCK:
        items = read_index()
        if any(it["name"] == name for it in items):
            raise Problem("Already saved.", 409)
        SAVED.mkdir(exist_ok=True)
        shutil.copyfile(src, SAVED / f"{name}.png")
        item = {"name": name, "url": f"/features/{name}.png", "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": note, **s, "target": asdict(t)}
        (SAVED / f"{name}.json").write_text(json.dumps(item, indent=1))
        items.append(item)
        write_index(items)
    say(f"saved {name}.png")
    return item


def delete_saved(name: str) -> None:
    if "/" in name or ".." in name:
        raise Problem("bad name")
    with SAVE_LOCK:
        items = read_index()
        keep = [it for it in items if it["name"] != name]
        if len(keep) == len(items):
            raise Problem("No such saved image.", 404)
        for ext in (".png", ".json"):
            p = SAVED / f"{name}{ext}"
            if p.exists():
                p.unlink()
        write_index(keep)
    say(f"deleted {name}")


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP
# ─────────────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a) -> None:  # quiet
        pass

    def _json(self, obj, status: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, ctype: str, cache: bool) -> None:
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=31536000, immutable" if cache else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n == 0:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except json.JSONDecodeError:
            raise Problem("invalid JSON")

    def _static(self, root: Path, sub: str, cache: bool) -> None:
        p = (root / sub).resolve()
        if not str(p).startswith(str(root.resolve())) or p.suffix != ".png":
            return self._json({"error": "not found"}, 404)
        self._file(p, "image/png", cache)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self._file(PAGE, "text/html; charset=utf-8", False)
            if u.path.startswith("/cache/"):
                return self._static(CACHE, u.path[len("/cache/"):], True)
            if u.path.startswith("/features/"):
                return self._static(SAVED, u.path[len("/features/"):], False)
            if u.path == "/api/models":
                return self._json({"builtin": vis.BUILTIN_MODELS, "loaded": list(vis._LOADED),
                                   "device": str(vis.DEVICE), "presets": vis.PRESETS})
            if u.path == "/api/layers":
                name = q.get("model") or "convnext_base"
                with MODEL_LOCK:
                    try:
                        lm = vis.load_model(name)
                    except Exception as e:  # noqa: BLE001
                        raise Problem(f"Could not load model {name!r}: {e}", 400)
                return self._json({"model": lm.name, "input_size": lm.input_size,
                                   "layers": [asdict(l) for l in lm.layers]})
            if u.path.startswith("/api/job/"):
                with JOBS_LOCK:
                    job = JOBS.get(u.path.rsplit("/", 1)[1])
                if job is None:
                    raise Problem("no such job", 404)
                return self._json(job_view(job))
            if u.path == "/api/features":
                return self._json({"items": list(reversed(read_index()))})
            raise Problem("not found", 404)
        except Problem as p:
            self._json({"error": str(p)}, p.status)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/render":
                s = parse_settings(body)
                targets = [parse_target(t) for t in body.get("targets") or []]
                if not targets:
                    raise Problem("no targets")
                if len(targets) > 256:
                    raise Problem("at most 256 targets per request")
                with MODEL_LOCK:
                    lm = vis.load_model(s["model"])
                info = next((l for l in lm.layers if l.name == s["layer"]), None)
                if info is None:
                    raise Problem(f"unknown layer {s['layer']!r}", 404)
                for t in targets:
                    if not 0 <= t.channel < info.channels:
                        raise Problem(f"channel {t.channel} out of range (0–{info.channels - 1})")
                    if t.is_neuron and not (0 <= t.y < info.height and 0 <= t.x < info.width):
                        raise Problem(f"location out of range for a {info.height}×{info.width} map")
                return self._json(submit(s, targets))
            if u.path == "/api/cancel":
                with JOBS_LOCK:
                    job = JOBS.get(str(body.get("job_id")))
                if job is not None:
                    job["cancelled"] = True
                return self._json({"ok": True})
            if u.path == "/api/save":
                s = parse_settings(body)
                t = parse_target(body.get("target") or {})
                return self._json(save_image(s, t, str(body.get("note") or "").strip()))
            raise Problem("not found", 404)
        except Problem as p:
            self._json({"error": str(p)}, p.status)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_DELETE(self) -> None:
        u = urlparse(self.path)
        try:
            if u.path.startswith("/api/features/"):
                delete_saved(u.path[len("/api/features/"):])
                return self._json({"ok": True})
            raise Problem("not found", 404)
        except Problem as p:
            self._json({"error": str(p)}, p.status)


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    for port in PORTS:
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        print("No free port in", PORTS)
        sys.exit(1)
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  Grammar of Ornament — {url}\n  device: {vis.DEVICE}   cache: {CACHE}   features: {SAVED}\n"
          f"  Leave this window open; Ctrl-C stops the server.\n", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye")


if __name__ == "__main__":
    main()
