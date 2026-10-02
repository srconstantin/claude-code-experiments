#!/usr/bin/env python3
"""
Best-Case Scenario — the browser version.

Runs a small server on this machine only (127.0.0.1) and opens the app in your
browser. It shares everything with the terminal app (bcs.py): the same prompts,
the same memory, the same ./threads/ files. Use either one, interchangeably.

Run:  python3 web.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import bcs

HERE = Path(__file__).resolve().parent
PAGE = HERE / "webapp.html"
PORTS = range(8765, 8776)
LOCK = threading.Lock()  # one writer at a time; the model calls themselves run outside it

NO_MORE_QUESTIONS = (
    "\n\nYou have already asked your clarifying questions; the user's answers are in the additional "
    "context above. Do not ask more. Produce the scenarios now."
)


class Problem(Exception):
    """Something to tell the user about; becomes an error message in the browser."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def say(msg: str) -> None:
    print(f"  {time.strftime('%H:%M:%S')}  {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Operations (one per button in the browser)
# ─────────────────────────────────────────────────────────────────────────────

def find(threads: list[dict], thread_id: str, n: int | None = None) -> tuple[dict, dict | None]:
    thread = next((t for t in threads if t["id"] == thread_id), None)
    if thread is None:
        raise Problem("That thread no longer exists.", 404)
    if n is None:
        return thread, None
    sc = next((s for s in thread.get("scenarios", []) if s["n"] == n), None)
    if sc is None:
        raise Problem("That scenario no longer exists.", 404)
    return thread, sc


def model_call(what: str, fn):
    """Run a model call, reporting progress in the terminal and turning failures into messages."""
    say(f"asking the advisor: {what} …")
    t0 = time.time()
    try:
        result = fn()
    except bcs.anthropic.APIError as e:
        raise Problem(f"The API call failed: {e}", 502)
    except bcs.BackToMenu:
        raise Problem("The advisor didn't return a usable answer. Try again.", 502)
    say(f"done in {int(time.time() - t0)}s")
    return result


def op_state(_: dict) -> dict:
    return {"threads": bcs.load_threads(), "today": bcs.today(), "model": bcs.MODEL,
            "threads_dir": str(bcs.THREADS_DIR)}


def op_new_thread(body: dict) -> dict:
    situation = str(body.get("situation", "")).strip()
    if not situation:
        raise Problem("Describe the situation first.")
    with LOCK:
        thread = bcs.new_thread(situation)
        bcs.save_thread(thread)
    say(f"new thread {thread['id']}")
    return {"thread": thread}


def op_generate(body: dict) -> dict:
    """Generate (or regenerate) scenarios. May come back with clarifying questions instead."""
    with LOCK:
        threads = bcs.load_threads()
        thread, _ = find(threads, body["id"])
        answers = [a for a in body.get("answers") or [] if a.get("q")]
        if answers:
            text = "\n\n".join(f"Q: {a['q']}\nA: {str(a.get('a', '')).strip() or '(no answer)'}" for a in answers)
            thread["context_additions"].append(text)
            bcs.log(thread, "clarified", questions=[a["q"] for a in answers])
        context = str(body.get("context", "")).strip()
        if context:
            thread["context_additions"].append(context)
            bcs.log(thread, "context_added")
        if answers or context:
            bcs.save_thread(thread)
    asked = any(e.get("event") == "clarified" for e in thread.get("log", []))
    prompt = bcs.situation_prompt_text(thread) + (NO_MORE_QUESTIONS if asked else "")
    system = bcs.build_system(bcs.memory_digest(threads, exclude_id=thread["id"]))
    result = model_call("scenarios", lambda: bcs.call_json(system, [{"role": "user", "content": prompt}]))
    questions = [q for q in result.get("clarifying_questions") or [] if str(q).strip()]
    with LOCK:
        if questions and not result.get("scenarios") and not asked:
            if not thread.get("title") and str(result.get("thread_title") or "").strip():
                thread["title"] = result["thread_title"].strip()
                bcs.save_thread(thread)
            return {"thread": thread, "questions": questions}
        bcs.store_scenarios(thread, result)
    return {"thread": thread}


def op_rate(body: dict) -> dict:
    with LOCK:
        thread, _ = find(bcs.load_threads(), body["id"])
        for r in body.get("ratings", []):
            _, sc = find([thread], thread["id"], int(r["n"]))
            sc["liked"], sc["possible"] = r.get("liked"), r.get("possible")
            if str(r.get("comment", "")).strip():
                sc["comment"] = r["comment"].strip()
            bcs.log(thread, "rated", n=sc["n"], liked=sc["liked"], possible=sc["possible"])
        bcs.save_thread(thread)
    return {"thread": thread}


def op_suggest(body: dict) -> dict:
    threads = bcs.load_threads()
    thread, sc = find(threads, body["id"], int(body["n"]))
    result = model_call(f"next actions for scenario {sc['n']}", lambda: bcs.fetch_actions(thread, sc, threads))
    return {"thread": thread, "note": result["note"], "count": len(result["actions"])}


def op_my_actions(body: dict) -> dict:
    added = [ln.strip(" -•") for ln in str(body.get("text", "")).splitlines() if ln.strip(" -•")]
    if not added:
        raise Problem("Type at least one action.")
    with LOCK:
        thread, sc = find(bcs.load_threads(), body["id"], int(body["n"]))
        sc["my_actions"].extend({"action": a, "added": bcs.today()} for a in added)
        bcs.log(thread, "my_actions", n=sc["n"], count=len(added))
        bcs.save_thread(thread)
    return {"thread": thread}


def op_mark(body: dict) -> dict:
    status = body.get("status")
    if status not in ("wont", "may", "did", None):
        raise Problem("Unknown mark.")
    with LOCK:
        thread, sc = find(bcs.load_threads(), body["id"], int(body["n"]))
        actions = bcs.scenario_actions(sc)
        i = int(body["index"])
        # the browser sends the action's text too, so a stale page can't mark the wrong one
        if not (0 <= i < len(actions)) or actions[i]["action"] != body.get("action"):
            raise Problem("That list is out of date. Reload the page and try again.", 409)
        bcs.set_mark(thread, sc, actions[i], status)
        bcs.save_thread(thread)
    return {"thread": thread}


def op_update(body: dict) -> dict:
    text = str(body.get("text", "")).strip()
    resolution = body.get("resolution")
    if not text:
        raise Problem("Write what happened first.")
    if resolution not in ("happened", "partly", "didnt", None):
        raise Problem("Unknown resolution.")
    with LOCK:
        thread, sc = find(bcs.load_threads(), body["id"], int(body["n"]))
        sc["updates"].append({"date": bcs.today(), "text": text, "resolution": resolution})
        if resolution:
            sc["status"] = "resolved"
        bcs.log(thread, "update", n=sc["n"], resolution=resolution)
        bcs.save_thread(thread)
    return {"thread": thread}


OPS = {
    "state": op_state, "new_thread": op_new_thread, "generate": op_generate, "rate": op_rate,
    "suggest": op_suggest, "my_actions": op_my_actions, "mark": op_mark, "update": op_update,
}


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP
# ─────────────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "BestCase"

    def log_message(self, *args) -> None:  # the ops print their own progress
        pass

    def local_only(self) -> bool:
        """This server is for the browser on this machine; refuse anything that looks like another site calling in."""
        port = self.server.server_address[1]
        ours = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in ours and (origin is None or origin.split("//")[-1] in ours)

    def send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, obj: dict) -> None:
        self.send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if not self.local_only():
            return self.send_json(403, {"error": "local use only"})
        if self.path in ("/", "/index.html"):
            return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        op = OPS.get(self.path.removeprefix("/api/")) if self.path.startswith("/api/") else None
        if not self.local_only() or "application/json" not in (self.headers.get("Content-Type") or ""):
            return self.send_json(403, {"error": "local use only"})
        if op is None:
            return self.send_json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            self.send_json(200, op(body))
        except Problem as e:
            self.send_json(e.status, {"error": str(e)})
        except (KeyError, ValueError, TypeError) as e:
            self.send_json(400, {"error": f"bad request: {e}"})
        except Exception as e:  # keep the server up; show the problem in the browser and here
            say(f"error in {self.path}: {e!r}")
            self.send_json(500, {"error": f"{type(e).__name__}: {e}"})


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Export it and run again.")
        sys.exit(1)
    server = None
    for port in PORTS:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        print(f"No free port between {PORTS[0]} and {PORTS[-1]}. Is the app already running?")
        sys.exit(1)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"\n  ✦ Best-Case Scenario is running at {url}")
    print(f"    threads: {bcs.THREADS_DIR}")
    print("    Leave this window open while you use the app. Ctrl-C stops it.\n", flush=True)
    bcs.ensure_titles(bcs.load_threads())
    if not os.environ.get("BCS_NO_BROWSER"):
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  ✦ go well.\n")


if __name__ == "__main__":
    main()
