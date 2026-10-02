#!/usr/bin/env python3
"""
Best-Case Scenario — a terminal chatbot that helps you see the realistic,
credible, *positive* ways a situation could turn out.

Powered by Claude Fable 5.1 via the Anthropic API (needs ANTHROPIC_API_KEY).

Every thread (situation → scenarios → your ratings → next actions → what
actually happened) is saved as JSON in ./threads/ and fed back to the model
as memory on later runs.

Run:  python3 bcs.py
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import re
import select
import shutil
import sys
import textwrap
import time
from pathlib import Path

# Without readline, input() reads through the tty's line buffer, which on macOS
# caps a single line at 1024 bytes (past that, even Enter is ignored). With it,
# a line can be any length, so long paragraphs can be typed or pasted.
try:
    import readline
    _libedit = "libedit" in (readline.__doc__ or "")
    # a pasted tab should be a tab, not a completion request
    readline.parse_and_bind("bind ^I ed-insert" if _libedit else "tab: self-insert")
except ImportError:  # pragma: no cover
    readline = None

try:
    import anthropic
except ImportError:  # pragma: no cover
    print("The 'anthropic' package is required:  pip install anthropic")
    sys.exit(1)

MODEL = "claude-fable-5-1"
HERE = Path(__file__).resolve().parent
THREADS_DIR = Path(os.environ.get("BCS_THREADS_DIR", HERE / "threads"))
PROFILE_FILE = HERE / "profile.md"
MEMORY_CHAR_BUDGET = 60_000  # ~15k tokens of past threads, newest first


# ─────────────────────────────────────────────────────────────────────────────
#  Look & feel
# ─────────────────────────────────────────────────────────────────────────────

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _sgr(code: str) -> str:
    return f"\033[{code}m" if USE_COLOR else ""


RESET = _sgr("0")
BOLD = _sgr("1")
ITAL = _sgr("3")
# Mid-tone colors, each readable on a white background and on a black one.
# No near-white and no SGR "faint": both vanish on a light terminal.
AMBER = _sgr("38;5;130")
TEAL = _sgr("38;5;30")
ROSE = _sgr("38;5;161")
LIME = _sgr("38;5;28")
SLATE = _sgr("38;5;241")
DIM = SLATE
INK = _sgr("39")  # the terminal's own text color
VIOLET = _sgr("38;5;93")
DARKBG = _sgr("48;5;236")

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def vlen(s: str) -> int:
    """Visible length of a string (ANSI codes stripped)."""
    return len(ANSI_RE.sub("", s))


def width() -> int:
    return max(60, min(shutil.get_terminal_size((90, 24)).columns, 96))


def clear() -> None:
    if USE_COLOR:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


BANNER = r"""
 ____  _____ ____ _____    ____    _    ____  _____
| __ )| ____/ ___|_   _|  / ___|  / \  / ___|| ____|
|  _ \|  _| \___ \ | |   | |     / _ \ \___ \|  _|
| |_) | |___ ___) || |   | |___ / ___ \ ___) | |___
|____/|_____|____/ |_|    \____/_/   \_\____/|_____|
"""


def banner() -> None:
    w = width()
    print()
    for line in BANNER.strip("\n").splitlines():
        print(f"  {AMBER}{BOLD}{line}{RESET}")
    sub = "S C E N A R I O   ·   realistic, credible, positive ways this could go"
    print(f"  {TEAL}{sub}{RESET}")
    print(f"  {SLATE}{'·' * min(w - 4, 70)}{RESET}")
    print(f"  {DIM}model: {MODEL}   threads: {THREADS_DIR.name}/   Ctrl-C returns to the menu{RESET}")
    print()


def hr(ch: str = "─", color: str = SLATE) -> None:
    print(f"{color}{ch * width()}{RESET}")


def wrap(text: str, indent: int = 0, w: int | None = None) -> list[str]:
    w = w or width()
    out: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(
            textwrap.wrap(
                para, width=w - indent, initial_indent=" " * indent,
                subsequent_indent=" " * indent, break_long_words=False,
            )
        )
    return out


def box(title: str, body_lines: list[str], color: str = TEAL, glyph: str = "✦") -> None:
    """Rounded box with the title embedded in the top border."""
    w = width()
    inner = w - 4
    head = f"{color}╭─ {glyph} {BOLD}{title}{RESET}{color} "
    head += "─" * max(0, w - vlen(head) - 1) + "╮" + RESET
    print(head)
    for raw in body_lines:
        for line in (wrap(raw, 0, inner) or [""]):
            pad = inner - vlen(line)
            print(f"{color}│{RESET} {line}{' ' * max(0, pad)} {color}│{RESET}")
    print(f"{color}╰{'─' * (w - 2)}╯{RESET}")


def say(text: str, color: str = INK, indent: int = 2) -> None:
    for line in wrap(text, indent):
        print(f"{color}{line}{RESET}")


def note(text: str) -> None:
    say(text, SLATE + ITAL)


def warn(text: str) -> None:
    say("! " + text, ROSE)


# ─────────────────────────────────────────────────────────────────────────────
#  Input helpers
# ─────────────────────────────────────────────────────────────────────────────

class BackToMenu(Exception):
    pass


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" {DIM}[{default}]{RESET}" if default else ""
    # Short answers skip readline (it miscounts the width of a colored prompt),
    # so this reads the line straight from the terminal.
    print(f"  {TEAL}❯{RESET} {prompt}{suffix} ", end="", flush=True)
    val = sys.stdin.readline()
    if not val:
        raise BackToMenu
    return val.strip() or (default or "")


def input_pending() -> bool:
    """True if more input is already waiting, i.e. we are in the middle of a paste."""
    try:
        return bool(select.select([sys.stdin], [], [], 0.05)[0])
    except (OSError, ValueError):
        return False


def ask_multiline(prompt: str) -> str:
    print(f"  {TEAL}❯{RESET} {prompt}")
    print(f"  {DIM}(type or paste as much as you like; finish with a blank line twice, or a lone '.'){RESET}")
    # readline needs a plain prompt to wrap long lines correctly
    gutter = "  │ " if readline and sys.stdin.isatty() else f"  {SLATE}│{RESET} "
    lines: list[str] = []
    blanks = 0
    while True:
        try:
            line = input(gutter)
        except EOFError:
            break
        # a terminator only counts when typed: inside a paste it is just text
        if line.strip() == "." and not input_pending():
            break
        if line.strip() == "":
            blanks += 1
            if blanks >= 2 and not input_pending():
                break
        else:
            blanks = 0
        lines.append(line)
    return "\n".join(lines).strip()


def choose(prompt: str, options: list[tuple[str, str]], default: str | None = None) -> str:
    """options: list of (key, label). Returns the chosen key."""
    print()
    for key, label in options:
        print(f"    {AMBER}[{key}]{RESET} {label}")
    keys = {k.lower() for k, _ in options}
    while True:
        val = ask(prompt, default).lower()
        if val in keys:
            return val
        warn(f"pick one of: {', '.join(sorted(keys))}")


def yes_no_skip(prompt: str) -> bool | None:
    while True:
        val = ask(f"{prompt} {DIM}(y / n / enter to skip){RESET}").lower()
        if val in ("y", "yes"):
            return True
        if val in ("n", "no"):
            return False
        if val == "":
            return None
        warn("y, n, or enter")


# ─────────────────────────────────────────────────────────────────────────────
#  Storage
# ─────────────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def today() -> str:
    return dt.date.today().isoformat()


def days_ago(date: str) -> str:
    n = (dt.date.today() - dt.date.fromisoformat(date[:10])).days
    return "today" if n <= 0 else f"{n} day{'s' if n != 1 else ''} ago"


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n].rstrip("-") or "situation"


def thread_path(thread: dict) -> Path:
    return THREADS_DIR / f"{thread['id']}.json"


def save_thread(thread: dict) -> None:
    THREADS_DIR.mkdir(exist_ok=True)
    thread["updated"] = now_iso()
    tmp = thread_path(thread).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(thread, indent=2, ensure_ascii=False))
    tmp.replace(thread_path(thread))


def load_threads() -> list[dict]:
    if not THREADS_DIR.exists():
        return []
    threads = []
    for p in sorted(THREADS_DIR.glob("*.json")):
        try:
            t = json.loads(p.read_text())
        except Exception as e:  # corrupt file: skip, don't crash
            warn(f"could not read {p.name}: {e}")
            continue
        upgrade_thread(t)
        threads.append(t)
    threads.sort(key=lambda t: t.get("created", ""))
    return threads


def new_thread(situation: str) -> dict:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return {
        "id": f"{ts}-{slugify(situation)}",
        "created": now_iso(),
        "updated": now_iso(),
        "situation": situation,
        "context_additions": [],
        "read": "",
        "scenarios": [],
        "note": "",
        "log": [{"ts": now_iso(), "event": "created"}],
    }


def log(thread: dict, event: str, **kw) -> None:
    thread.setdefault("log", []).append({"ts": now_iso(), "event": event, **kw})


def logged_dates(thread: dict, events: tuple[str, ...], n: int | None, how_many: int) -> list[str]:
    """Dates for list items that were stored without one, recovered from the log.

    Each matching log entry covers `count` items (one if it has no count). Items
    line up with the newest entries; any the log can't account for get the
    thread's creation date.
    """
    dates = [
        e["ts"][:10]
        for e in thread.get("log", [])
        if e.get("event") in events and e.get("n") == n
        for _ in range(e.get("count", 1))
    ]
    dates = dates[-how_many:] if how_many else []
    return [thread["created"][:10]] * (how_many - len(dates)) + dates


def upgrade_thread(thread: dict) -> None:
    """Bring a thread saved by an older version up to date.

    Suggested actions get the date they were suggested, and the user's own
    actions become records like the bot's (they used to be bare strings), so
    both kinds can carry a date and a won't / may / did mark.
    """
    for sc in thread.get("scenarios", []):
        bot = sc.get("bot_actions", [])
        for a, d in zip(bot, logged_dates(thread, ("bot_actions",), sc["n"], len(bot))):
            a.setdefault("suggested", d)
        mine = sc.get("my_actions", [])
        dates = logged_dates(thread, ("my_actions",), sc["n"], len(mine))
        sc["my_actions"] = [a if isinstance(a, dict) else {"action": a, "added": d} for a, d in zip(mine, dates)]


def scenario_actions(sc: dict) -> list[dict]:
    """The bot's suggestions followed by the user's own, in the order they are numbered on screen."""
    return sc.get("bot_actions", []) + sc.get("my_actions", [])


def action_date(a: dict) -> str:
    return a.get("suggested") or a.get("added") or ""


def is_open(a: dict) -> bool:
    """Unmarked and "may" actions are still open; "won't" and "did" are closed."""
    return a.get("status") not in ("wont", "did")


STATUS_WORD = {"wont": "won't", "may": "may", "did": "did"}


def status_note(a: dict) -> str:
    """Plain-text mark for the model, e.g. ' [user: did, 2026-09-20]'."""
    if not a.get("status"):
        return ""
    return f" [user: {STATUS_WORD[a['status']]}, {a.get('status_date', '?')}]"


def thread_title(t: dict) -> str:
    if t.get("title"):
        return t["title"]
    sit = t.get("situation", "").replace("\n", " ")
    return sit[:60] + ("…" if len(sit) > 60 else "")


def addition_dates(thread: dict) -> list[str]:
    return logged_dates(thread, ("clarified", "context_added"), None, len(thread.get("context_additions", [])))


def by_date(items: list, dates: list[str]) -> list[tuple[str, list]]:
    """Group consecutive items that share a date: [(date, [items...]), ...]."""
    return [(d, [x for x, _ in grp]) for d, grp in itertools.groupby(zip(items, dates), key=lambda p: p[1])]


def resolution_word(sc: dict) -> str:
    """How a resolved outcome turned out, from its most recent resolving update."""
    res = next((u["resolution"] for u in reversed(sc.get("updates", [])) if u.get("resolution")), None)
    return {"happened": "happened", "partly": "partly", "didnt": "didn't"}.get(res, "resolved")


def is_active(sc: dict) -> bool:
    return bool(sc.get("liked")) and bool(sc.get("possible")) and sc.get("status", "active") == "active"


# ─────────────────────────────────────────────────────────────────────────────
#  Memory digest (past threads → model context)
# ─────────────────────────────────────────────────────────────────────────────

def _yn(v) -> str:
    return {True: "yes", False: "no"}.get(v, "unrated")


def format_thread_for_memory(t: dict, full: bool = False) -> str:
    sit_limit = 2000 if full else 700
    lines = [f"### Thread \"{thread_title(t)}\", started {t['created'][:10]}  (id {t['id']})"]
    lines.append("Situation: " + t["situation"][:sit_limit].replace("\n", " "))
    for add, d in zip(t.get("context_additions", []), addition_dates(t)):
        lines.append(f"Added context ({d}): " + add[:600].replace("\n", " "))
    if t.get("read"):
        lines.append("Advisor's read: " + t["read"][:500])
    for sc in t.get("scenarios", []):
        lines.append(
            f"- Scenario {sc['n']}: {sc['title']} — liked: {_yn(sc.get('liked'))}; "
            f"possible: {_yn(sc.get('possible'))}; status: {sc.get('status', 'active')}"
        )
        if full:
            lines.append("    story: " + sc.get("story", "")[:800].replace("\n", " "))
        if sc.get("comment"):
            lines.append("    user comment: " + sc["comment"][:500].replace("\n", " "))
        acts = sc.get("bot_actions", [])
        for d, grp in by_date(acts, [a.get("suggested", "") for a in acts]):
            lines.append(f"    suggested actions ({d}): " + "; ".join(a["action"] + status_note(a) for a in grp)[:700])
        mine = sc.get("my_actions", [])
        for d, grp in by_date(mine, [a.get("added", "") for a in mine]):
            lines.append(f"    user's own actions ({d}): " + "; ".join(a["action"] + status_note(a) for a in grp)[:700])
        for u in sc.get("updates", []):
            res = f" [{u['resolution'].upper()}]" if u.get("resolution") else ""
            lines.append(f"    update {u['date']}{res}: " + u["text"][:600].replace("\n", " "))
    if t.get("note"):
        lines.append("Advisor note: " + t["note"][:300])
    return "\n".join(lines)


def memory_digest(threads: list[dict], exclude_id: str | None = None) -> str:
    """Newest threads first, until the character budget is spent."""
    chunks: list[str] = []
    used = 0
    for t in reversed(threads):
        if t["id"] == exclude_id:
            continue
        s = format_thread_for_memory(t)
        if used + len(s) > MEMORY_CHAR_BUDGET:
            break
        chunks.append(s)
        used += len(s)
    if not chunks:
        return "(no past threads yet)"
    # calibration tally: how did resolved outcomes go?
    tally = {"happened": 0, "partly": 0, "didnt": 0}
    for t in threads:
        for sc in t.get("scenarios", []):
            for u in sc.get("updates", []):
                if u.get("resolution") in tally:
                    tally[u["resolution"]] += 1
    head = (
        f"Calibration so far — outcomes the user marked desirable+possible and later resolved: "
        f"{tally['happened']} happened, {tally['partly']} partly, {tally['didnt']} didn't.\n\n"
    )
    return head + "\n\n".join(reversed(chunks))


def load_profile() -> str:
    if PROFILE_FILE.exists():
        return PROFILE_FILE.read_text().strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_CORE = """You are the Best-Case Scenario advisor: a trusted, clear-eyed friend with excellent judgment whose job is to help the user see the realistic, credible, POSITIVE ways a situation could turn out.

What you are NOT: a cheerleader, a motivational speaker, or a generator of pleasant fantasies. Your entire value rests on credibility. The user needs to be able to trust that anything you call a best-case scenario is one you honestly believe could happen.

THE BAR. Only include an outcome if, assuming the user makes a genuine best effort, you would honestly put its chance at better than roughly one in ten. Think of it as the upper tail of the realistic distribution: good outcomes conditioned on effort and ordinary luck, not on miracles, other people transforming, or the user becoming someone else. Never put explicit probability numbers in your output; the bar is for your own filtering. If only one outcome clears the bar, give one. If the honestly-best plausible outcome is modest, say so plainly. A modest, credible good outcome is far more useful than an inflated one.

HOW TO THINK. Reason like a good analyst and a good friend at once:
- Base rates: how do situations like this usually go for people like this user? What fraction end well, and what distinguishes those cases?
- Specifics: which facts in the description move the odds up or down? Take the user's constraints seriously (money, time, family, health, geography, the other people involved).
- Control: what is within the user's power vs. what depends on others or on chance? Best cases that route mostly through the user's own actions are more credible than ones that need someone else to change.
- Time: over what horizon does each outcome play out? Be concrete about rough timing.
- Mechanism: for each scenario, be able to say HOW it happens, step by step, not just that it does.
- Distinctness: scenarios should be genuinely different paths, not restatements of one path with different adjectives.
- Second-order goods: sometimes the best case is not "the thing works" but "the thing fails cleanly and something better becomes possible." Include those when they are real.

STYLE. Plain, specific, warm but unsentimental. Second person. No hedging boilerplate, no lists of caveats for their own sake, no "it's important to remember." Short paragraphs. Do not flatter the user. If the situation contains a mistaken assumption that matters, say so briefly and then proceed.

MEMORY. You are given the user's past threads: previous situations, the scenarios you offered, which ones the user found desirable and/or possible, their comments on why, next actions, and what actually happened. Use them to calibrate: learn what kinds of outcomes this user finds credible or not, what they value, how their past situations resolved, and whether your earlier scenarios were too rosy or too timid. If a past thread is clearly related to the current situation, connect them in one or two sentences. Do not recite the memory back.

DATES. Every request tells you today's date and when the current thread started, and the memory and context are dated: when each thread began, when context was added, when actions were suggested or listed, when updates were recorded. Reason about elapsed time. A suggestion from three weeks ago with no update since is probably stale or undone; a window that was open when the thread started may have closed. Give timing as actual dates or date ranges rather than "soon" or "next week".

ACTION MARKS. The user marks next actions as "did" (done), "may" (might do it), or "won't" (not going to do it); unmarked means still open. These marks are some of the best evidence you have. Build on what they did. Treat "won't" as information about what this user is unwilling or unable to do: do not re-suggest it in other words, and let it shape what you consider realistic for them.

CLARIFYING QUESTIONS. Ask at most two or three, and only if the answers would materially change which scenarios clear the bar. Otherwise state your key assumptions and proceed."""

SCENARIO_FORMAT = """OUTPUT FORMAT. Respond with a single JSON object and nothing else (no markdown fences, no prose outside the JSON):
{
  "thread_title": "A 4-8 word title for this thread that lets the user recognize it at a glance in a list: name the specific people, project or decision, not the general theme.",
  "clarifying_questions": ["..."],   // usually empty. If non-empty, leave "scenarios" empty and ask; you will be re-prompted with answers.
  "read": "Your honest read of the situation in two to five sentences: what it is, what the real constraints are, what the odds hinge on. Mention any related past thread here if relevant.",
  "assumptions": ["key assumptions you are making, if any"],
  "scenarios": [
    {
      "title": "Short, specific title (6-10 words)",
      "story": "What this outcome concretely looks like and roughly when, told as a plausible sequence of events. One to two short paragraphs.",
      "why_credible": "Why you believe this clears the bar: base rates, specifics from the situation, what's in the user's control.",
      "depends_on": ["the two to four things that most need to go right"],
      "early_signs": ["one to three observable signs, in the next weeks or months, that this path is in play"]
    }
  ],
  "note": "Optional. Anything else worth saying, e.g. if the bar was hard to clear, or a good outcome you deliberately left out and why."
}
Give between one and five scenarios, ordered from most likely to least. Fewer, better scenarios beat more, weaker ones."""

ACTIONS_FORMAT = """The user has marked one scenario as both desirable and possible, and wants concrete next actions to make it more likely.

Suggest three to six actions. Favor actions that are specific (what exactly to do, with whom, by when), that the user can start within days, that address the "depends_on" factors, and that are cheap to try relative to their information value or leverage. Order by expected impact. Do not pad with generic advice (no "stay positive", "network more", "take care of yourself") unless it is tied to a specific mechanism in this situation. If the user has already listed their own actions, build on them rather than repeating them, and say briefly if any of theirs seems low-value.

OUTPUT FORMAT. Respond with a single JSON object and nothing else:
{
  "actions": [
    {"action": "the specific thing to do", "why": "how it raises the odds of THIS scenario", "when": "timing as a date or date range, worked out from today's date"}
  ],
  "note": "Optional: one or two sentences, e.g. about sequencing or the single most important lever."
}"""


TITLES_FORMAT = """Give each of the user's numbered threads below a 4-8 word title that lets them recognize it at a glance in a list: name the specific people, project or decision, not the general theme.

Respond with a single JSON object and nothing else: {"titles": {"1": "...", "2": "..."}}"""


def build_system(memory: str) -> list[dict]:
    blocks = [{"type": "text", "text": SYSTEM_CORE}]
    profile = load_profile()
    if profile:
        blocks.append({"type": "text", "text": "ABOUT THE USER (from their profile file):\n\n" + profile})
    blocks.append({
        "type": "text",
        "text": "MEMORY — the user's past threads, newest last:\n\n" + memory,
        "cache_control": {"type": "ephemeral"},
    })
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
#  Model calls
# ─────────────────────────────────────────────────────────────────────────────

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            warn("ANTHROPIC_API_KEY is not set. Export it and run again.")
            sys.exit(1)
        _client = anthropic.Anthropic()
    return _client


def call_model(system: list[dict], messages: list[dict], max_tokens: int = 12000) -> str:
    """Stream the response (so long thinks don't time out) and show a spinner."""
    glyphs = "◐◓◑◒"
    t0 = time.time()
    i = 0
    text_parts: list[str] = []
    with client().messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=system,
        messages=messages,
    ) as stream:
        for event in stream:
            et = getattr(event, "type", "")
            if et == "content_block_delta":
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", "") == "text_delta":
                    text_parts.append(delta.text)
            if USE_COLOR and (i % 3 == 0):
                phase = "writing" if text_parts else "thinking"
                sys.stdout.write(f"\r  {VIOLET}{glyphs[(i // 3) % 4]} {phase}… {int(time.time() - t0)}s{RESET}   ")
                sys.stdout.flush()
            i += 1
        final = stream.get_final_message()
    if USE_COLOR:
        sys.stdout.write("\r" + " " * 40 + "\r")
    if not text_parts:
        text_parts = [b.text for b in final.content if getattr(b, "type", "") == "text"]
    u = final.usage
    note(f"({int(time.time() - t0)}s · tokens in/out {u.input_tokens}/{u.output_tokens})")
    return "".join(text_parts)


def parse_json(raw: str) -> dict:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            return json.loads(s[start:end + 1])
        raise


def call_json(system: list[dict], messages: list[dict], max_tokens: int = 12000) -> dict:
    for attempt in range(2):
        raw = call_model(system, messages, max_tokens)
        try:
            return parse_json(raw)
        except json.JSONDecodeError:
            if attempt == 0:
                warn("the model didn't return clean JSON; asking again…")
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "That was not valid JSON. Return ONLY the JSON object, nothing else."},
                ]
            else:
                warn("still not valid JSON. Raw response follows:")
                print(raw)
                raise BackToMenu


# ─────────────────────────────────────────────────────────────────────────────
#  Rendering threads and scenarios
# ─────────────────────────────────────────────────────────────────────────────

def show_scenario(sc: dict, color: str = TEAL) -> None:
    body: list[str] = []
    body.extend(sc.get("story", "").split("\n"))
    body.append("")
    body.append(f"{LIME}why credible:{RESET} {sc.get('why_credible', '')}")
    if sc.get("depends_on"):
        body.append("")
        body.append(f"{AMBER}depends on:{RESET}")
        body.extend(f"  • {d}" for d in sc["depends_on"])
    if sc.get("early_signs"):
        body.append("")
        body.append(f"{VIOLET}early signs:{RESET}")
        body.extend(f"  • {d}" for d in sc["early_signs"])
    box(f"SCENARIO {sc['n']} · {sc['title']}", body, color)


def rating_line(sc: dict) -> str:
    def mark(v, word):
        if v is True:
            return f"{LIME}✔ {word}{RESET}"
        if v is False:
            return f"{ROSE}✘ {word}{RESET}"
        return f"{SLATE}– {word}?{RESET}"
    st = sc.get("status", "active")
    st_s = f"{AMBER}● active{RESET}" if st == "active" else f"{SLATE}○ {st}{RESET}"
    return f"{mark(sc.get('liked'), 'desirable')}   {mark(sc.get('possible'), 'possible')}   {st_s}"


def status_mark(a: dict) -> str:
    glyph = {"did": "✔", "wont": "✘", "may": "◐"}[a["status"]]
    return f"{glyph} {STATUS_WORD[a['status']]} ({a.get('status_date', '?')})"


def show_scenario_full(sc: dict, thread: dict) -> None:
    show_scenario(sc)
    print(f"  {rating_line(sc)}")
    if sc.get("comment"):
        say(f"your comment: {sc['comment']}", SLATE + ITAL)
    n_bot = len(sc.get("bot_actions", []))
    for k, a in enumerate(scenario_actions(sc), 1):  # one numbering across both lists, used for marking
        if k == 1 and n_bot:
            print(f"\n  {LIME}suggested next actions:{RESET}")
        if k == n_bot + 1:
            print(f"\n  {AMBER}your next actions:{RESET}")
        say(f"{k}. {a['action']}", INK if is_open(a) else SLATE, 4)
        bits = [status_mark(a)] if a.get("status") else []
        if "why" in a:
            bits += [f"why: {a.get('why', '')}", f"when: {a.get('when', '')}"]
        bits.append(("suggested " if "suggested" in a else "added ") + (action_date(a) or "?"))
        say("  ·  ".join(bits), SLATE, 7)
    if sc.get("updates"):
        print(f"\n  {VIOLET}updates:{RESET}")
        for u in sc["updates"]:
            res = f" [{u['resolution']}]" if u.get("resolution") else ""
            say(f"{u['date']}{res}: {u['text']}", INK, 4)
    print()


def show_thread(thread: dict) -> None:
    hr("═", AMBER)
    print(f"  {AMBER}{BOLD}{thread_title(thread)}{RESET}  {DIM}started {thread['created'][:16].replace('T', ' ')}{RESET}")
    hr("═", AMBER)
    box("SITUATION", thread["situation"].split("\n"), AMBER, "◆")
    for add in thread.get("context_additions", []):
        box("ADDED CONTEXT", add.split("\n"), AMBER, "◇")
    if thread.get("read"):
        box("READ", [thread["read"]], VIOLET, "◈")
    for sc in thread.get("scenarios", []):
        show_scenario_full(sc, thread)
    if thread.get("note"):
        note("advisor note: " + thread["note"])


# ─────────────────────────────────────────────────────────────────────────────
#  Flows
# ─────────────────────────────────────────────────────────────────────────────

def situation_prompt_text(thread: dict) -> str:
    created = thread["created"][:10]
    parts = [
        f"TODAY'S DATE: {today()} ({dt.date.today():%A})",
        f"SITUATION (thread started {created}, {days_ago(created)}):\n" + thread["situation"],
    ]
    for add, d in zip(thread.get("context_additions", []), addition_dates(thread)):
        parts.append(f"ADDITIONAL CONTEXT FROM THE USER (added {d}):\n" + add)
    parts.append(SCENARIO_FORMAT)
    return "\n\n".join(parts)


def generate_scenarios(thread: dict, threads: list[dict]) -> None:
    system = build_system(memory_digest(threads, exclude_id=thread["id"]))
    messages = [{"role": "user", "content": situation_prompt_text(thread)}]
    while True:
        result = call_json(system, messages)
        qs = [q for q in result.get("clarifying_questions") or [] if q.strip()]
        if qs and not result.get("scenarios"):
            box("A FEW QUESTIONS FIRST", [f"{k}. {q}" for k, q in enumerate(qs, 1)], VIOLET, "?")
            answers = []
            for k, q in enumerate(qs, 1):
                a = ask_multiline(f"{k}. {q}")
                answers.append(f"Q: {q}\nA: {a or '(no answer)'}")
            answer_text = "\n\n".join(answers)
            thread["context_additions"].append(answer_text)
            log(thread, "clarified", questions=qs)
            save_thread(thread)
            messages = messages + [
                {"role": "assistant", "content": json.dumps(result)},
                {"role": "user", "content": "ANSWERS:\n\n" + answer_text + "\n\nNow produce the scenarios in the required JSON format."},
            ]
            continue
        break
    store_scenarios(thread, result)


def store_scenarios(thread: dict, result: dict) -> None:
    """Save the model's scenario response into the thread (used by the terminal app and the web app)."""
    if not thread.get("title") and (result.get("thread_title") or "").strip():
        thread["title"] = result["thread_title"].strip()
    thread["read"] = result.get("read", "")
    thread["note"] = result.get("note", "")
    thread["assumptions"] = result.get("assumptions", [])
    thread["scenarios"] = []
    for n, sc in enumerate(result.get("scenarios", []), 1):
        thread["scenarios"].append({
            "n": n,
            "title": sc.get("title", f"Scenario {n}"),
            "story": sc.get("story", ""),
            "why_credible": sc.get("why_credible", ""),
            "depends_on": sc.get("depends_on", []),
            "early_signs": sc.get("early_signs", []),
            "liked": None, "possible": None, "comment": "",
            "bot_actions": [], "my_actions": [], "updates": [],
            "status": "active",
        })
    log(thread, "scenarios_generated", count=len(thread["scenarios"]))
    save_thread(thread)


def present_scenarios(thread: dict) -> None:
    print()
    box("READ", [thread["read"]], VIOLET, "◈")
    if thread.get("assumptions"):
        note("assuming: " + "; ".join(thread["assumptions"]))
    print()
    if not thread["scenarios"]:
        warn("No scenario cleared the bar. That is itself information; see the note below.")
    for sc in thread["scenarios"]:
        show_scenario(sc)
        print()
    if thread.get("note"):
        note("note: " + thread["note"])
    print()


def rate_scenarios(thread: dict) -> None:
    hr()
    say("Rate each scenario. Honest ratings make future scenarios better.", AMBER + BOLD)
    for sc in thread["scenarios"]:
        print()
        print(f"  {TEAL}{BOLD}Scenario {sc['n']} · {sc['title']}{RESET}")
        sc["liked"] = yes_no_skip("Would you want this outcome?")
        sc["possible"] = yes_no_skip("Do you believe it's genuinely possible?")
        c = ask("Comment on why (optional):")
        if c:
            sc["comment"] = c
        log(thread, "rated", n=sc["n"], liked=sc["liked"], possible=sc["possible"])
        save_thread(thread)


def fetch_actions(thread: dict, sc: dict, threads: list[dict]) -> dict:
    """Ask the model for next actions and save them on the scenario. Returns {"actions": [...], "note": str}."""
    system = build_system(memory_digest(threads, exclude_id=thread["id"]))
    ctx = [
        situation_prompt_text(thread).replace(SCENARIO_FORMAT, "").strip(),
        "ADVISOR'S READ:\n" + thread.get("read", ""),
        "THE SCENARIO THE USER CHOSE:\n" + json.dumps(
            {k: sc[k] for k in ("title", "story", "why_credible", "depends_on", "early_signs")}, indent=1),
    ]
    if sc.get("comment"):
        ctx.append("USER'S COMMENT ON THIS SCENARIO:\n" + sc["comment"])
    if sc.get("my_actions"):
        ctx.append("ACTIONS THE USER HAS ALREADY LISTED:\n- " + "\n- ".join(
            f"[{a.get('added', '?')}] {a['action']}{status_note(a)}" for a in sc["my_actions"]))
    if sc.get("bot_actions"):
        ctx.append("ACTIONS YOU SUGGESTED EARLIER (don't repeat; refine or add):\n- " + "\n- ".join(
            f"[suggested {a.get('suggested', '?')}; timing given: {a.get('when', '?')}] {a['action']}{status_note(a)}"
            for a in sc["bot_actions"]))
    if sc.get("updates"):
        ctx.append("UPDATES SO FAR:\n" + "\n".join(f"{u['date']}: {u['text']}" for u in sc["updates"]))
    others = [s for s in thread["scenarios"] if s is not sc]
    if others:
        ctx.append("OTHER SCENARIOS IN THIS THREAD (for context): " + "; ".join(
            f"{s['title']} (liked {_yn(s.get('liked'))}, possible {_yn(s.get('possible'))})" for s in others))
    ctx.append(ACTIONS_FORMAT)
    result = call_json(system, [{"role": "user", "content": "\n\n".join(ctx)}], max_tokens=8000)
    actions = [a for a in result.get("actions", []) if a.get("action")]
    for a in actions:
        a["suggested"] = today()
    sc["bot_actions"].extend(actions)
    log(thread, "bot_actions", n=sc["n"], count=len(actions))
    save_thread(thread)
    return {"actions": actions, "note": result.get("note", "")}


def suggest_actions(thread: dict, sc: dict, threads: list[dict]) -> None:
    result = fetch_actions(thread, sc, threads)
    actions = result["actions"]
    body = []
    for k, a in enumerate(actions, 1):
        body.append(f"{BOLD}{k}. {a['action']}{RESET}")
        body.append(f"   {SLATE}why:{RESET} {a.get('why', '')}")
        body.append(f"   {SLATE}when:{RESET} {a.get('when', '')}")
        body.append("")
    if result.get("note"):
        body.append(f"{ITAL}{result['note']}{RESET}")
    box(f"NEXT ACTIONS · {sc['title']}", body, LIME, "➜")


def add_my_actions(thread: dict, sc: dict) -> None:
    say("Enter your own next actions, one per line.", AMBER)
    text = ask_multiline("Your actions:")
    added = [ln.strip(" -•") for ln in text.splitlines() if ln.strip(" -•")]
    if added:
        sc["my_actions"].extend({"action": a, "added": today()} for a in added)
        log(thread, "my_actions", n=sc["n"], count=len(added))
        save_thread(thread)
        say(f"added {len(added)} action(s).", LIME)


MARKS = {"w": "wont", "m": "may", "d": "did", "o": None}
MARK_HELP = "w = won't · m = may · d = did · o = open again"


def parse_mark(val: str) -> tuple[list[int], str]:
    """'3 d', '3,5 w', '4 did', "2 won't" -> the numbers and the first letter of the word ('' if none)."""
    nums = [int(x) for x in re.findall(r"\d+", val)]
    word = re.sub(r"[^a-z]", "", val.lower())
    return nums, word[:1]


def set_mark(thread: dict, sc: dict, a: dict, status: str | None) -> None:
    if status:
        a["status"], a["status_date"] = status, today()
    else:
        a.pop("status", None)
        a.pop("status_date", None)
    log(thread, "action_marked", n=sc["n"], action=a["action"][:80], status=status)


def mark_actions(entries: list[tuple[dict, dict, dict]], nums: list[int], key: str) -> bool:
    """entries: the numbered (thread, scenario, action) rows on screen. Returns True if anything was marked."""
    if key not in MARKS or not nums or not all(1 <= n <= len(entries) for n in nums):
        warn(f"type action number(s) and a letter, e.g. '3 d'   ({MARK_HELP})")
        time.sleep(1.5)
        return False
    for n in nums:
        set_mark(*entries[n - 1], MARKS[key])
    for thread in {id(e[0]): e[0] for e in (entries[n - 1] for n in nums)}.values():
        save_thread(thread)
    return True


def actions_menu(thread: dict, sc: dict, threads: list[dict]) -> None:
    while True:
        k = choose(
            f"Scenario {sc['n']} — next actions?",
            [("b", "ask the bot to suggest next actions"),
             ("m", "add my own next actions"),
             ("d", "done with this scenario")],
            default="d",
        )
        if k == "b":
            suggest_actions(thread, sc, threads)
        elif k == "m":
            add_my_actions(thread, sc)
        else:
            return


def flow_new_situation(threads: list[dict]) -> None:
    clear()
    banner()
    box("NEW SITUATION", [
        "Describe a situation you're in. Include what's at stake, what you've tried,",
        "the real constraints, and anything about the other people involved.",
        "The more specific you are, the more credible the scenarios will be.",
    ], AMBER, "◆")
    situation = ask_multiline("What's the situation?")
    if not situation:
        warn("nothing entered.")
        return
    thread = new_thread(situation)
    save_thread(thread)
    threads.append(thread)

    while True:
        generate_scenarios(thread, threads)
        present_scenarios(thread)
        k = choose(
            "What next?",
            [("r", "rate these scenarios"),
             ("a", "add context and regenerate"),
             ("m", "back to menu (thread is saved)")],
            default="r",
        )
        if k == "a":
            add = ask_multiline("What else should the advisor know?")
            if add:
                thread["context_additions"].append(add)
                log(thread, "context_added")
                save_thread(thread)
            continue
        if k == "m":
            return
        break

    rate_scenarios(thread)
    keepers = [sc for sc in thread["scenarios"] if is_active(sc)]
    print()
    if not keepers:
        say("No scenario was marked both desirable and possible. Thread saved; the advisor will learn from your ratings.", SLATE)
        return
    say(f"{len(keepers)} scenario(s) marked desirable and possible:", LIME + BOLD)
    for sc in keepers:
        say(f"• Scenario {sc['n']} · {sc['title']}", INK, 4)
    for sc in keepers:
        actions_menu(thread, sc, threads)
    say("Thread saved.", LIME)


def flow_active_outcomes(threads: list[dict]) -> None:
    while True:
        clear()
        banner()
        items = [(t, sc) for t in threads for sc in t.get("scenarios", []) if is_active(sc)]
        if not items:
            say("No active outcomes yet. Rate a scenario as desirable and possible to track it here.", SLATE)
            ask("enter to return")
            return
        print(f"  {AMBER}{BOLD}ACTIVE OUTCOMES{RESET}  {DIM}(desirable + possible, not yet resolved){RESET}\n")
        # each situation, with its active outcomes branching off it; resolved ones stay on as dashed, unnumbered branches
        k = 0
        for t in threads:
            active = [sc for sc in t.get("scenarios", []) if is_active(sc)]
            if not active:
                continue
            ended = [sc for sc in t.get("scenarios", []) if sc.get("status") == "resolved"]
            print(f"\n  {AMBER}{BOLD}{thread_title(t)}{RESET}  {SLATE}({t['created'][:10]}, {days_ago(t['created'])}){RESET}")
            for j, sc in enumerate(active + ended):
                last = j == len(active) + len(ended) - 1
                if j >= len(active):
                    print(f"   {SLATE}{'└╌╌' if last else '├╌╌'} {sc['title']}  [{resolution_word(sc)}]{RESET}")
                    continue
                k += 1
                n_open = sum(1 for a in scenario_actions(sc) if is_open(a))
                n_up = len(sc.get("updates", []))
                extra = "  ·  ".join(x for x in (f"{n_open} open action(s)" if n_open else "", f"{n_up} update(s)" if n_up else "") if x)
                print(f"   {SLATE}{'└──' if last else '├──'}{RESET} {AMBER}[{k}]{RESET} {BOLD}{sc['title']}{RESET}"
                      + (f"  {DIM}{extra}{RESET}" if extra else ""))
        print()
        val = ask("number to open, or enter to go back")
        if not val:
            return
        if not val.isdigit() or not (1 <= int(val) <= len(items)):
            warn("not a valid number")
            continue
        t, sc = items[int(val) - 1]
        outcome_menu(t, sc, threads)


def outcome_menu(thread: dict, sc: dict, threads: list[dict]) -> None:
    while True:
        clear()
        banner()
        box("SITUATION", thread["situation"].split("\n"), AMBER, "◆")
        show_scenario_full(sc, thread)
        k = choose(
            "What would you like to do?",
            [("u", "record an update / what actually happened"),
             ("b", "ask the bot to suggest next actions"),
             ("m", "add my own next actions"),
             ("k", "mark actions: won't / may / did"),
             ("t", "view the whole thread"),
             ("x", "back")],
            default="x",
        )
        if k == "u":
            text = ask_multiline("What happened?")
            if not text:
                continue
            r = choose(
                "Does this resolve the outcome?",
                [("h", "yes — it happened (more or less)"),
                 ("p", "partly — some of it happened"),
                 ("d", "no — it didn't happen / no longer possible"),
                 ("o", "still open — this is just a progress note")],
                default="o",
            )
            resolution = {"h": "happened", "p": "partly", "d": "didnt", "o": None}[r]
            sc["updates"].append({"date": today(), "text": text, "resolution": resolution})
            if resolution:
                sc["status"] = "resolved"
            log(thread, "update", n=sc["n"], resolution=resolution)
            save_thread(thread)
            say("saved.", LIME)
            if resolution:
                ask("enter to continue")
                return
        elif k == "b":
            suggest_actions(thread, sc, threads)
            ask("enter to continue")
        elif k == "m":
            add_my_actions(thread, sc)
        elif k == "k":
            if not scenario_actions(sc):
                warn("no actions on this outcome yet")
                time.sleep(1)
                continue
            val = ask(f"number(s) + letter, e.g. '2 d'  {DIM}({MARK_HELP}){RESET}")
            if val:
                mark_actions([(thread, sc, a) for a in scenario_actions(sc)], *parse_mark(val))
        elif k == "t":
            show_thread(thread)
            ask("enter to continue")
        else:
            return


def active_actions(threads: list[dict]) -> list[tuple[dict, dict, dict]]:
    """(thread, scenario, action) for every action on every active outcome, newest thread first."""
    return [(t, sc, a) for t in reversed(threads) for sc in t.get("scenarios", [])
            if is_active(sc) for a in scenario_actions(sc)]


def flow_all_actions(threads: list[dict]) -> None:
    while True:
        clear()
        banner()
        every = active_actions(threads)
        if not every:
            say("No next actions yet. Open an active outcome and ask the bot to suggest some.", SLATE)
            ask("enter to return")
            return
        count = lambda st: sum(1 for e in every if e[2].get("status") == st)
        n_new, n_may = count(None), count("may")
        print(f"  {LIME}{BOLD}NEXT ACTIONS{RESET}  {DIM}({n_new} unmarked  ·  {n_may} may  ·  {count('did')} did  ·  {count('wont')} won't){RESET}")
        opts = []
        if n_new:
            opts.append(("r", f"review the unmarked ones, one at a time  {DIM}({n_new}){RESET}"))
        if n_may:
            opts.append(("m", f"review the \"may\" ones again, one at a time  {DIM}({n_may}){RESET}"))
        opts += [("l", "list them all on one page"), ("x", "back")]
        k = choose("Choose:", opts, default=opts[0][0])
        if k == "x":
            return
        if k == "l" or step_through_actions(threads, "may" if k == "m" else None) == "list":
            list_all_actions(threads)


def step_through_actions(threads: list[dict], status: str | None) -> str | None:
    """One action per screen: mark it won't / may / did, or skip. Returns "list" if the user asks for the full page."""
    entries = [e for e in active_actions(threads) if e[2].get("status") == status]
    done: dict[int, str] = {}  # position -> what the user chose, for the tally and for going back
    i = 0
    while i < len(entries):
        t, sc, a = entries[i]
        clear()
        banner()
        print(f"  {LIME}{BOLD}NEXT ACTION {i + 1} of {len(entries)}{RESET}\n")
        print(f"  {AMBER}{BOLD}{thread_title(t)}{RESET}")
        print(f"  {TEAL}{sc['title']}{RESET}\n")
        say(a["action"], INK, 4)
        print()
        if a.get("why"):
            say(f"why: {a['why']}", SLATE, 4)
        if a.get("when"):
            say(f"when: {a['when']}", SLATE, 4)
        say(("suggested " if "suggested" in a else "yours, added ") + (action_date(a) or "?")
            + (f"  ·  currently: {status_mark(a)}" if a.get("status") else ""), SLATE, 4)
        print()
        say("w = won't   m = may   d = did   enter = skip" + ("   b = back one" if i else "")
            + "   l = list all on one page   x = stop", SLATE)
        key = re.sub(r"[^a-z]", "", ask("").lower())[:1]
        if key in ("w", "m", "d"):
            set_mark(t, sc, a, MARKS[key])
            save_thread(t)
            done[i] = MARKS[key]
            i += 1
        elif key == "":
            done.setdefault(i, "skipped")
            i += 1
        elif key == "b" and i:
            i -= 1
        elif key == "l":
            return "list"
        elif key == "x":
            break
        else:
            warn("w, m, d, enter, b, l or x")
            time.sleep(1)
    tally = {k: sum(1 for v in done.values() if v == k) for k in ("did", "may", "wont", "skipped")}
    print()
    say(f"Reviewed {len(done)} of {len(entries)}:  {tally['did']} did  ·  {tally['may']} may  ·  "
        f"{tally['wont']} won't  ·  {tally['skipped']} skipped", LIME)
    ask("enter to continue")
    return None


def list_all_actions(threads: list[dict]) -> None:
    """Open next actions for every active outcome: thread title, outcome title, actions. Nothing else."""
    show_closed = False
    while True:
        clear()
        banner()
        every = active_actions(threads)
        entries = [e for e in every if show_closed or is_open(e[2])]
        n_did = sum(1 for e in every if e[2].get("status") == "did")
        n_wont = sum(1 for e in every if e[2].get("status") == "wont")
        if not every:
            say("No next actions yet. Open an active outcome and ask the bot to suggest some.", SLATE)
            ask("enter to return")
            return
        tally = f"{len(every) - n_did - n_wont} open  ·  {n_did} did  ·  {n_wont} won't"
        print(f"  {LIME}{BOLD}NEXT ACTIONS{RESET}  {DIM}({tally}{'' if show_closed else ', closed ones hidden'}){RESET}")
        last_t = last_sc = None
        for k, (t, sc, a) in enumerate(entries, 1):
            if t is not last_t:
                print(f"\n  {AMBER}{BOLD}{thread_title(t)}{RESET}")
            if sc is not last_sc:
                print(f"\n    {TEAL}{sc['title']}{RESET}")
            last_t, last_sc = t, sc
            say(f"{k}. {a['action']}", INK if is_open(a) else SLATE, 6)
            if a.get("status"):
                say(status_mark(a), AMBER if a["status"] == "may" else SLATE, 6 + len(str(k)) + 2)
        if not entries:
            print()
            say("Nothing open. Every action is marked did or won't.", LIME)
        print()
        say(f"mark: number(s) + letter, e.g. '3 d' or '4 7 w'   ({MARK_HELP})", SLATE)
        say(f"a number alone opens that outcome  ·  h {'hides' if show_closed else 'shows'} did / won't  ·  enter goes back", SLATE)
        val = ask("")
        if not val:
            return
        nums, key = parse_mark(val)
        if key == "h" and not nums:
            show_closed = not show_closed
        elif len(nums) == 1 and not key and 1 <= nums[0] <= len(entries):
            t, sc, _ = entries[nums[0] - 1]
            outcome_menu(t, sc, threads)
        else:
            mark_actions(entries, nums, key)


def flow_browse_threads(threads: list[dict]) -> None:
    while True:
        clear()
        banner()
        if not threads:
            say("No threads yet.", SLATE)
            ask("enter to return")
            return
        print(f"  {AMBER}{BOLD}ALL THREADS{RESET}\n")
        for k, t in enumerate(threads, 1):
            n_act = sum(1 for sc in t.get("scenarios", []) if is_active(sc))
            n_res = sum(1 for sc in t.get("scenarios", []) if sc.get("status") == "resolved")
            tag = f"{LIME}{n_act} active{RESET}" if n_act else f"{SLATE}0 active{RESET}"
            if n_res:
                tag += f" {SLATE}· {n_res} resolved{RESET}"
            print(f"    {AMBER}[{k}]{RESET} {BOLD}{thread_title(t)}{RESET}")
            print(f"        {SLATE}{t['created'][:10]}{RESET}  {tag}")
        print()
        val = ask("number to open, or enter to go back")
        if not val:
            return
        if not val.isdigit() or not (1 <= int(val) <= len(threads)):
            warn("not a valid number")
            continue
        t = threads[int(val) - 1]
        clear()
        show_thread(t)
        # allow rating/actions on unrated or active scenarios from here too
        opts = [("x", "back")]
        if any(sc.get("liked") is None and sc.get("possible") is None for sc in t.get("scenarios", [])):
            opts.insert(0, ("r", "rate the scenarios in this thread"))
        active = [sc for sc in t.get("scenarios", []) if is_active(sc)]
        if active:
            opts.insert(0, ("o", "open an active outcome from this thread"))
        k = choose("Thread options", opts, default="x")
        if k == "r":
            rate_scenarios(t)
            for sc in [s for s in t["scenarios"] if is_active(s)]:
                actions_menu(t, sc, threads)
        elif k == "o":
            if len(active) == 1:
                outcome_menu(t, active[0], threads)
            else:
                for sc in active:
                    print(f"    {AMBER}[{sc['n']}]{RESET} {sc['title']}")
                v = ask("scenario number")
                match = [sc for sc in active if str(sc["n"]) == v]
                if match:
                    outcome_menu(t, match[0], threads)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def ensure_titles(threads: list[dict]) -> None:
    """One-time: ask the model to title threads saved before threads had titles."""
    untitled = [t for t in threads if not t.get("title") and t.get("situation")]
    if not untitled or not os.environ.get("ANTHROPIC_API_KEY"):
        return
    say(f"giving {len(untitled)} older thread(s) a title…", SLATE)
    body = "\n\n".join(f"THREAD {k}:\n{t['situation'][:1500]}" for k, t in enumerate(untitled, 1))
    try:
        result = call_json([{"type": "text", "text": TITLES_FORMAT}], [{"role": "user", "content": body}], max_tokens=4000)
    except (anthropic.APIError, BackToMenu, KeyboardInterrupt):
        return  # not essential: untitled threads fall back to the start of the situation
    for k, t in enumerate(untitled, 1):
        title = str((result.get("titles") or {}).get(str(k), "")).strip()
        if title:
            t["title"] = title
            log(t, "titled")
            save_thread(t)


def main() -> None:
    threads = load_threads()
    ensure_titles(threads)
    while True:
        try:
            clear()
            banner()
            n_active = sum(1 for t in threads for sc in t.get("scenarios", []) if is_active(sc))
            n_actions = sum(1 for t in threads for sc in t.get("scenarios", []) if is_active(sc)
                            for a in scenario_actions(sc) if is_open(a))
            try:
                k = choose(
                    "Choose:",
                    [("n", "new situation"),
                     ("a", f"active outcomes  {DIM}({n_active}){RESET}"),
                     ("s", f"next actions: review one at a time, or list all  {DIM}({n_actions} open){RESET}"),
                     ("t", f"browse threads  {DIM}({len(threads)}){RESET}"),
                     ("q", "quit")],
                )
            except (BackToMenu, KeyboardInterrupt):
                k = "q"  # EOF or Ctrl-C at the top menu means quit
            if k == "n":
                flow_new_situation(threads)
                ask("enter to return to the menu")
            elif k == "a":
                flow_active_outcomes(threads)
            elif k == "s":
                flow_all_actions(threads)
            elif k == "t":
                flow_browse_threads(threads)
            else:
                print(f"\n  {AMBER}✦ go well.{RESET}\n")
                return
        except (BackToMenu, KeyboardInterrupt):
            print(f"\n  {SLATE}(back to menu){RESET}")
            time.sleep(0.4)
        except anthropic.APIError as e:
            warn(f"API error: {e}")
            try:
                ask("enter to continue")
            except (BackToMenu, KeyboardInterrupt):
                pass


if __name__ == "__main__":
    main()
