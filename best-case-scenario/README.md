# Best-Case Scenario

An advisor, powered by Claude Fable 5.1, that runs in your browser or your terminal and that helps you see the realistic, credible, positive ways a situation could turn out. It is deliberately not a cheerleader: a scenario only makes the cut if the model honestly thinks it has a real chance (better than roughly one in ten) given your best effort.

## Run it

There are two versions, a browser app and a terminal app. They share the same threads, memory and prompts, so you can switch between them freely.

**Browser.** From Claude Code, type `/best-case web`. Or from any terminal:

```
python3 best-case-scenario/web.py
```

It starts a small server on this machine only (`http://127.0.0.1:8765`) and opens the app in your browser. Everything is a button. Leave the Terminal window open while you use it; Ctrl-C there stops it.

**Terminal.** From Claude Code, type `/best-case`; it opens the chatbot in a new Terminal window. Or from any terminal:

```
python3 best-case-scenario/bcs.py
```

Needs `ANTHROPIC_API_KEY` in your environment and the `anthropic` Python package.

## What it does

- **New situation.** Describe what you're in. The advisor gives the thread a short title so you can recognize it in lists (threads from before titles existed are titled once, at startup). The advisor may ask a couple of clarifying questions, then gives its honest read plus one to five distinct best-case scenarios, each with a concrete story, why it's credible, what it depends on, and early signs to watch for. You can add context and regenerate.
- **Rate.** For each scenario: do you want it? Do you believe it's possible? Optional comment on why.
- **Next actions.** For any scenario you marked both desirable and possible, ask the bot for concrete next actions, add your own, or both.
- **Active outcomes.** A list of every desirable-and-possible scenario not yet resolved. Open one to record a progress note or what actually happened (happened / partly / didn't), or to add more next actions.
- **Next actions.** Review the actions for every active outcome one at a time: each screen shows the thread's title, the outcome's title and one action, and you press `w` (won't do it), `m` (may do it), `d` (did it) or enter to skip (`b` goes back one, `x` stops). "May" ones can be reviewed again later. Or choose **list them all on one page** (also `l` from any review screen) to see every open action at once and mark by number, e.g. `3 d` or `4 7 w`; `o` reopens one and `h` shows the hidden "won't" and "did" ones. Marked actions stay in the thread file, and the advisor sees every mark. You can also mark actions from inside an outcome.
- **Browse threads.** Read any past thread in full.

## Memory

Every thread is a JSON file in `threads/`. On each model call the app sends the advisor a digest of all past threads (newest first, within a size budget): situations, scenarios, your ratings and comments, next actions, and outcomes. That is how it calibrates to what you find credible and what has actually happened.

Everything is dated. Each thread records when it was created, and every event in it (context added, ratings, actions suggested or added, updates) is timestamped in the thread's `log`. On every call the advisor is told today's date, when the current thread started, and the date of each piece of context, action and update, so it can reason about elapsed time and give timing as real dates.

`profile.md` is sent as background on every call. Copy `profile.example.md` to `profile.md` and edit it to tell the advisor about yourself. Both `profile.md` and the thread files are gitignored, so they stay on your machine.

## Files

| file | purpose |
|---|---|
| `bcs.py` | the terminal app, and the shared logic (prompts, memory, storage) |
| `web.py` | the browser version's local server |
| `webapp.html` | the browser version's page |
| `profile.md` | who you are, sent to the model every time (gitignored; start from `profile.example.md`) |
| `threads/` | saved threads, one JSON per situation (gitignored) |
| `launch.sh` | opens the terminal app in a new Terminal window (used by `/best-case`) |
| `launch-web.sh` | starts the browser version (used by `/best-case web`) |
