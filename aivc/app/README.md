# PE Deal Memo Generator (web app)

Generates a structured PE deal-memo PDF from raw meeting notes, using Claude API + web search.

## Flow

1. **Upload** meeting notes (paste text or upload a `.txt` / `.md` file).
2. **Review & fill unknowns.** Claude inspects the notes against `deal_memo_template.md` and identifies which `[Seller]`-tagged template topics the notes cover vs. don't. The app presents a per-topic text box for each uncovered item — fill what you can, leave blank for *Unknown*.
3. **Generate.** Claude (Opus 4.7) writes the full memo using:
   - **Notes + your follow-up answers** for `[Seller]` fields
   - **`web_search` tool** for `[Research]` fields (TAM, recent comparable transactions, consolidator activity, multiples)
   - **Synthesis** for `[Inference]` fields (risk scoring, valuation range, recommendation, AI-fit ratings)
   - Anything still unaddressed is marked `**Unknown — not addressed in meeting or follow-up**`.
4. **Preview & download** the memo as a PDF (in-browser preview + download button).

## Setup

### 1. Install system dependencies for WeasyPrint

macOS:

```sh
brew install pango cairo gdk-pixbuf libffi
```

Ubuntu/Debian:

```sh
sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 libffi-dev
```

### 2. Python environment

```sh
cd aivc/app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. API key

```sh
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Run

```sh
python server.py
```

Open <http://127.0.0.1:8000>.

## Architecture

- **`server.py`** — FastAPI app. Three routes: `/`, `/analyze`, `/generate/{session_id}`, plus `/view`, `/download`, `/markdown` for the PDF artifact. Single-process in-memory session store keyed by UUID.
- **`memo_generator.py`** — two Claude calls:
  - **`analyze_notes`** — uses `client.messages.parse()` with a Pydantic schema (`AnalysisResult` → `covered[]` + `uncovered[]`) for structured output. Caches the template via `cache_control` so repeated runs are cheap.
  - **`generate_memo`** — uses `client.messages.stream()` with the `web_search_20260209` tool, adaptive thinking, and `effort: "high"`. Streams to avoid HTTP timeouts on long generations. Template is cached.
- **`pdf_render.py`** — markdown → HTML (`markdown` library) → PDF (WeasyPrint) with CSS that puts the doc-control version line in the page header and `CONFIDENTIAL` watermarks on every page footer.
- **`templates/`** — three Jinja2 pages: upload, unknowns, result.
- **`static/style.css`** — single stylesheet for all pages.
- **`outputs/`** — generated `.pdf` and `.md` files keyed by session UUID.

The template itself lives at `../deal_memo_template.md` (one level up).

## Expected timing

- **Analyze step:** ~15–30 seconds. Reads template (~16 KB) + notes; returns ~10–30 covered / uncovered topics.
- **Generate step:** ~60–180 seconds. The web search tool makes multiple internal queries for industry data (TAM, multiples, comparable transactions); generation produces ~10–20 KB of markdown. Tab must stay open.

## Notes / known limits

- **Single-process only.** Sessions are in-memory dict, lost on restart. Production would need Redis/disk + a queue for the long-running generation step.
- **No auth.** Run locally or behind your own VPN/auth proxy.
- **PDF artifacts persist on disk** in `outputs/`. Wipe them as needed.
- **Web search citations.** Claude will include them inline as markdown links — they pass through to the PDF.
- **Cost per memo:** the template is cached so repeat runs are cheap (~$0.10–$0.20 per memo at Opus 4.7 prices, dominated by the generation step's output tokens and the web search costs).
