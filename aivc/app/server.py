"""FastAPI web app: meeting notes -> screening memo PDF.

Flow:
1. GET  /                    upload page (paste text or upload file)
2. POST /analyze             runs analyze_notes(), shows covered/uncovered list
3. POST /generate/{sid}      collects user answers for unknowns, generates memo
                             via Claude + web_search, renders to PDF
4. GET  /view/{sid}          serves PDF inline for in-browser preview
5. GET  /download/{sid}      serves PDF as attachment
6. GET  /markdown/{sid}      serves raw markdown (for debugging)
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from memo_generator import (
    AnalysisResult,
    analyze_notes,
    generate_memo,
    load_template,
)
from pdf_render import guess_target_company_name, markdown_to_pdf

APP_DIR = Path(__file__).parent
TEMPLATE_PATH = APP_DIR.parent / "deal_memo_template.md"
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PE Deal Memo Generator")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


# In-memory session store. Single-process only. For production, swap to Redis / disk.
SESSIONS: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not TEMPLATE_PATH.exists():
        raise HTTPException(500, f"Deal memo template not found at {TEMPLATE_PATH}")
    return templates.TemplateResponse(request, "upload.html")


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    notes_text: str | None = Form(None),
    notes_file: UploadFile | None = File(None),
):
    notes = ""
    if notes_file is not None and notes_file.filename:
        raw = await notes_file.read()
        notes = raw.decode("utf-8", errors="replace")
    elif notes_text:
        notes = notes_text
    notes = notes.strip()
    if not notes:
        raise HTTPException(400, "No meeting notes provided.")

    template = load_template(TEMPLATE_PATH)

    print(f"[server] /analyze: {len(notes)} chars of notes", file=sys.stderr, flush=True)
    result: AnalysisResult = analyze_notes(notes, template)

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "notes": notes,
        "template": template,
        "analysis": result,
    }

    return templates.TemplateResponse(
        request,
        "unknowns.html",
        {
            "session_id": session_id,
            "covered": result.covered,
            "uncovered": result.uncovered,
        },
    )


@app.post("/generate/{session_id}", response_class=HTMLResponse)
async def generate(request: Request, session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found or expired.")

    form = await request.form()
    user_answers: dict[str, str] = {}
    for u in session["analysis"].uncovered:
        key = f"topic_{u.id}"
        val = (form.get(key) or "").strip()
        if val:
            user_answers[u.topic] = val

    print(
        f"[server] /generate {session_id}: "
        f"{len(user_answers)} of {len(session['analysis'].uncovered)} unknowns filled",
        file=sys.stderr,
        flush=True,
    )

    # Try to surface the target company name from the notes (the analysis
    # almost always extracts it as a covered topic).
    target_name = None
    for c in session["analysis"].covered:
        if "company" in c.topic.lower() and "name" in c.topic.lower():
            target_name = c.answer_from_notes.split("\n")[0].strip(" .,")[:80]
            break

    memo_md = generate_memo(
        notes=session["notes"],
        template=session["template"],
        covered=session["analysis"].covered,
        user_answers=user_answers,
        target_company_name=target_name,
    )

    pdf_path = OUTPUT_DIR / f"{session_id}.pdf"
    md_path = OUTPUT_DIR / f"{session_id}.md"
    md_path.write_text(memo_md)
    markdown_to_pdf(memo_md, pdf_path)

    company_for_filename = guess_target_company_name(memo_md) or "deal_memo"
    safe_filename = (
        "".join(c if c.isalnum() or c in "-_ " else "_" for c in company_for_filename)
        .strip()
        .replace(" ", "_")
    )

    session["memo_md"] = memo_md
    session["pdf_path"] = str(pdf_path)
    session["md_path"] = str(md_path)
    session["filename_stem"] = safe_filename

    print(
        f"[server] /generate {session_id}: wrote {pdf_path} ({pdf_path.stat().st_size} bytes)",
        file=sys.stderr,
        flush=True,
    )

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "session_id": session_id,
            "filename_stem": safe_filename,
        },
    )


@app.get("/view/{session_id}")
async def view_pdf(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None or "pdf_path" not in session:
        raise HTTPException(404, "Memo not found.")
    return FileResponse(session["pdf_path"], media_type="application/pdf")


@app.get("/download/{session_id}")
async def download_pdf(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None or "pdf_path" not in session:
        raise HTTPException(404, "Memo not found.")
    return FileResponse(
        session["pdf_path"],
        media_type="application/pdf",
        filename=f"{session['filename_stem']}.pdf",
    )


@app.get("/markdown/{session_id}", response_class=PlainTextResponse)
async def view_markdown(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None or "memo_md" not in session:
        raise HTTPException(404, "Memo not found.")
    return session["memo_md"]


if __name__ == "__main__":
    import uvicorn

    print(f"[server] Loading template from {TEMPLATE_PATH}", file=sys.stderr)
    print("[server] Starting on http://127.0.0.1:8000", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
