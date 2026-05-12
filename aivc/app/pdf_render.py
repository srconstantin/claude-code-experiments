"""Render markdown memo to PDF via markdown -> HTML -> WeasyPrint."""
from __future__ import annotations

import re
from pathlib import Path

import markdown
from weasyprint import HTML, CSS


PDF_CSS = """
@page {
    size: letter;
    margin: 0.8in 0.65in 0.9in 0.65in;
    @top-left {
        content: string(doc-version);
        font-family: 'Helvetica', sans-serif;
        font-size: 8pt;
        color: #555;
    }
    @top-right {
        content: "CONFIDENTIAL";
        font-family: 'Helvetica', sans-serif;
        font-size: 8pt;
        color: #b00;
        font-weight: bold;
    }
    @bottom-center {
        content: string(doc-footer) " — Page " counter(page) " of " counter(pages);
        font-family: 'Helvetica', sans-serif;
        font-size: 8pt;
        color: #555;
    }
}

body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 10pt;
    line-height: 1.42;
    color: #1a1a1a;
}

h1 {
    font-size: 18pt;
    margin-top: 0;
    margin-bottom: 4pt;
    string-set: doc-footer content();
}

h1 + p strong:first-child {
    string-set: doc-version content();
}

h2 {
    font-size: 13pt;
    border-bottom: 1pt solid #888;
    padding-bottom: 2pt;
    margin-top: 18pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}

h3 { font-size: 11pt; margin-top: 12pt; margin-bottom: 4pt; page-break-after: avoid; }
h4 { font-size: 10pt; font-weight: bold; margin-top: 8pt; margin-bottom: 2pt; }

p { margin: 4pt 0; }

table {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0;
    font-size: 9pt;
    page-break-inside: avoid;
}
th, td {
    border: 0.5pt solid #999;
    padding: 4pt 6pt;
    text-align: left;
    vertical-align: top;
}
th { background: #ecedef; font-weight: bold; }

ul, ol { margin: 4pt 0; padding-left: 18pt; }
li { margin: 1pt 0; }

blockquote {
    border-left: 2pt solid #888;
    margin: 8pt 0;
    padding: 4pt 10pt;
    color: #555;
    font-style: italic;
}

code {
    font-family: 'Menlo', 'Courier New', monospace;
    font-size: 8.5pt;
    background: #f0f0f0;
    padding: 0 2pt;
}

pre {
    font-family: 'Menlo', 'Courier New', monospace;
    font-size: 8pt;
    background: #f5f5f5;
    padding: 6pt;
    border-radius: 3pt;
    page-break-inside: avoid;
}

hr { border: 0; border-top: 0.5pt solid #ccc; margin: 12pt 0; }

a { color: #235080; text-decoration: none; }

strong { font-weight: bold; }
em { font-style: italic; }
"""


def _strip_yaml_frontmatter(md: str) -> str:
    """Pandoc-style YAML frontmatter is metadata, not displayable text."""
    if md.startswith("---\n") or md.startswith("---\r\n"):
        end = md.find("\n---", 4)
        if end > 0:
            after = md[end + 4 :].lstrip()
            return after
    return md


def markdown_to_pdf(md_text: str, output_path: Path) -> None:
    """Convert markdown text to PDF and write to output_path."""
    body_md = _strip_yaml_frontmatter(md_text)

    html_body = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br", "attr_list"],
    )

    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Deal Memo</title></head>
<body>{html_body}</body></html>"""

    HTML(string=html_doc).write_pdf(
        str(output_path),
        stylesheets=[CSS(string=PDF_CSS)],
    )


def guess_target_company_name(md_text: str) -> str | None:
    """Extract the target company name from the first H1 line."""
    m = re.search(r"^#\s+(.+?)\s+—\s+", md_text, re.MULTILINE)
    if m:
        candidate = m.group(1).strip()
        if candidate and "[" not in candidate:
            return candidate
    return None
