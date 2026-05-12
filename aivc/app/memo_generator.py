"""Memo generation using Claude API with web search.

Two-phase flow:
1. analyze_notes: identify which [Seller]-required topics are covered in the notes
   (uses structured outputs via messages.parse + prompt caching of the template).
2. generate_memo: produce the full memo with web_search tool for [Research] sections
   (uses streaming to avoid HTTP timeouts on long generations).
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from pydantic import BaseModel, Field

MODEL = "claude-opus-4-7"

# System instructions are static — keep them in one place so prompt caching works.
ANALYZE_SYSTEM = """You are a senior private-equity associate analyzing potential investments.

You are given:
1. A deal-memo template (markdown).
2. Meeting notes from a screening call with a target company's seller(s).

Your job: identify which pieces of [Seller]-tagged information the template requires, and check whether the meeting notes provide an answer for each. Be thorough — find every distinct [Seller] information request in the template, not just the obvious ones.

Topics typically include (non-exhaustive): company name and location, founding year, TTM revenue and EBITDA (claimed), addback claims, employee count and headcount split, locations, recurring revenue %, customer count and concentration, service/revenue mix, owner age and exit intent, asking multiple / price, licensing situation, workforce demographics and tech stance, tech systems mentioned (CRM, dispatch, accounting), real estate situation, family on payroll, specific competitors named, top concerns or risks raised in the meeting, identified successor, owner post-close role, etc.

For each topic, you must classify it as either covered (clearly answered by the notes) or uncovered (template asks for it but notes are silent / vague / explicitly unknown). If the notes are vague or partial, lean toward uncovered and ask a sharper question of the user.
"""


GENERATE_SYSTEM = """You are a senior private-equity associate writing an initial screening memo for an investment committee.

You are given:
1. A deal-memo template (markdown) which is the structure you must follow.
2. Meeting notes from the seller (raw and pre-extracted).
3. Additional information provided by the user for topics the notes didn't cover.
4. Today's date.

Output the COMPLETED memo as markdown, following the template's structure exactly. Rules:

1. **Replace bracketed placeholders** like [X], $[X]M, [TARGET COMPANY NAME], [YYYY-MM-DD] with real content. The current date applies to the YAML frontmatter, the version line under the title, and any `Date:` field in the metadata block. Use version `v0.1` (first draft).

2. **For sections / fields tagged [Seller] (or implicit seller-stated facts)**: use the meeting notes and additional user information. Inline-tag with `[Seller]` after the fact, e.g. `TTM Revenue: $19.4M [Seller]`.

3. **For sections tagged [Research]**: use the web_search tool to find authoritative current data — industry TAM, recent comparable transactions, active consolidators in space, industry multiples, fragmentation, etc. Search broadly and cite sources inline as markdown links. Tag with `[Research]` and include the source URL.

4. **For [Inference] / our-judgment fields** (assessments, ratings, recommendations, valuation ranges, risks ranked, AI-fit ratings): synthesize from the available [Seller] and [Research] data. Be specific, not generic. Tag with `[Inference]`.

5. **If a [Seller] field is NOT covered in notes and NOT provided by the user**: mark it as `**Unknown — not addressed in meeting or follow-up**` rather than fabricating. Do not invent seller-stated facts.

6. **Delete the italicized template-instruction guidance** in each section once you've filled it in. Delete the `> **Template instructions** (delete in final version)` block at the top. Delete the `> **For the analyst writing this memo:**` footer at the end.

7. **Keep the YAML frontmatter and the document-control header line under the title.** Populate version (`v0.1`), date (today), classification (`CONFIDENTIAL`), status (`DRAFT`), and the company name.

8. **Keep all markdown structure intact**: section numbers, headers, tables, bullet lists, checkboxes. Use markdown tables with proper pipes and alignment.

9. **For the AI / automation thesis fields specifically**: these are at the core of this firm's strategy. Even when based on inference, be concrete about workflows susceptible to automation, vertical-typical tech stacks, and AI-fit rating with one-sentence rationale.

10. **Recommendation (§1 and §11)**: explicit Pursue / Pass / Hold. Authors' confidence per the footer.

Be specific. Generic boilerplate ("there are risks in this industry") is useless — name the specific risks for THIS target. The goal is a memo a partner can read and decide on.

Output ONLY the completed markdown memo. No preamble, no commentary, no "here is the memo:".
"""


class CoveredTopic(BaseModel):
    """A template topic that the meeting notes answer."""
    id: str = Field(description="Stable snake_case identifier, e.g. 'company_name', 'ttm_revenue', 'top_customer_concentration'")
    topic: str = Field(description="Short human-readable name, e.g. 'TTM Revenue (claimed)'")
    section: str = Field(description="Template section, e.g. '§1 Snapshot', '§3.1 Tech maturity'")
    answer_from_notes: str = Field(description="What the notes say (concise summary, 1-2 sentences)")
    evidence_quote: str = Field(description="Short quoted excerpt from the notes (up to 200 chars)")


class UncoveredTopic(BaseModel):
    """A template topic that the notes don't answer."""
    id: str = Field(description="Stable snake_case identifier")
    topic: str = Field(description="Short human-readable name")
    section: str = Field(description="Template section it belongs to")
    question_for_user: str = Field(description="A direct, single-sentence question to ask the user")
    why_needed: str = Field(description="One sentence on why this info matters for the memo")


class AnalysisResult(BaseModel):
    """Output of phase 1 — what's covered vs not covered in the notes."""
    covered: list[CoveredTopic]
    uncovered: list[UncoveredTopic]


def _client() -> Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running the server."
        )
    return Anthropic()


def analyze_notes(notes: str, template: str) -> AnalysisResult:
    """Identify which [Seller]-required topics are covered in the notes.

    Uses prompt caching on the template (the only large stable input) so repeated
    runs against the same template are cheap.
    """
    print("[memo] Analyzing notes...", file=sys.stderr, flush=True)
    client = _client()

    response = client.messages.parse(
        model=MODEL,
        max_tokens=8192,
        system=[
            {"type": "text", "text": ANALYZE_SYSTEM},
            {
                "type": "text",
                "text": f"DEAL MEMO TEMPLATE:\n\n{template}",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    "MEETING NOTES:\n\n"
                    + notes
                    + "\n\nIdentify every distinct [Seller] information request in the "
                    "template, and for each one determine if the meeting notes "
                    "answer it. Return the structured result."
                ),
            }
        ],
        output_format=AnalysisResult,
    )

    usage = response.usage
    print(
        f"[memo] Analysis complete. covered={len(response.parsed_output.covered)} "
        f"uncovered={len(response.parsed_output.uncovered)} "
        f"tokens(in/out/cache_read)={usage.input_tokens}/{usage.output_tokens}/{usage.cache_read_input_tokens}",
        file=sys.stderr,
        flush=True,
    )
    return response.parsed_output


def generate_memo(
    notes: str,
    template: str,
    covered: list[CoveredTopic],
    user_answers: dict[str, str],
    target_company_name: str | None = None,
) -> str:
    """Generate the full memo. Uses web_search for [Research] sections.

    Streams to avoid SDK HTTP timeouts on long generations (web search +
    big template = potentially many minutes of work).
    """
    print("[memo] Generating memo...", file=sys.stderr, flush=True)
    client = _client()

    covered_lines = "\n".join(
        f"- [{c.section}] {c.topic}: {c.answer_from_notes}" for c in covered
    ) or "(none extracted)"

    user_lines = (
        "\n".join(f"- {topic}: {answer}" for topic, answer in user_answers.items())
        if user_answers
        else "(none provided)"
    )

    today = date.today().isoformat()
    company_hint = (
        f"\nThe target company name appears to be: {target_company_name}.\n"
        if target_company_name
        else ""
    )

    user_message = f"""TODAY'S DATE: {today}
{company_hint}
INFORMATION EXTRACTED FROM THE MEETING NOTES (already structured):
{covered_lines}

ADDITIONAL INFORMATION PROVIDED BY THE USER (filling gaps the notes didn't cover):
{user_lines}

RAW MEETING NOTES (for full context and direct quotes):

{notes}

---

Now write the completed deal memo as markdown, following the template's structure exactly. Use the web_search tool for every [Research] field. Mark unaddressed [Seller] fields as `**Unknown — not addressed in meeting or follow-up**`. Output ONLY the completed memo."""

    with client.messages.stream(
        model=MODEL,
        max_tokens=32000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        system=[
            {"type": "text", "text": GENERATE_SYSTEM},
            {
                "type": "text",
                "text": f"DEAL MEMO TEMPLATE:\n\n{template}",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        final_message = stream.get_final_message()

    print(
        f"[memo] Generation complete. "
        f"tokens(in/out/cache_read)={final_message.usage.input_tokens}/"
        f"{final_message.usage.output_tokens}/"
        f"{final_message.usage.cache_read_input_tokens}",
        file=sys.stderr,
        flush=True,
    )

    text_parts: list[str] = []
    for block in final_message.content:
        if block.type == "text":
            text_parts.append(block.text)
    memo = "".join(text_parts).strip()

    # Drop any leading code-fence wrapper the model sometimes adds.
    if memo.startswith("```"):
        first_nl = memo.find("\n")
        memo = memo[first_nl + 1 :] if first_nl > 0 else memo
        if memo.endswith("```"):
            memo = memo.rsplit("```", 1)[0].rstrip()

    return memo


def load_template(path: Path) -> str:
    return path.read_text()
