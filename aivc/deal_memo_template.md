---
title: "[TARGET COMPANY NAME] — Initial Screening Memo"
version: "[X.X]"
date: "[YYYY-MM-DD]"
classification: "CONFIDENTIAL"
status: "[DRAFT / FINAL]"
---

# [TARGET COMPANY NAME] — Initial Screening Memo

**v[X.X] · [YYYY-MM-DD] · CONFIDENTIAL · [DRAFT / FINAL]**

> **Template instructions** (delete in final version):
> This is an *initial screening* memo, written after the first meeting with a target and basic online/industry research. It is **pre-LOI, pre-QoE, pre-financial-diligence**. We do not yet have access to audited financials, tax returns, customer lists, employee rosters, or third-party diligence reports — only what the seller showed us in the meeting + what public/industry sources can verify.
>
> **The job of this memo is to recommend a binary decision: pursue or pass.** "Pursue" means we move to LOI, engage QoE, and commit real diligence dollars. "Pass" means we drop it. A third option is "wait and revisit" if we want to see specific things from the seller before deciding.
>
> **This firm's investment thesis specifically targets AI / automation transformation of acquired companies and synergistic use of data across the portfolio.** A target's *AI-transformability* is a core screening dimension alongside traditional financials — it is treated as a primary value-creation lever, not an afterthought.
>
> Every claim in this memo should be tagged for source: **[Seller]** = seller stated, unverified. **[Research]** = independently verified from public/industry sources. **[Inference]** = our own analysis or judgment. Delete this guidance before circulating. Target length: 6–10 pages.
>
> **Document control:** Update the YAML frontmatter and the version line above on every revision (e.g., `v0.1` → first draft, `v0.2` → after partner edits, `v1.0` → IC-ready, etc.). On PDF export, the version line should be applied as a page header on every page, and "CONFIDENTIAL — [TARGET COMPANY NAME]" as a footer watermark. Pandoc reads the YAML frontmatter and can drive headers/footers automatically — recommended pipeline: `pandoc memo.md -o memo.pdf --template=pe-memo.tex` with the firm's standard template.

**Prepared by:** [Author Name(s)] · [Title]
**Date:** [YYYY-MM-DD]
**Memo type:** Initial screen (pre-LOI, pre-diligence)
**Meeting date / format:** [Date, in-person / video, location, attendees]
**Source materials:** Meeting notes ([file]), seller deck ([file]), independent research, [other]

---

## 1. Snapshot & Recommendation

*Half page. Read-it-and-decide format.*

**The ask:** ☐ Proceed to LOI at $[X]–$[Y]M EV range ☐ Pass ☐ Hold; request specific info before deciding

**Target one-liner:** [e.g., "Regional HVAC + plumbing contractor, $19M revenue, 63-yr-old owner retiring, 17K recurring maintenance member base."]

**Quick snapshot** (all [Seller] unless noted):

| Company / location | [Name, City State] |
| Industry | [vertical] |
| Founded | [Year] |
| TTM Revenue (claimed) | $[X]M |
| TTM EBITDA (claimed) | $[X]M ([X]%) |
| Adjusted EBITDA (claimed) | $[X]M after $[X]M in addbacks |
| Employees | ~[#] |
| Locations | [#] |
| Recurring rev share | [X]% [Seller] |
| Owner age / intent | [X] / [retiring, partial exit, etc.] |
| Asking multiple | [X.X]x adj. EBITDA = ~$[X]M EV |
| **AI-transformability rating** | ☐ High ☐ Medium ☐ Low — *see §3* |

**Top 3 reasons this is interesting:**
- [Reason 1, e.g., real recurring revenue base in a fragmented vertical we already know]
- [Reason 2]
- [Reason 3 — should typically reference AI / data thesis fit if relevant]

**Top 3 reasons it might not be:**
- [Concern 1, e.g., owner = brand, no clear successor identified]
- [Concern 2]
- [Concern 3]

**Recommendation:** [Pursue / Pass / Hold for more info] — [one-sentence rationale].

---

## 2. Business Overview

*1 page. What we learned in the meeting about how the company makes money. Mostly [Seller] sourced — flag what's been independently confirmed.*

### 2.1 What they do
*Brief description. Service lines / product mix / customer types as understood from the meeting.*

| Revenue line | Claimed % of revenue | Source |
|---|---|---|
| [Line 1] | [X]% | [Seller] |
| [Line 2] | [X]% | [Seller] |

### 2.2 How they operate
- Locations, footprint, fleet/equipment as described: [summary]
- Service / production model: [e.g., route-based residential service, project-based commercial, etc.]
- Headcount split (rough): [#] field, [#] office, [#] sales

### 2.3 Customers (what we know so far)
- Total customer count claimed: [#] [Seller]
- Recurring revenue %: [X]% [Seller]
- Concentration we've been told about: [e.g., "top 3 customers = ~40% of revenue per seller"]
- Customer detail received: ☐ Names of top customers ☐ Contract terms ☐ Tenure / churn data ☐ None yet
- **Overlap with existing portfolio companies** [Inference]: [e.g., "appears to serve same QSR / multi-unit retail customers as our signage and pest-control platforms — cross-sell potential"]

### 2.4 Licensing / regulatory
*This category routinely kills deals — flag what we know now.*
- License(s) required to operate: [list]
- Current license holder: [Owner / employee / company entity]
- Transferability on sale: [Known transferable / Known non-transferable / Unknown — to verify]
- Regulatory constraints on AI use in this vertical [Inference]: [e.g., "HIPAA-regulated; AI in clinical workflows requires careful design" or "no material constraint"]

---

## 3. Tech Stack, Data, and AI-Transformability

*Half page. The dedicated section for our firm's core thesis. At the screening stage, we won't have a system inventory or workflow-level data — only what came up in the meeting, what's typical for the vertical, and what we can infer. Detailed system mapping is a post-LOI workstream. This section is an impressionistic read, not an audit.*

### 3.1 Tech / data maturity (initial read)
*One short paragraph. What was observed in the meeting + what's industry-typical for this vertical and company size.*

- **Systems mentioned in meeting** [Seller]: [e.g., "uses ServiceTitan for dispatch" / "all on QuickBooks Desktop" / "did not discuss"]
- **Inferred maturity given vertical and size** [Inference]: [e.g., "HVAC contractors of this size typically use a field service management tool — likely ServiceTitan or FieldEdge"]
- **Overall data infrastructure read:** ☐ Modern cloud stack — data is portable and integrable ☐ Mixed — some systems, some manual ☐ Mostly manual / paper — significant lift to digitize ☐ Unknown — to be confirmed in diligence

### 3.2 Where AI is most likely to add value (initial read)
*Narrative, 3–5 bullets. Based on this vertical's typical labor-cost structure and what the company-specific notes flagged. Don't try to size in FTEs or dollars at the screening stage — that's a post-LOI exercise.*

- *[e.g., Customer intake / scheduling — owner mentioned having 3 dispatchers, after-hours call handling is manual. Voice-agent + scheduling automation is a credible compression target.]*
- *[e.g., Quoting / estimating — owner said he does all major commercial quotes himself. Historical-pricing models + vision-from-site-photos would replicate his judgment and de-risk the succession.]*
- *[e.g., Field-tech productivity — utilization at ~71% per seller; portfolio benchmark is ~82%. AI-assisted dispatching + technician knowledge tools have been the highest-ROI investment at our other portcos in this vertical.]*

### 3.3 Workforce receptivity to AI / automation (initial read)
*Briefly assess — this is a real implementation risk in traditional industries.*
- Workforce demographic (impression from meeting / typical for vertical): [skews older / younger / mixed]
- Owner / management stance on technology (from meeting): [enthusiastic / cautious / dismissive / not discussed]
- Union or labor-agreement constraints: [yes / no / N/A]

### 3.4 AI-thesis fit summary
- **AI-transformability rating:** ☐ High ☐ Medium ☐ Low
- **Implementation difficulty (cost / time to capture value):** ☐ Low ☐ Moderate ☐ Heavy lift
- **One-sentence rationale:**

*Cross-portfolio data synergy is treated as a value-creation lever in §8.2, not as a screening-stage read. Detailed system inventory and workflow-level labor mapping are post-LOI workstreams — see §9.2 and §11.2.*

---

## 4. Market Opportunity

*1 page. Mostly [Research] sourced — this is where the desk research lives.*

### 4.1 Market size & growth
- **TAM (industry, global / US):** $[X]B [Research — source]
- **SAM (geographic + product segment served):** $[X]M [Research / Inference]
- **Growth (last 5 yrs / projected):** [X]% CAGR [Research — source]
- **Cyclicality:** [highly / mildly / counter / non-cyclical]

### 4.2 Industry tailwinds / headwinds
- **Tailwinds:** [e.g., aging owner cohort drives sell-side supply; demographic demand growth; regulatory tightening favors scaled players; pricing power in hard market]
- **Headwinds:** [e.g., trade labor shortage — *also creates urgency for AI augmentation*; reimbursement pressure; commodity input volatility; substitute technologies]

### 4.3 Rollup / consolidation landscape
- **Active consolidators in space:** [names, recent acquisitions, multiples paid where known]
- **Fragmentation:** [#] estimated independent firms in target's region [Research]
- **Implied add-on pipeline:** [order of magnitude — "30+ targets in 200-mile radius at $0.5–2M EBITDA"]
- **AI-thesis-specific competitive dynamics:** [are other PE players in this vertical pursuing similar AI/data thesis, or is the field still operationally traditional?]

---

## 5. Competitive Position

*Half to one page. Initial read — to be pressure-tested in diligence.*

### 5.1 Who they compete against
| Competitor | Type | Geography overlap | Relative size | Source |
|---|---|---|---|---|
| [Name] | [Local / Regional / National] | [Direct / Partial / None] | [Larger / Similar / Smaller] | [Research / Seller] |

### 5.2 Where the seller claims they win
- *[e.g., only contractor in service area with both HVAC and plumbing licenses — verifiable from state contractor board]*
- *[e.g., largest recurring maintenance member base in market — seller-claimed, hard to verify pre-LOI]*

### 5.3 Where they could lose
*Honest [Inference] — the failure modes we want to test:*
- *[e.g., if a national consolidator (Caliber, Aire Serv, etc.) enters the market with capital and brand, the target's relationship advantage may erode]*
- *[e.g., if the owner is the rainmaker, his exit could trigger top-account churn]*
- *[e.g., if a competitor adopts AI-enabled scheduling / quoting before we integrate ours, our window narrows]*

---

## 6. Management & Transition

*Half to one page. The single most important section in lower-MM screening — people drive outcomes here.*

### 6.1 Key people met or referenced
For each:
- **Name, role, age, tenure**
- **Background as described**
- **Day-1 post-close intent:** [exit / consulting / employment / rollover equity]
- **Initial read:** [Inference — strength, credibility, energy, fit]
- **Stance on technology / change:** [Inference — receptive / neutral / resistant — important for AI rollout]

### 6.2 Operator / successor question
**The single biggest question in a lower-MM deal:** *Who runs the company on day 1 post-close, and who runs it in year 3?*

- Day-1 operator candidate: [Internal name / external hire required / our platform CEO absorbs]
- Long-term operator: [same / different]
- Key man risk severity: ☐ High ☐ Medium ☐ Low
- Replacement cost / timeline if needed: $[X]K/yr, [X] months to hire [Inference]
- **Operator profile fit with AI-transformation roadmap:** [Are they someone who can execute on technology-driven change, or is that going to require additional management bandwidth from us?]

### 6.3 Cultural / fit notes
*Subjective but matters. From the meeting — does this person work with us for the next 2–5 years?*

---

## 7. Preliminary Financial Read

*1 page. Everything here is [Seller] unless flagged. We're not pretending to have the real numbers yet — this section frames what we'd want to test in QoE.*

### 7.1 Seller-stated financials

| ($M) | FY-1 | TTM | Source / confidence |
|---|---|---|---|
| Revenue | | | [Seller — tax returns / internal P&L / verbal] |
| Reported EBITDA | | | |
| Adjusted EBITDA (per seller) | | | |
| Margin % | | | |
| Recurring revenue % | | | |
| Gross margin (if disclosed) | | | |
| Total labor as % of revenue | | | [Critical input to AI-thesis sizing] |

**Confidence in numbers:** ☐ High (saw tax returns + P&Ls) ☐ Medium (saw a CIM / deck) ☐ Low (verbal only)

### 7.2 Addback claims (to be tested in QoE)
*The seller's claimed addbacks, as we heard them. Flag the ones that look standard vs. aggressive — but do not pre-judge the QoE outcome.*

| Addback claimed | $ amount (annual) | Category | Initial read |
|---|---|---|---|
| Owner above-market comp | $[X] | Comp | [Standard / Need to verify market replacement rate] |
| Spouse / family on payroll | $[X] | Comp | [Standard if non-working] |
| Personal vehicles / boat | $[X] | Non-business | [Standard] |
| Country club / personal travel / entertainment | $[X] | Non-business | [Standard] |
| Personal insurance (life, health) | $[X] | Non-business | [Standard] |
| Above-market related-party rent | $[X] | Real estate | [To normalize at market] |
| "One-time" legal / professional / one-off | $[X] | One-time | [Aggressive — usually QoE drops some] |
| Other | $[X] | | |
| **Seller total addbacks** | **$[X]** | | |

**Initial view of "real" normalized EBITDA range:** $[X]–$[Y]M (vs. seller's claimed $[Z]M).

### 7.3 Key financial questions for diligence
*The 3–6 things we don't know that most matter:*
- [e.g., How much of revenue is genuinely recurring vs. one-time project work mis-categorized?]
- [e.g., Top 10 customer concentration — they implied "low" but won't share names yet]
- [e.g., Maintenance vs. growth capex split — fleet age suggests heavy refresh needed]
- [e.g., Labor cost breakdown by function — what's customer-service vs. dispatch vs. field service?]
- [e.g., Software / IT spend today — what's the current technology cost baseline?]

---

## 8. Preliminary Investment Thesis

*One page. The case for spending real diligence dollars on this. Be specific.*

### 8.1 One-paragraph thesis
*Single paragraph. Why is this worth pursuing, in concrete terms, beyond "it looks interesting"?*

### 8.2 Preliminary value-creation levers
*Rough sketch — don't quantify precisely until we have real numbers. **AI / data levers are first-class, not afterthoughts.***

**Traditional levers:**
- **Organic growth:** [e.g., underpriced maintenance plans, geographic infill, cross-sell of plumbing into HVAC base]
- **Margin improvement (traditional):** [e.g., dispatch optimization, technician utilization, procurement scale]
- **Add-on activity (if platform thesis):** [#] credible targets identified in region; estimated $[X]M EBITDA addable in 3 yrs
- **Multiple expansion at exit:** [moving from a $[X]M-EBITDA single-region asset to a $[X]M-EBITDA multi-region asset]

**AI / automation levers (firm-specific thesis):**
- **Back-office workflow automation:** AI for AR/AP, scheduling, intake, document processing. Estimated payroll cost compression: $[X]M annual → ~$[X]M annual EBITDA, full-capture by year [X]. Implementation: leverage our platform AI tooling already deployed at [PortCo X, Y].
- **Customer-facing AI augmentation:** Voice agents for after-hours intake, AI-assisted quoting, scheduling automation. Revenue / capacity uplift: [X]% capacity increase without proportional headcount.
- **Vision / inspection automation:** [If applicable to vertical — quality, completion confirmation, damage assessment, etc.]
- **Pricing optimization through portfolio data:** Cross-portfolio quote benchmarking exposes underpricing; expected [X]–[Y]% gross margin uplift.
- **Field-service knowledge replication:** AI assistants trained on best-tech SOPs / call notes to bring lower-quartile productivity toward the median.

**Cross-portfolio data synergies:**
- **Shared customer identity:** [If target serves customers already in portfolio, cross-sell opportunity sized at $[X]M.]
- **Procurement consolidation:** Combined vendor spend across portfolio of $[X]M; estimated 3–8% rationalization = $[X]M COGS savings.
- **Operational benchmarking:** Cross-portfolio metrics on utilization, completion times, comp/labor productivity inform best-practice transfer at this target.
- **Centralized back-office:** Shared services (HR, AP, IT, marketing) across portfolio; estimated SG&A reduction $[X]M.
- **Talent / workforce:** Portfolio-wide hiring and movement of high-performers across geographies.

**Total preliminary value-creation estimate** [Inference, rough]: $[X]–$[Y]M of EBITDA uplift over 3–4 years from AI + portfolio integration levers, on top of $[X]–$[Y]M from traditional organic + add-on growth.

### 8.3 Why now / why us
- **Why now:** [Specific seller motivation; specific industry window]
- **Why us:** Beyond capital, we bring (a) our AI integration playbook proven at [other portcos], (b) shared data infrastructure that this target gains access to on day 1, (c) cross-portfolio customer relationships, (d) our platform-vertical operating expertise. **A non-AI-focused PE bidder gets a smaller value-creation case from this asset than we do — this is a real bidding advantage if we're disciplined about price.**

---

## 9. Key Risks & Open Questions

*1 page. At this stage, "risks" are mostly "things we don't yet know but need to understand." Frame as questions to answer in diligence, not as fully-formed risk assessments.*

### 9.1 Top risks (preliminary)

For each:
- **Risk description**
- **What we don't know yet**
- **What would resolve it** (specific diligence item)
- **Deal-killer potential:** ☐ Likely fatal if confirmed ☐ Price-affecting ☐ Manageable

Categories to consider:
- **Customer concentration / quality** — true top-customer mix unknown
- **Key person dependency** — owner = ?, license holder = ?, top rainmaker = ?
- **License / regulatory transfer** — pathway confirmed?
- **Workforce / retention** — will key associates stay through transition?
- **Workforce receptivity to AI / automation** — will the field workforce adopt new tools, or sabotage / resign?
- **Data quality and digitization gap** — if records are paper / unstructured, the AI thesis cost goes up significantly
- **Technology integration cost** — replacing legacy systems (e.g., a 20-year-old proprietary ERP) is expensive and slow
- **Industry-specific AI regulation** — HIPAA, GLBA, state-licensure-board restrictions on automated decisions, etc.
- **Real estate** — owned vs. leased, assignability, deferred maintenance
- **Customer / revenue quality** — recurring claims unverified
- **Industry / cyclical** — exposure to specific macro or regulatory event
- **Competitive threats** — national consolidator entering geography; competitor adopting AI before we integrate ours
- **Deal-specific** — estate sale, partnership conflict, divorce, key associate departure already announced, etc.

### 9.2 Things we'd need to see before going under LOI
*Specific list of items we'd request from the seller / broker in next round.*
1. [e.g., 3 years of tax returns + monthly financials]
2. [e.g., Customer list with revenue by account, top 20]
3. [e.g., Employee roster with comp, tenure, and function]
4. [e.g., License documentation and transferability research]
5. [e.g., Existing debt + personal-guarantee summary]
6. [e.g., Any prior LOIs received]
7. **[AI-thesis-specific] Tech stack inventory** — current systems, vendors, contracts, annual IT spend
8. **[AI-thesis-specific] Data sample** — anonymized customer / job / dispatch records sufficient to assess data quality and integration difficulty
9. **[AI-thesis-specific] Workforce composition by function** — to size labor-cost exposure to automation

---

## 10. Preliminary Valuation Read

*Half to one page. Not a model. A range and a structural sketch.*

### 10.1 Valuation range (preliminary)
| Scenario | Adj. EBITDA assumption | Multiple range | Implied EV |
|---|---|---|---|
| Seller's case | $[X]M | [X.X]x asked | $[X]M |
| Our base case (if EBITDA holds in QoE) | $[X]M | [X.X]–[X.X]x | $[X]–$[Y]M |
| Our downside case (if QoE haircuts EBITDA) | $[X]M | [X.X]–[X.X]x | $[X]–$[Y]M |

**Walk-away price (preliminary):** $[X]M — above this, the deal stops being interesting at the risk-adjusted return required.

**Note:** Because our AI / data thesis adds value creation that other bidders cannot replicate, we can sometimes outbid more traditional PE buyers without overpaying *on our return math*. We should still hold to our walk-away price — paying for value we plan to create ourselves is the classic rollup mistake.

### 10.2 Comparable transactions
*What recent comps in the space suggest [Research]:*

| Transaction | Date | Approx. size | Multiple paid | Source |
|---|---|---|---|---|
| [Target/Buyer] | [Date] | $[X]M EBITDA | [X.X]x | [Press release / industry trade pub] |

### 10.3 Structural preferences (preliminary)
*Not a fixed cap stack — directional preferences we'd seek to negotiate:*
- **Rollover equity required:** [Yes — at least [X]%, prefer [X]%] — important for alignment
- **Earnout:** [appropriate given X risk]
- **Seller note:** [yes / no / contingent]
- **Real estate:** [include / separate sale-leaseback / continued lease]
- **Owner post-close role:** [target term — e.g., 12-mo consulting, no longer]
- **Technology / AI integration reserve:** budget $[X]K–$[Y]K in our capex plan for year-1 AI/data integration — to be funded from sponsor equity, not added to purchase price

### 10.4 Rough returns check (back-of-envelope)
*Quick sanity check on whether this is in the box of what the fund needs to return.*
- Entry: ~$[X]M at [X.X]x EBITDA
- Year-5 EBITDA: ~$[X]M (organic + add-on + AI/data uplift)
  - of which: organic $[X]M / add-ons $[X]M / **AI & data levers $[X]M**
- Exit at [X.X]x: ~$[X]M EV
- After debt paydown, sponsor equity proceeds: ~$[X]M
- On sponsor check of ~$[X]M: ~[X.X]x MOIC, [X]% IRR
- **Is this in the box?** ☐ Yes ☐ Marginal ☐ No
- **Sensitivity to AI-thesis success:** if AI/data levers deliver 50% of plan, returns become [X.X]x / [X]%. If they deliver 0%, returns become [X.X]x / [X]%. *This sensitivity is the firm's core underwriting question.*

---

## 11. Next Steps

*Half page. Concrete actions and decision.*

### 11.1 Recommended path

**☐ Pursue to LOI**
- Send IOI / preliminary letter at $[X]–$[Y]M range
- Request documents (see §9.2), including AI-thesis-specific items (tech stack, data sample, workforce-by-function)
- Engage QoE [firm] if seller cooperative — estimated cost $[X]K, [X] weeks
- Engage our internal AI / data team for a parallel **AI-readiness assessment** during LOI period — estimated [X] internal-hours
- Internal calendar: target LOI within [X] weeks, LOI exclusivity [X] days

**☐ Hold — request info first**
- Specific items needed: [list, max 3–5]
- Re-screen after receipt; target decision in [X] weeks

**☐ Pass**
- Reasons: [specific, listed]
- Send polite decline to seller / broker; preserve relationship for future deal flow
- Disposition: ☐ Could revisit if [conditions change] ☐ Permanent pass

### 11.2 Resource estimate if we pursue
- Internal time: [X] hours / [X] weeks of associate + [X] of partner + [X] of AI / data team
- External diligence cost (if we go to LOI): $[X]K–$[X]K for QoE + legal + insurance
- Estimated year-1 AI / data integration cost (post-close, separate budget): $[X]K–$[X]K
- Expected timeline from LOI to close: [X] months

### 11.3 Sourcing & relationship notes
- Broker: [Name, firm]; prior deals with us: [Yes/No, # and outcomes]
- Other bidders we believe are in process: [#, names if known]
- Seller's perceived urgency: [High / Medium / Low — based on signals]
- Our positioning vs. competing bidders: [Why might seller pick us over others? Note: AI / data integration story can be a relationship win with sellers who want their company to be future-proof, not a relationship loss with sellers worried about "PE strip-mining" their business.]

---

*End of memo. Next decision point: [date / IC meeting / partner approval].*

> **For the analyst writing this memo:**
> - This is a *screen*, not an *IC memo*. If a section can't be filled out with reasonable confidence from the meeting + research, write "Unknown — to be requested" rather than guessing.
> - Tag every claim with **[Seller]**, **[Research]**, or **[Inference]** so the partner reading it knows what's verified vs. asserted.
> - Section 3 (Tech Stack, Data, and AI-Transformability) is core to our thesis. Even with limited information, make a defensible initial read — what was observed in the meeting, what the seller said about their systems, what's a reasonable inference for this vertical and company size.
> - The recommendation must be binary and actionable. "Interesting, more to learn" is not a recommendation — it's a stall. If you want more info, say so explicitly in §11.1 and list what.
> - Resist the temptation to do a full IC-memo-quality job here. The point of a screen is to decide whether the deal is worth the QoE / legal / AI-readiness / partner-hour investment. Spend 6–10 hours, not 60.
