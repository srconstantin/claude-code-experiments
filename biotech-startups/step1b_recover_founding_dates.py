#!/usr/bin/env python3
"""
Step 1B: Recover founding dates from companies where Step 1 didn't find them.

Uses expanded regex patterns and also looks at the full filing text more carefully.
"""

import requests
import csv
import time
import re
from pathlib import Path
from datetime import datetime

HEADERS = {
    "User-Agent": "Academic Research sarah.constantin@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

RATE_LIMIT_DELAY = 0.15


def get_company_filings(cik: str) -> list[dict]:
    """Get S-1 filings for a company."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    time.sleep(RATE_LIMIT_DELAY)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return []

    data = response.json()
    filings = []
    recent = data.get("filings", {}).get("recent", {})

    if not recent:
        return []

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form in ["S-1", "S-1/A"]:
            filings.append({
                "form": form,
                "date": dates[i] if i < len(dates) else None,
                "accession": accessions[i] if i < len(accessions) else None,
                "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
            })

    return filings


def download_filing_text(cik: str, accession: str, primary_doc: str) -> str | None:
    """Download the full text of an S-1 filing."""
    accession_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}/{primary_doc}"

    time.sleep(RATE_LIMIT_DELAY)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return None

    return response.text


def extract_founding_date_expanded(text: str) -> tuple[str | None, str | None]:
    """
    Extract founding date using expanded patterns.
    Returns (date_string, pattern_matched) for debugging.
    """
    # Clean up the text - remove HTML tags for better matching
    text_clean = re.sub(r'<[^>]+>', ' ', text)
    text_clean = re.sub(r'\s+', ' ', text_clean)

    # Expanded patterns - ordered from most specific to least
    patterns = [
        # Pattern 1: "incorporated in [State] in [Month] [Year]"
        (r'(?:incorporated|organized|formed)\s+(?:in\s+)?(?:the\s+)?(?:State\s+of\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:in|on)\s+([A-Z][a-z]+\s+\d{4})', 'state_month_year'),

        # Pattern 2: "incorporated in [Month] [Year]"
        (r'(?:incorporated|organized|formed)\s+in\s+([A-Z][a-z]+\s+\d{4})', 'in_month_year'),

        # Pattern 3: "incorporated on [Month] [Day], [Year]"
        (r'(?:incorporated|organized|formed)\s+on\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})', 'on_full_date'),

        # Pattern 4: "incorporated ... in [Year]" (with state in between)
        (r'(?:incorporated|organized|formed)\s+(?:under\s+the\s+laws\s+of\s+)?[A-Za-z\s]+\s+in\s+(\d{4})', 'state_year'),

        # Pattern 5: "a [State] corporation incorporated in [Year]"
        (r'[A-Z][a-z]+\s+corporation\s+(?:that\s+was\s+)?incorporated\s+in\s+(\d{4})', 'corp_year'),

        # Pattern 6: "founded in [Year]"
        (r'(?:was\s+)?founded\s+in\s+(\d{4})', 'founded_year'),

        # Pattern 7: "established in [Year]"
        (r'(?:was\s+)?established\s+in\s+(\d{4})', 'established_year'),

        # Pattern 8: "since our founding in [Year]"
        (r'since\s+(?:our\s+)?(?:founding|formation|incorporation)\s+in\s+(\d{4})', 'since_founding'),

        # Pattern 9: "commenced operations in [Year]"
        (r'commenced\s+(?:our\s+)?operations\s+in\s+(\d{4})', 'commenced_year'),

        # Pattern 10: "We were incorporated in [Year]"
        (r'[Ww]e\s+were\s+incorporated\s+in\s+(\d{4})', 'we_incorporated'),

        # Pattern 11: "The Company was incorporated in [Year]"
        (r'[Tt]he\s+[Cc]ompany\s+was\s+incorporated\s+in\s+(\d{4})', 'company_incorporated'),

        # Pattern 12: "incorporated in [State] on [Date]"
        (r'incorporated\s+in\s+[A-Z][a-z]+\s+on\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})', 'state_on_date'),

        # Pattern 13: Look in "History" section - "In [Year], we were founded"
        (r'[Ii]n\s+(\d{4}),?\s+(?:we\s+were\s+|the\s+[Cc]ompany\s+was\s+)?(?:founded|incorporated|formed|established)', 'history_year'),

        # Pattern 14: "[Company name] was incorporated in [State] in [Year]"
        (r'was\s+incorporated\s+(?:in\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+in\s+(\d{4})', 'was_incorporated_year'),

        # Pattern 15: "a Delaware corporation formed in [Year]"
        (r'[Dd]elaware\s+corporation\s+(?:that\s+was\s+)?(?:formed|incorporated)\s+in\s+(\d{4})', 'delaware_formed'),

        # Pattern 16: Date format like "(f/k/a ... incorporated in [Year])"
        (r'\(f/k/a[^)]+incorporated\s+in\s+(\d{4})\)', 'fka_incorporated'),

        # Pattern 17: "originally incorporated in [Year]"
        (r'originally\s+incorporated\s+in\s+(\d{4})', 'originally_incorporated'),

        # Pattern 18: Specific date format "incorporated ... [Month] [Day], [Year]"
        (r'incorporated[^.]{0,50}([A-Z][a-z]+\s+\d{1,2},\s+\d{4})', 'incorporated_nearby_date'),

        # Pattern 19: "Our company was incorporated on [Date]"
        (r'[Oo]ur\s+[Cc]ompany\s+was\s+incorporated\s+on\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})', 'our_company_date'),

        # Pattern 20: Look for "Formation" or "Organization" section dates
        (r'(?:Formation|Organization|History)[^.]{0,100}in\s+(\d{4})', 'section_year'),
    ]

    for pattern, name in patterns:
        match = re.search(pattern, text_clean)
        if match:
            date_str = match.group(1).strip()
            normalized = normalize_date(date_str)
            if normalized:
                return normalized, name

    return None, None


def normalize_date(date_str: str) -> str | None:
    """Normalize various date formats to YYYY-MM-DD or YYYY-01-01."""
    date_str = date_str.replace(",", "").strip()

    # Try various formats
    formats = [
        ("%B %d %Y", True),   # January 15 2020
        ("%B %Y", False),     # January 2020
        ("%Y", False),        # 2020
    ]

    for fmt, has_day in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if has_day:
                return dt.strftime("%Y-%m-%d")
            elif fmt == "%B %Y":
                return dt.strftime("%Y-%m-01")
            else:
                return f"{date_str}-01-01"
        except ValueError:
            continue

    # If it's just a 4-digit year
    if re.match(r'^\d{4}$', date_str):
        return f"{date_str}-01-01"

    return None


def main():
    print("=" * 60)
    print("Step 1B: Recovering founding dates from unknown companies")
    print("=" * 60)

    # Load companies with unknown founding dates
    with open('step1_biotech_ipos_all.csv', 'r') as f:
        reader = csv.DictReader(f)
        all_companies = list(reader)

    unknowns = [r for r in all_companies if r['founded_since_2000'] == '']
    print(f"\nCompanies with unknown founding date: {len(unknowns)}")

    recovered = []
    still_unknown = []
    pattern_stats = {}

    for i, co in enumerate(unknowns):
        cik = co['cik']
        name = co['name']

        print(f"\n[{i+1}/{len(unknowns)}] Processing {name[:50]}...")

        # Get S-1 filings
        filings = get_company_filings(cik)
        if not filings:
            print(f"  No S-1 filings found")
            still_unknown.append(co)
            continue

        # Get the first S-1 filing
        s1_filing = None
        for f in sorted(filings, key=lambda x: x["date"] or ""):
            if f["form"] == "S-1":
                s1_filing = f
                break
        if not s1_filing:
            s1_filing = filings[0]

        # Download and parse
        if s1_filing.get("accession") and s1_filing.get("primary_doc"):
            text = download_filing_text(cik, s1_filing["accession"], s1_filing["primary_doc"])
            if text:
                founding_date, pattern = extract_founding_date_expanded(text)
                if founding_date:
                    print(f"  FOUND: {founding_date} (pattern: {pattern})")
                    pattern_stats[pattern] = pattern_stats.get(pattern, 0) + 1

                    co['founding_date'] = founding_date
                    try:
                        year = int(founding_date[:4])
                        co['founded_since_2000'] = 'True' if year >= 2000 else 'False'
                    except:
                        co['founded_since_2000'] = ''
                    co['recovery_pattern'] = pattern
                    recovered.append(co)
                else:
                    print(f"  Still not found")
                    still_unknown.append(co)
            else:
                print(f"  Could not download filing")
                still_unknown.append(co)
        else:
            print(f"  Missing filing info")
            still_unknown.append(co)

        # Save intermediate results every 50
        if (i + 1) % 50 == 0:
            save_intermediate(recovered, still_unknown, pattern_stats)

    # Final save
    save_intermediate(recovered, still_unknown, pattern_stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Originally unknown: {len(unknowns)}")
    print(f"Recovered: {len(recovered)}")
    print(f"Still unknown: {len(still_unknown)}")

    recovered_since_2000 = sum(1 for r in recovered if r['founded_since_2000'] == 'True')
    print(f"Recovered and founded since 2000: {recovered_since_2000}")

    print(f"\nPatterns that worked:")
    for pattern, count in sorted(pattern_stats.items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")

    # Update the main files
    update_main_files(all_companies, recovered)


def save_intermediate(recovered, still_unknown, pattern_stats):
    """Save intermediate results."""
    if recovered:
        with open('step1b_recovered.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(recovered[0].keys()))
            writer.writeheader()
            writer.writerows(recovered)
        print(f"\nSaved {len(recovered)} recovered to step1b_recovered.csv")

    if still_unknown:
        with open('step1b_still_unknown.csv', 'w', newline='') as f:
            keys = ['name', 'cik', 'sic_code', 's1_filing_date']
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(still_unknown)


def update_main_files(all_companies, recovered):
    """Update the main output files with recovered dates."""
    # Create a lookup of recovered data
    recovered_lookup = {r['cik']: r for r in recovered}

    # Update all_companies with recovered data
    updated = []
    for co in all_companies:
        if co['cik'] in recovered_lookup:
            rec = recovered_lookup[co['cik']]
            co['founding_date'] = rec['founding_date']
            co['founded_since_2000'] = rec['founded_since_2000']
        updated.append(co)

    # Save updated all file
    with open('step1_biotech_ipos_all_updated.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=updated[0].keys())
        writer.writeheader()
        writer.writerows(updated)

    # Save updated filtered file
    filtered = [r for r in updated if r['founded_since_2000'] == 'True']
    with open('step1_biotech_ipos_since_2000_updated.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=filtered[0].keys())
        writer.writeheader()
        writer.writerows(filtered)

    print(f"\nUpdated files saved:")
    print(f"  step1_biotech_ipos_all_updated.csv ({len(updated)} companies)")
    print(f"  step1_biotech_ipos_since_2000_updated.csv ({len(filtered)} companies)")


if __name__ == "__main__":
    main()
