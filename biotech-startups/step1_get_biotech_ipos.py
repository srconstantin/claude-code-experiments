#!/usr/bin/env python3
"""
Step 1: Build list of US biotech IPOs from SEC EDGAR

Queries SEC EDGAR for S-1 filings (IPO registration statements) in:
- SIC 2834: Pharmaceutical Preparations
- SIC 2836: Biological Products (except diagnostic)

Then parses each filing to extract founding date, filtering to companies
founded since 2000.

Output: CSV with company name, CIK, ticker, SIC code, filing date, founding date
"""

import requests
import json
import time
import re
import csv
from datetime import datetime
from pathlib import Path

# SEC requires a user-agent with contact info
HEADERS = {
    "User-Agent": "Academic Research sarah.constantin@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# SIC codes for therapeutics biotechs
SIC_CODES = ["2834", "2836"]

# Rate limit: SEC asks for max 10 requests/second
RATE_LIMIT_DELAY = 0.15  # seconds between requests


def get_companies_by_sic(sic_code: str) -> list[dict]:
    """
    Get list of companies with a given SIC code from SEC EDGAR.
    Uses the company search endpoint.
    """
    companies = []

    # SEC EDGAR company search API
    url = f"https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "action": "getcompany",
        "SIC": sic_code,
        "type": "S-1",  # IPO registration statements
        "dateb": "",
        "owner": "include",
        "count": 100,
        "start": 0,
        "output": "atom",
    }

    print(f"Fetching companies with SIC code {sic_code}...")

    while True:
        time.sleep(RATE_LIMIT_DELAY)
        response = requests.get(url, params=params, headers=HEADERS)

        if response.status_code != 200:
            print(f"  Error: HTTP {response.status_code}")
            break

        # Parse the Atom feed response
        content = response.text

        # Extract entries using regex (simple parsing for Atom feed)
        entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)

        if not entries:
            break

        for entry in entries:
            # Extract company info
            title_match = re.search(r'<title>(.*?)</title>', entry)
            cik_match = re.search(r'CIK=(\d+)', entry)
            updated_match = re.search(r'<updated>(.*?)</updated>', entry)

            if title_match and cik_match:
                # Title format is usually "COMPANY NAME (SIC-CODE) FORM-TYPE"
                title = title_match.group(1)
                company_name = re.sub(r'\s*\(.*?\)\s*S-1.*$', '', title).strip()

                companies.append({
                    "name": company_name,
                    "cik": cik_match.group(1).zfill(10),
                    "sic_code": sic_code,
                    "filing_date": updated_match.group(1)[:10] if updated_match else None,
                })

        print(f"  Found {len(entries)} entries, total so far: {len(companies)}")

        # Check if there are more results
        if len(entries) < 100:
            break

        params["start"] += 100

    return companies


def get_company_filings(cik: str) -> list[dict]:
    """
    Get all filings for a company using the SEC submissions API.
    """
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
        if form in ["S-1", "S-1/A"]:  # Include amendments
            filings.append({
                "form": form,
                "date": dates[i] if i < len(dates) else None,
                "accession": accessions[i] if i < len(accessions) else None,
                "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
            })

    return filings


def get_company_info(cik: str) -> dict:
    """
    Get detailed company info from SEC submissions API.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    time.sleep(RATE_LIMIT_DELAY)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return {}

    data = response.json()

    return {
        "name": data.get("name", ""),
        "cik": cik,
        "sic": data.get("sic", ""),
        "sic_description": data.get("sicDescription", ""),
        "tickers": data.get("tickers", []),
        "exchanges": data.get("exchanges", []),
        "state": data.get("addresses", {}).get("business", {}).get("stateOrCountry", ""),
        "city": data.get("addresses", {}).get("business", {}).get("city", ""),
    }


def extract_founding_date_from_filing(cik: str, accession: str, primary_doc: str) -> str | None:
    """
    Download an S-1 filing and extract the founding date.

    Founding date is typically mentioned in phrases like:
    - "We were incorporated in Delaware in [month] [year]"
    - "The Company was founded in [year]"
    - "incorporated on [date]"
    """
    # Construct URL for the filing
    accession_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}/{primary_doc}"

    time.sleep(RATE_LIMIT_DELAY)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return None

    text = response.text

    # Common patterns for founding/incorporation date
    patterns = [
        # "incorporated in [State] in [Month] [Year]"
        r'(?:incorporated|organized|formed)\s+(?:in\s+)?(?:the\s+)?(?:State\s+of\s+)?(?:\w+)\s+(?:in|on)\s+(\w+\s+\d{4})',
        # "incorporated in [Month] [Year]"
        r'(?:incorporated|organized|formed)\s+in\s+(\w+\s+\d{4})',
        # "incorporated on [Month] [Day], [Year]"
        r'(?:incorporated|organized|formed)\s+on\s+(\w+\s+\d{1,2},?\s+\d{4})',
        # "founded in [Year]"
        r'(?:founded|established)\s+in\s+(\d{4})',
        # "since our founding in [Year]"
        r'(?:since\s+(?:our\s+)?(?:founding|incorporation)\s+in)\s+(\d{4})',
        # "a [State] corporation incorporated in [Year]"
        r'corporation\s+incorporated\s+in\s+(\d{4})',
        # "incorporated under the laws of [State] in [Year]"
        r'(?:incorporated|organized)\s+under\s+the\s+laws\s+of\s+\w+\s+in\s+(\d{4})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Try to parse and normalize the date
            try:
                # Try various date formats
                for fmt in ["%B %Y", "%B %d, %Y", "%B %d %Y", "%Y"]:
                    try:
                        dt = datetime.strptime(date_str.replace(",", ""), fmt)
                        return dt.strftime("%Y-%m-%d") if fmt != "%Y" else f"{date_str}-01-01"
                    except ValueError:
                        continue
                # If can't parse, return as-is
                return date_str
            except Exception:
                return date_str

    return None


def main():
    print("=" * 60)
    print("Step 1: Building list of biotech IPOs from SEC EDGAR")
    print("=" * 60)

    # Collect all companies from both SIC codes
    all_companies = []
    seen_ciks = set()

    for sic in SIC_CODES:
        companies = get_companies_by_sic(sic)
        for co in companies:
            if co["cik"] not in seen_ciks:
                seen_ciks.add(co["cik"])
                all_companies.append(co)

    print(f"\nFound {len(all_companies)} unique companies with S-1 filings")
    print("\nFetching detailed company info and founding dates...")

    # Get detailed info for each company
    results = []

    for i, co in enumerate(all_companies):
        cik = co["cik"]
        print(f"\n[{i+1}/{len(all_companies)}] Processing {co['name']}...")

        # Get company info
        info = get_company_info(cik)
        if not info:
            print(f"  Could not fetch company info, skipping")
            continue

        # Get S-1 filings
        filings = get_company_filings(cik)
        if not filings:
            print(f"  No S-1 filings found, skipping")
            continue

        # Get the first S-1 filing (original, not amendment)
        s1_filing = None
        for f in sorted(filings, key=lambda x: x["date"] or ""):
            if f["form"] == "S-1":
                s1_filing = f
                break

        if not s1_filing:
            s1_filing = filings[0]  # Use first available

        print(f"  S-1 filed: {s1_filing['date']}")

        # Extract founding date from filing
        founding_date = None
        if s1_filing.get("accession") and s1_filing.get("primary_doc"):
            founding_date = extract_founding_date_from_filing(
                cik, s1_filing["accession"], s1_filing["primary_doc"]
            )
            if founding_date:
                print(f"  Founding date found: {founding_date}")
            else:
                print(f"  Founding date not found in filing")

        # Check if founded since 2000
        founded_since_2000 = None
        if founding_date:
            try:
                year = int(founding_date[:4])
                founded_since_2000 = year >= 2000
            except (ValueError, TypeError):
                pass

        results.append({
            "name": info.get("name", co["name"]),
            "cik": cik,
            "sic_code": info.get("sic", co["sic_code"]),
            "sic_description": info.get("sic_description", ""),
            "ticker": info.get("tickers", [""])[0] if info.get("tickers") else "",
            "exchange": info.get("exchanges", [""])[0] if info.get("exchanges") else "",
            "state": info.get("state", ""),
            "city": info.get("city", ""),
            "s1_filing_date": s1_filing["date"],
            "founding_date": founding_date,
            "founded_since_2000": founded_since_2000,
        })

        # Save intermediate results every 50 companies
        if (i + 1) % 50 == 0:
            save_results(results, "step1_biotech_ipos_intermediate.csv")
            print(f"\n  Saved intermediate results ({len(results)} companies)")

    # Save final results
    save_results(results, "step1_biotech_ipos_all.csv")

    # Filter to companies founded since 2000
    filtered = [r for r in results if r["founded_since_2000"] == True]
    save_results(filtered, "step1_biotech_ipos_since_2000.csv")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total companies with S-1 filings: {len(results)}")
    print(f"Companies with founding date found: {sum(1 for r in results if r['founding_date'])}")
    print(f"Companies founded since 2000: {len(filtered)}")
    print(f"Companies with unknown founding date: {sum(1 for r in results if r['founded_since_2000'] is None)}")

    print(f"\nOutput files:")
    print(f"  - step1_biotech_ipos_all.csv (all companies)")
    print(f"  - step1_biotech_ipos_since_2000.csv (filtered to founded >= 2000)")


def save_results(results: list[dict], filename: str):
    """Save results to CSV."""
    if not results:
        return

    filepath = Path(__file__).parent / filename

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} companies to {filename}")


if __name__ == "__main__":
    main()
