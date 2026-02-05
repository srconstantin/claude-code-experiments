#!/usr/bin/env python3
"""
Step 2B: Find exit information for companies without tickers.

For each company without a ticker, search:
1. SEC 8-K filings for merger/acquisition/bankruptcy keywords
2. SEC 15-12B/G filings (deregistration - signals delisting)
3. Extract acquirer name if acquired

Output: CSV with exit type (acquired/bankrupt/delisted/ipo_withdrawn/unknown) and details
"""

import csv
import time
import re
import requests
from pathlib import Path

HEADERS = {
    "User-Agent": "Academic Research sarah.constantin@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

RATE_LIMIT_DELAY = 0.15


def get_all_filings(cik: str) -> list[dict]:
    """Get all filings for a company."""
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
    descriptions = recent.get("primaryDocDescription", [])

    for i in range(len(forms)):
        filings.append({
            "form": forms[i] if i < len(forms) else "",
            "date": dates[i] if i < len(dates) else "",
            "accession": accessions[i] if i < len(accessions) else "",
            "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
            "description": descriptions[i] if i < len(descriptions) else "",
        })

    return filings


def download_filing_text(cik: str, accession: str, primary_doc: str) -> str | None:
    """Download the text of a filing."""
    accession_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}/{primary_doc}"

    time.sleep(RATE_LIMIT_DELAY)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return None

    return response.text


def analyze_filings_for_exit(filings: list[dict], cik: str) -> dict:
    """
    Analyze filings to determine exit type.

    Returns dict with:
    - exit_type: acquired/bankrupt/delisted/ipo_withdrawn/active/unknown
    - exit_date: date of exit event
    - acquirer: name of acquiring company (if acquired)
    - details: additional details
    """

    # Look for specific filing types that indicate exits

    # 15-12B or 15-12G = deregistration (company no longer public)
    deregistration = [f for f in filings if f['form'] in ['15-12B', '15-12G', '15', '15-15D']]

    # 8-K filings - look at descriptions and content
    eight_ks = [f for f in filings if f['form'] == '8-K']

    # DEFM14A = merger proxy
    merger_proxies = [f for f in filings if 'DEFM14A' in f['form'] or 'PREM14A' in f['form']]

    # SC 14D = tender offer
    tender_offers = [f for f in filings if 'SC 14D' in f['form'] or 'SC TO' in f['form']]

    # Check for S-1 withdrawal (RW = registration withdrawal)
    withdrawals = [f for f in filings if f['form'] in ['RW', 'RW WD']]

    # Keywords to search in 8-K descriptions
    acquisition_keywords = ['merger', 'acquisition', 'acquired', 'business combination',
                           'tender offer', 'change of control', 'going private']
    bankruptcy_keywords = ['bankruptcy', 'chapter 11', 'chapter 7', 'liquidation',
                          'wind down', 'cease operations', 'dissolution']

    result = {
        'exit_type': 'unknown',
        'exit_date': '',
        'acquirer': '',
        'details': '',
        'last_filing_date': filings[0]['date'] if filings else '',
        'last_filing_type': filings[0]['form'] if filings else '',
    }

    # Check for merger proxies or tender offers (strong signal of acquisition)
    if merger_proxies:
        result['exit_type'] = 'acquired'
        result['exit_date'] = merger_proxies[0]['date']
        result['details'] = f"Merger proxy filed: {merger_proxies[0]['form']}"

        # Try to get acquirer name from the filing
        filing = merger_proxies[0]
        if filing.get('accession') and filing.get('primary_doc'):
            text = download_filing_text(cik, filing['accession'], filing['primary_doc'])
            if text:
                acquirer = extract_acquirer_name(text)
                if acquirer:
                    result['acquirer'] = acquirer
        return result

    if tender_offers:
        result['exit_type'] = 'acquired'
        result['exit_date'] = tender_offers[0]['date']
        result['details'] = f"Tender offer filed: {tender_offers[0]['form']}"
        return result

    # Check 8-K descriptions for keywords
    for filing in eight_ks[:20]:  # Check recent 8-Ks
        desc = (filing.get('description') or '').lower()

        for kw in acquisition_keywords:
            if kw in desc:
                result['exit_type'] = 'acquired'
                result['exit_date'] = filing['date']
                result['details'] = filing['description'][:100]

                # Try to get acquirer from filing
                if filing.get('accession') and filing.get('primary_doc'):
                    text = download_filing_text(cik, filing['accession'], filing['primary_doc'])
                    if text:
                        acquirer = extract_acquirer_name(text)
                        if acquirer:
                            result['acquirer'] = acquirer
                return result

        for kw in bankruptcy_keywords:
            if kw in desc:
                result['exit_type'] = 'bankrupt'
                result['exit_date'] = filing['date']
                result['details'] = filing['description'][:100]
                return result

    # Check for deregistration (could be after acquisition or bankruptcy)
    if deregistration:
        result['exit_type'] = 'delisted'
        result['exit_date'] = deregistration[0]['date']
        result['details'] = f"Deregistered: {deregistration[0]['form']}"
        return result

    # Check for S-1 withdrawal (never completed IPO)
    if withdrawals:
        result['exit_type'] = 'ipo_withdrawn'
        result['exit_date'] = withdrawals[0]['date']
        result['details'] = 'IPO registration withdrawn'
        return result

    # If only S-1 filings and nothing else, IPO may never have completed
    forms = set(f['form'] for f in filings)
    if forms.issubset({'S-1', 'S-1/A', 'DRS', 'DRS/A'}):
        result['exit_type'] = 'ipo_incomplete'
        result['details'] = 'Only S-1 filings found, IPO may not have completed'
        return result

    # Check if there are 10-K/10-Q filings (sign of operating company)
    periodic_reports = [f for f in filings if f['form'] in ['10-K', '10-Q', '10-K/A', '10-Q/A']]
    if periodic_reports:
        last_report = periodic_reports[0]
        result['last_periodic_report'] = last_report['date']

        # If last periodic report is old, company may be defunct
        # (We'll flag this but can't determine exact exit)
        result['details'] = f"Last periodic report: {last_report['date']}"

    return result


def extract_acquirer_name(text: str) -> str | None:
    """
    Try to extract acquirer name from merger filing text.
    """
    # Common patterns
    patterns = [
        r'(?:merger|acquisition|acquired|combination)\s+(?:with|by)\s+([A-Z][A-Za-z\s,\.]+(?:Inc\.|Corp\.|Ltd\.|LLC|Company|Pharmaceuticals|Therapeutics))',
        r'([A-Z][A-Za-z\s]+(?:Inc\.|Corp\.|Ltd\.))\s+(?:will|to|shall)\s+acquire',
        r'(?:Parent|Acquiror|Purchaser)["\s:]+([A-Z][A-Za-z\s,\.]+(?:Inc\.|Corp\.|Ltd\.))',
        r'wholly[- ]owned subsidiary of\s+([A-Z][A-Za-z\s,\.]+(?:Inc\.|Corp\.|Ltd\.))',
    ]

    # Clean HTML
    text_clean = re.sub(r'<[^>]+>', ' ', text)

    for pattern in patterns:
        match = re.search(pattern, text_clean)
        if match:
            acquirer = match.group(1).strip()
            # Clean up
            acquirer = re.sub(r'\s+', ' ', acquirer)
            if len(acquirer) > 5 and len(acquirer) < 100:
                return acquirer

    return None


def main():
    print("=" * 70)
    print("Step 2B: Finding exit information for companies without tickers")
    print("=" * 70)

    # Load companies without tickers
    with open('step2_company_status.csv', 'r') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    no_ticker = [r for r in all_rows if r['status'] == 'no_ticker']
    print(f"\nCompanies to analyze: {len(no_ticker)}")

    results = []
    exit_counts = {}

    for i, co in enumerate(no_ticker):
        name = co['name']
        cik = co['cik']

        print(f"\n[{i+1}/{len(no_ticker)}] {name[:50]}")

        # Get all filings
        filings = get_all_filings(cik)

        if not filings:
            print(f"  No filings found")
            result = {
                'name': name,
                'cik': cik,
                'founding_date': co.get('founding_date', ''),
                's1_filing_date': co.get('s1_filing_date', ''),
                'exit_type': 'no_filings',
                'exit_date': '',
                'acquirer': '',
                'details': 'No SEC filings found',
            }
        else:
            print(f"  Found {len(filings)} filings")

            # Analyze filings
            exit_info = analyze_filings_for_exit(filings, cik)

            result = {
                'name': name,
                'cik': cik,
                'founding_date': co.get('founding_date', ''),
                's1_filing_date': co.get('s1_filing_date', ''),
                **exit_info,
            }

            print(f"  Exit type: {exit_info['exit_type']}")
            if exit_info.get('acquirer'):
                print(f"  Acquirer: {exit_info['acquirer'][:50]}")

        results.append(result)
        exit_counts[result['exit_type']] = exit_counts.get(result['exit_type'], 0) + 1

        # Save intermediate results every 50
        if (i + 1) % 50 == 0:
            save_results(results)
            print(f"\n  Saved intermediate results")
            print(f"  Exit counts so far: {exit_counts}")

    # Final save
    save_results(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal companies analyzed: {len(results)}")
    print(f"\nExit type breakdown:")
    for exit_type, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
        print(f"  {exit_type}: {count}")

    # Show some acquired companies with acquirers
    acquired = [r for r in results if r['exit_type'] == 'acquired' and r.get('acquirer')]
    print(f"\nSample of acquired companies (with acquirer identified):")
    for r in acquired[:20]:
        print(f"  {r['name'][:35]:<37} -> {r['acquirer'][:30]}")


def save_results(results: list[dict]):
    """Save results to CSV."""
    if not results:
        return

    fieldnames = ['name', 'cik', 'founding_date', 's1_filing_date', 'exit_type',
                  'exit_date', 'acquirer', 'details', 'last_filing_date',
                  'last_filing_type', 'last_periodic_report']

    with open('step2b_exit_info.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} companies to step2b_exit_info.csv")


if __name__ == "__main__":
    main()
