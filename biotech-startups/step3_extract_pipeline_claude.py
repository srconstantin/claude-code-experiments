#!/usr/bin/env python3
"""
Step 3: Extract pipeline information using Claude API.

For each company, downloads S-1 filing and uses Claude to extract:
- Disease area(s)
- Modality category (small molecule / biologic / other)
- Modality subtype (for biologics: antibody, ADC, RNA, cell therapy, gene therapy, etc.)
- Primary drug target(s)
- Lead program stage at IPO

Uses Claude Haiku for fast, cost-effective extraction.
"""

import csv
import time
import re
import json
import os
import requests
from pathlib import Path
from collections import Counter

import anthropic

HEADERS = {
    "User-Agent": "Academic Research sarah.constantin@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

RATE_LIMIT_DELAY = 0.15

# Initialize Anthropic client
client = anthropic.Anthropic()

EXTRACTION_PROMPT = """Analyze this biotech company's S-1 filing excerpt and extract the following information about their drug pipeline. Return your answer as a JSON object.

<filing_text>
{text}
</filing_text>

Extract and return a JSON object with these fields:

1. "disease_areas": List of disease areas they are targeting. Use standardized terms from this list when applicable: oncology, immunology, cns (central nervous system/neurology), rare_disease, infectious_disease, cardiovascular, metabolic, ophthalmology, dermatology, respiratory, gastroenterology, renal, hematology. You can add other areas if they don't fit these categories.

2. "primary_disease_area": The single most important disease area for this company (their main focus).

3. "modality_category": One of: "small_molecule", "biologic", "both", or "other"

4. "modality_subtype": Be specific. For biologics, specify: "antibody", "adc" (antibody-drug conjugate), "bispecific", "rna" (siRNA, mRNA, antisense), "gene_therapy", "cell_therapy", "car_t", "peptide", "protein", "vaccine", or other. For small molecules, just say "small_molecule". If multiple, list the primary one.

5. "modality_details": List all modality types if the company has multiple (e.g., ["antibody", "adc", "small_molecule"])

6. "targets": List of specific drug targets mentioned (e.g., "PD-1", "HER2", "KRAS", "IL-17", "CD19"). Include up to 5 most important targets. Use standard nomenclature.

7. "lead_stage": The most advanced clinical stage of their pipeline at the time of filing. One of: "approved", "phase_3", "phase_2", "phase_1", "preclinical", "discovery", or "unknown"

8. "pipeline_summary": A one-sentence summary of what the company does (e.g., "Develops CAR-T cell therapies for blood cancers targeting CD19 and BCMA")

Return ONLY the JSON object, no other text. If information is not available, use empty strings or empty lists."""


def get_s1_filing_info(cik: str) -> dict | None:
    """Get the first S-1 filing accession number and primary doc."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    time.sleep(RATE_LIMIT_DELAY)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return None

    data = response.json()
    recent = data.get("filings", {}).get("recent", {})

    if not recent:
        return None

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    # Find first S-1 (not amendment)
    for i, form in enumerate(forms):
        if form == "S-1":
            return {
                "accession": accessions[i] if i < len(accessions) else None,
                "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
            }

    # Fall back to S-1/A
    for i, form in enumerate(forms):
        if form == "S-1/A":
            return {
                "accession": accessions[i] if i < len(accessions) else None,
                "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
            }

    return None


def download_s1_text(cik: str, accession: str, primary_doc: str) -> str | None:
    """Download S-1 filing text."""
    accession_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}/{primary_doc}"

    time.sleep(RATE_LIMIT_DELAY)
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return None

    return response.text


def extract_business_section(text: str, max_chars: int = 30000) -> str:
    """Extract and clean the Business section from S-1 filing."""
    # Remove HTML tags
    text_clean = re.sub(r'<[^>]+>', ' ', text)
    text_clean = re.sub(r'&nbsp;', ' ', text_clean)
    text_clean = re.sub(r'&amp;', '&', text_clean)
    text_clean = re.sub(r'&#\d+;', ' ', text_clean)
    text_clean = re.sub(r'\s+', ' ', text_clean)

    # Better patterns - avoid matching Table of Contents
    patterns = [
        # Look for "Overview" sections with substantial content (not TOC)
        r'(?:Company\s+)?Overview\s+(.{500,}?)(?:Our\s+Strategy|Our\s+Pipeline|Risk\s+Factors|ITEM\s+1A|ITEM\s+2)',

        # Look for "We are a" intro patterns common in biotech S-1s
        r'(We\s+are\s+(?:a|an)\s+[^.]{10,200}(?:biopharmaceutical|biotechnology|pharmaceutical|therapeutics|clinical)[^.]*\.(?:.*?))(?:ITEM\s+1A|Risk\s+Factors|PROPERTIES)',

        # ITEM 1 BUSINESS with substantial content (5000+ chars, not just TOC entries)
        r'ITEM\s+1[.\s]+BUSINESS\s+(?![\d\s,]+(?:ITEM|MANAGEMENT|PROPERTIES))(.{5000,}?)(?:ITEM\s+1A|ITEM\s+2|RISK\s+FACTORS|PROPERTIES)',

        # Look for "Our Company" or "Business Overview" with content
        r'(?:Our\s+Company|Business\s+Overview|Company\s+Overview)\s+(.{1000,}?)(?:Our\s+Strategy|Risk\s+Factors|ITEM)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE | re.DOTALL)
        if match:
            section = match.group(1)
            # Verify it's not just TOC entries (which have lots of numbers)
            if len(re.findall(r'\d{2,}', section[:500])) < 10:  # Not too many page numbers
                return section[:max_chars]

    # Fallback: find "We are a" and grab surrounding context
    we_are_match = re.search(r'(We\s+are\s+(?:a|an)\s+[^.]{10,300}\.)', text_clean, re.IGNORECASE)
    if we_are_match:
        start = max(0, we_are_match.start())
        end = min(len(text_clean), start + max_chars)
        return text_clean[start:end]

    # Last resort: return first chunk
    return text_clean[:max_chars]


def extract_with_claude(text: str, company_name: str) -> dict:
    """Use Claude to extract pipeline information from filing text."""

    prompt = EXTRACTION_PROMPT.format(text=text)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)

        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        return {}
    except anthropic.APIError as e:
        print(f"    API error: {e}")
        return {}
    except Exception as e:
        print(f"    Error: {e}")
        return {}


def main():
    print("=" * 70)
    print("Step 3: Extracting pipeline information using Claude API")
    print("=" * 70)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    # Load companies
    with open('step2_company_status_complete.csv', 'r') as f:
        reader = csv.DictReader(f)
        companies = list(reader)

    print(f"\nProcessing {len(companies)} companies...")

    results = []
    disease_counts = Counter()
    modality_counts = Counter()
    api_calls = 0
    errors = 0

    for i, co in enumerate(companies):
        name = co['name']
        cik = co['cik']

        print(f"\n[{i+1}/{len(companies)}] {name[:50]}")

        # Get S-1 filing
        filing_info = get_s1_filing_info(cik)

        pipeline_info = {
            'disease_areas': '',
            'primary_disease_area': '',
            'modality_category': '',
            'modality_subtype': '',
            'modality_details': '',
            'targets': '',
            'lead_stage': '',
            'pipeline_summary': '',
            'extraction_source': '',
        }

        if filing_info and filing_info.get('accession'):
            text = download_s1_text(cik, filing_info['accession'], filing_info['primary_doc'])

            if text:
                business_text = extract_business_section(text)

                if len(business_text) > 500:  # Only call API if we have substantial text
                    # Check if it looks like real content (not TOC)
                    preview = business_text[:200].lower()
                    if 'we are' in preview or 'company' in preview or 'develop' in preview or 'therapeutic' in preview:
                        print(f"  Extracted {len(business_text):,} chars of business text", flush=True)
                        print(f"  Calling Claude API...", flush=True)
                        extracted = extract_with_claude(business_text, name)
                        api_calls += 1
                    else:
                        print(f"  Text doesn't look like business description, skipping API", flush=True)
                        print(f"  Preview: {business_text[:150]}...", flush=True)
                        extracted = {}

                    if extracted:
                        # Process extracted data
                        disease_list = extracted.get('disease_areas', [])
                        if isinstance(disease_list, list):
                            pipeline_info['disease_areas'] = ','.join(disease_list)
                        else:
                            pipeline_info['disease_areas'] = str(disease_list)

                        pipeline_info['primary_disease_area'] = extracted.get('primary_disease_area', '')
                        pipeline_info['modality_category'] = extracted.get('modality_category', '')
                        pipeline_info['modality_subtype'] = extracted.get('modality_subtype', '')

                        modality_details = extracted.get('modality_details', [])
                        if isinstance(modality_details, list):
                            pipeline_info['modality_details'] = ','.join(modality_details)
                        else:
                            pipeline_info['modality_details'] = str(modality_details)

                        targets = extracted.get('targets', [])
                        if isinstance(targets, list):
                            pipeline_info['targets'] = ','.join(targets[:10])
                        else:
                            pipeline_info['targets'] = str(targets)

                        pipeline_info['lead_stage'] = extracted.get('lead_stage', '')
                        pipeline_info['pipeline_summary'] = extracted.get('pipeline_summary', '')
                        pipeline_info['extraction_source'] = 's1_filing'

                        # Print summary
                        print(f"  Disease: {pipeline_info['primary_disease_area']}")
                        print(f"  Modality: {pipeline_info['modality_category']} ({pipeline_info['modality_subtype']})")
                        print(f"  Stage: {pipeline_info['lead_stage']}")

                        # Track counts
                        if pipeline_info['primary_disease_area']:
                            disease_counts[pipeline_info['primary_disease_area']] += 1
                        if pipeline_info['modality_category']:
                            modality_counts[pipeline_info['modality_category']] += 1
                    else:
                        errors += 1
                        print(f"  No data extracted")
                else:
                    print(f"  Insufficient text in filing")
        else:
            print(f"  No S-1 filing found")

        # Merge with existing data
        result = {**co, **pipeline_info}
        results.append(result)

        # Save intermediate results every 50
        if (i + 1) % 50 == 0:
            save_results(results)
            print(f"\n  Saved intermediate results ({api_calls} API calls, {errors} errors)")
            print(f"  Disease areas: {dict(disease_counts.most_common(5))}")
            print(f"  Modalities: {dict(modality_counts)}")

        # Rate limiting for API
        time.sleep(0.5)

    # Final save
    save_results(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"API calls made: {api_calls}")
    print(f"Extraction errors: {errors}")

    print(f"\nDisease Area Distribution:")
    for area, count in disease_counts.most_common(15):
        pct = count / len(companies) * 100
        print(f"  {area:25} {count:4} ({pct:5.1f}%)")

    print(f"\nModality Distribution:")
    for mod, count in modality_counts.most_common():
        pct = count / len(companies) * 100
        print(f"  {mod:20} {count:4} ({pct:5.1f}%)")

    # Stage distribution
    stage_counts = Counter(r['lead_stage'] for r in results if r.get('lead_stage'))
    print(f"\nLead Stage at IPO:")
    for stage, count in stage_counts.most_common():
        pct = count / len(companies) * 100
        print(f"  {stage:20} {count:4} ({pct:5.1f}%)")


def save_results(results: list[dict]):
    """Save results to CSV."""
    if not results:
        return

    # Define field order
    key_fields = [
        'name', 'cik', 'ticker', 'founding_date', 's1_filing_date', 'status',
        'primary_disease_area', 'disease_areas', 'modality_category',
        'modality_subtype', 'modality_details', 'targets', 'lead_stage',
        'pipeline_summary', 'extraction_source',
    ]

    # Get all fields
    all_fields = set()
    for r in results:
        all_fields.update(r.keys())

    # Order fields: key fields first, then others
    other_fields = sorted(all_fields - set(key_fields))
    fieldnames = key_fields + other_fields

    with open('step3_pipeline_info.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} companies to step3_pipeline_info.csv")


if __name__ == "__main__":
    main()
