#!/usr/bin/env python3
"""
Step 3: Extract pipeline information from S-1 filings and company websites.

Extracts:
- Disease area (oncology, immunology, CNS, rare disease, etc.)
- Modality category (small molecule / biologic / other)
- Modality subtype (for biologics: antibody, ADC, RNA, peptide, cell therapy, gene therapy, etc.)
- Drug targets (if mentioned)
- Lead program stage at IPO
- Number of programs

Sources:
1. S-1 filing business description
2. Company website pipeline page (for trading companies)
"""

import csv
import time
import re
import requests
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

HEADERS = {
    "User-Agent": "Academic Research sarah.constantin@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

RATE_LIMIT_DELAY = 0.15

# Disease area keywords
DISEASE_AREAS = {
    'oncology': ['cancer', 'tumor', 'oncology', 'carcinoma', 'melanoma', 'leukemia',
                 'lymphoma', 'myeloma', 'sarcoma', 'glioblastoma', 'metastatic',
                 'solid tumor', 'hematologic malignancies', 'immuno-oncology'],
    'immunology': ['autoimmune', 'inflammatory', 'rheumatoid arthritis', 'lupus',
                   'psoriasis', 'crohn', 'colitis', 'multiple sclerosis', 'immunology',
                   'inflammation', 'atopic dermatitis', 'eczema'],
    'cns': ['neurological', 'neurodegenerative', 'alzheimer', 'parkinson',
            'huntington', 'als', 'amyotrophic lateral sclerosis', 'epilepsy',
            'depression', 'schizophrenia', 'anxiety', 'cns', 'central nervous system',
            'brain', 'neuroscience', 'neuropathy', 'migraine'],
    'rare_disease': ['rare disease', 'orphan', 'genetic disease', 'inherited',
                     'lysosomal storage', 'muscular dystrophy', 'cystic fibrosis',
                     'hemophilia', 'sickle cell', 'thalassemia'],
    'infectious': ['infectious disease', 'viral', 'bacterial', 'antiviral',
                   'antibiotic', 'vaccine', 'hiv', 'hepatitis', 'covid', 'influenza',
                   'antimicrobial', 'fungal'],
    'cardiovascular': ['cardiovascular', 'heart failure', 'cardiac', 'hypertension',
                       'atherosclerosis', 'thrombosis', 'stroke', 'arrhythmia'],
    'metabolic': ['metabolic', 'diabetes', 'obesity', 'nafld', 'nash',
                  'hyperlipidemia', 'cholesterol', 'metabolic syndrome'],
    'ophthalmology': ['ophthalmology', 'eye disease', 'macular degeneration', 'amd',
                      'glaucoma', 'retinal', 'diabetic retinopathy', 'blindness',
                      'ocular', 'ophthalmic'],
    'dermatology': ['dermatology', 'skin', 'acne', 'alopecia', 'wound healing'],
    'respiratory': ['respiratory', 'pulmonary', 'copd', 'asthma', 'lung disease',
                    'pulmonary fibrosis', 'cystic fibrosis'],
    'gastro': ['gastrointestinal', 'gi ', 'liver', 'hepatic', 'ibd',
               'irritable bowel', 'celiac'],
    'renal': ['renal', 'kidney', 'nephrology', 'dialysis', 'ckd'],
    'hematology': ['hematology', 'blood disorder', 'anemia', 'bleeding disorder',
                   'platelet', 'coagulation'],
}

# Modality keywords
MODALITY_SMALL_MOLECULE = [
    'small molecule', 'small-molecule', 'oral', 'orally', 'tablet', 'capsule',
    'chemical compound', 'synthetic compound', 'kinase inhibitor',
    'receptor antagonist', 'enzyme inhibitor'
]

MODALITY_BIOLOGICS = {
    'antibody': ['antibody', 'antibodies', 'monoclonal', 'mab', 'igg',
                 'bispecific', 'nanobody', 'fab', 'immunoglobulin'],
    'adc': ['antibody-drug conjugate', 'antibody drug conjugate', 'adc',
            'immunoconjugate', 'armed antibody'],
    'rna': ['rna', 'mrna', 'sirna', 'antisense', 'oligonucleotide', 'rnai',
            'microrna', 'mirna', 'aptamer', 'nucleic acid'],
    'gene_therapy': ['gene therapy', 'gene-therapy', 'aav', 'adeno-associated virus',
                     'lentiviral', 'viral vector', 'gene transfer', 'gene editing',
                     'crispr', 'gene correction'],
    'cell_therapy': ['cell therapy', 'cell-therapy', 'car-t', 'car t', 'chimeric antigen',
                     'tcr', 't cell receptor', 'nk cell', 'stem cell', 'ipsc',
                     'adoptive cell', 'til ', 'tumor infiltrating'],
    'peptide': ['peptide', 'polypeptide', 'cyclic peptide', 'stapled peptide'],
    'protein': ['recombinant protein', 'fusion protein', 'enzyme replacement',
                'therapeutic protein', 'biologic'],
    'vaccine': ['vaccine', 'immunization', 'prophylactic'],
}

# Common drug targets
TARGETS = [
    # Oncology targets
    'PD-1', 'PD-L1', 'CTLA-4', 'HER2', 'EGFR', 'VEGF', 'KRAS', 'BRAF', 'ALK',
    'CDK4', 'CDK6', 'BTK', 'BCL-2', 'PI3K', 'mTOR', 'MEK', 'JAK', 'FLT3',
    'BCR-ABL', 'MET', 'RET', 'NTRK', 'IDH1', 'IDH2', 'PARP', 'BET', 'MDM2',
    'CD19', 'CD20', 'CD38', 'BCMA', 'CD47', 'TIGIT', 'LAG-3', 'TIM-3', 'STING',
    # Immunology targets
    'TNF', 'IL-17', 'IL-23', 'IL-6', 'IL-4', 'IL-13', 'JAK1', 'JAK2', 'TYK2',
    'S1P', 'integrin', 'CD40', 'BAFF', 'complement',
    # CNS targets
    'tau', 'amyloid', 'alpha-synuclein', 'NMDA', 'GABA', 'dopamine', 'serotonin',
    # Other common targets
    'GLP-1', 'PCSK9', 'SGLT2', 'DPP-4', 'CFTR', 'SMN', 'dystrophin',
]


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
    dates = recent.get("filingDate", [])

    # Find first S-1 (not amendment)
    for i, form in enumerate(forms):
        if form == "S-1":
            return {
                "accession": accessions[i] if i < len(accessions) else None,
                "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
                "date": dates[i] if i < len(dates) else None,
            }

    # Fall back to S-1/A if no original S-1
    for i, form in enumerate(forms):
        if form == "S-1/A":
            return {
                "accession": accessions[i] if i < len(accessions) else None,
                "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
                "date": dates[i] if i < len(dates) else None,
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


def extract_business_section(text: str) -> str:
    """Extract the Business section from S-1 filing."""
    # Remove HTML tags
    text_clean = re.sub(r'<[^>]+>', ' ', text)
    text_clean = re.sub(r'&nbsp;', ' ', text_clean)
    text_clean = re.sub(r'&amp;', '&', text_clean)
    text_clean = re.sub(r'\s+', ' ', text_clean)

    # Try to find Business section
    patterns = [
        r'(?:ITEM\s*1\.?\s*)?BUSINESS[.\s]+(.*?)(?:ITEM\s*[12]A|RISK FACTORS|PROPERTIES)',
        r'(?:Our\s+)?Business\s+Overview(.*?)(?:Our\s+Strategy|Risk\s+Factors)',
        r'Overview\s+(.*?)(?:Our\s+Strategy|Our\s+Pipeline|Risk\s+Factors)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE | re.DOTALL)
        if match:
            section = match.group(1)
            # Limit to reasonable size
            if len(section) > 50000:
                section = section[:50000]
            return section

    # If no match, return first 50K chars (likely contains business description)
    return text_clean[:50000]


def classify_disease_areas(text: str) -> list[str]:
    """Identify disease areas from text."""
    text_lower = text.lower()
    found = []

    for area, keywords in DISEASE_AREAS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                found.append(area)
                break

    return list(set(found))


def classify_modality(text: str) -> dict:
    """Classify modality from text."""
    text_lower = text.lower()

    result = {
        'modality_category': '',  # small_molecule / biologic / other
        'modality_subtype': '',   # specific type for biologics
        'modality_details': [],   # all modalities found
    }

    # Check for small molecule
    is_small_molecule = any(kw.lower() in text_lower for kw in MODALITY_SMALL_MOLECULE)

    # Check for biologics
    biologic_types = []
    for bio_type, keywords in MODALITY_BIOLOGICS.items():
        if any(kw.lower() in text_lower for kw in keywords):
            biologic_types.append(bio_type)

    # Determine primary category
    if biologic_types:
        result['modality_category'] = 'biologic'
        result['modality_subtype'] = biologic_types[0]  # Primary subtype
        result['modality_details'] = biologic_types
        if is_small_molecule:
            result['modality_details'].append('small_molecule')
    elif is_small_molecule:
        result['modality_category'] = 'small_molecule'
        result['modality_subtype'] = 'small_molecule'
    else:
        result['modality_category'] = 'other'

    return result


def extract_targets(text: str) -> list[str]:
    """Extract drug targets from text."""
    found_targets = []
    text_upper = text.upper()

    for target in TARGETS:
        # Look for target with word boundaries
        pattern = r'\b' + re.escape(target.upper()) + r'\b'
        if re.search(pattern, text_upper):
            found_targets.append(target)

    return list(set(found_targets))


def extract_pipeline_stage(text: str) -> dict:
    """Extract pipeline stage information."""
    text_lower = text.lower()

    stages = {
        'phase_3': bool(re.search(r'phase\s*(?:3|iii|three)', text_lower)),
        'phase_2': bool(re.search(r'phase\s*(?:2|ii|two)', text_lower)),
        'phase_1': bool(re.search(r'phase\s*(?:1|i(?![iv])|one)', text_lower)),
        'preclinical': bool(re.search(r'preclinical|pre-clinical|ind-enabling|discovery', text_lower)),
    }

    # Determine most advanced stage
    if stages['phase_3']:
        lead_stage = 'phase_3'
    elif stages['phase_2']:
        lead_stage = 'phase_2'
    elif stages['phase_1']:
        lead_stage = 'phase_1'
    elif stages['preclinical']:
        lead_stage = 'preclinical'
    else:
        lead_stage = 'unknown'

    return {
        'lead_stage': lead_stage,
        'has_phase_3': stages['phase_3'],
        'has_phase_2': stages['phase_2'],
        'has_phase_1': stages['phase_1'],
        'has_preclinical': stages['preclinical'],
    }


def try_fetch_pipeline_page(company_name: str, ticker: str = None) -> str | None:
    """Try to fetch company pipeline page from website."""
    # Skip if no good identifier
    if not ticker and not company_name:
        return None

    # Try common URL patterns
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', company_name.lower().split()[0])

    urls_to_try = []
    if ticker:
        # Sometimes companies use ticker as domain
        urls_to_try.append(f"https://www.{ticker.lower()}.com/pipeline")
        urls_to_try.append(f"https://{ticker.lower()}.com/pipeline")

    urls_to_try.extend([
        f"https://www.{clean_name}.com/pipeline",
        f"https://{clean_name}.com/pipeline",
        f"https://www.{clean_name}therapeutics.com/pipeline",
        f"https://www.{clean_name}pharma.com/pipeline",
        f"https://www.{clean_name}bio.com/pipeline",
    ])

    for url in urls_to_try[:3]:  # Limit attempts
        try:
            time.sleep(0.5)
            response = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            }, timeout=5)

            if response.status_code == 200:
                return response.text
        except:
            continue

    return None


def analyze_company(cik: str, name: str, ticker: str = None) -> dict:
    """Analyze a company's pipeline from S-1 and website."""
    result = {
        'disease_areas': [],
        'primary_disease_area': '',
        'modality_category': '',
        'modality_subtype': '',
        'modality_details': '',
        'targets': [],
        'lead_stage': '',
        'has_phase_3': False,
        'has_phase_2': False,
        'has_phase_1': False,
        'data_source': '',
    }

    # Get S-1 filing
    filing_info = get_s1_filing_info(cik)

    if filing_info and filing_info.get('accession'):
        text = download_s1_text(cik, filing_info['accession'], filing_info['primary_doc'])

        if text:
            business_text = extract_business_section(text)

            # Extract all info
            disease_areas = classify_disease_areas(business_text)
            modality = classify_modality(business_text)
            targets = extract_targets(business_text)
            stages = extract_pipeline_stage(business_text)

            result['disease_areas'] = disease_areas
            result['primary_disease_area'] = disease_areas[0] if disease_areas else ''
            result['modality_category'] = modality['modality_category']
            result['modality_subtype'] = modality['modality_subtype']
            result['modality_details'] = ','.join(modality['modality_details'])
            result['targets'] = targets
            result['lead_stage'] = stages['lead_stage']
            result['has_phase_3'] = stages['has_phase_3']
            result['has_phase_2'] = stages['has_phase_2']
            result['has_phase_1'] = stages['has_phase_1']
            result['data_source'] = 's1_filing'

    # Try website if still missing key info and company is trading
    if (not result['modality_category'] or not result['disease_areas']) and ticker:
        pipeline_html = try_fetch_pipeline_page(name, ticker)
        if pipeline_html:
            # Extract additional info from website
            if not result['disease_areas']:
                result['disease_areas'] = classify_disease_areas(pipeline_html)
                result['primary_disease_area'] = result['disease_areas'][0] if result['disease_areas'] else ''

            if not result['modality_category']:
                modality = classify_modality(pipeline_html)
                result['modality_category'] = modality['modality_category']
                result['modality_subtype'] = modality['modality_subtype']
                result['modality_details'] = ','.join(modality['modality_details'])

            if not result['targets']:
                result['targets'] = extract_targets(pipeline_html)

            result['data_source'] = 's1_filing+website' if result['data_source'] else 'website'

    return result


def main():
    print("=" * 70)
    print("Step 3: Extracting pipeline information")
    print("=" * 70)

    # Load companies
    with open('step2_company_status_complete.csv', 'r') as f:
        reader = csv.DictReader(f)
        companies = list(reader)

    print(f"\nProcessing {len(companies)} companies...")

    results = []
    disease_counts = Counter()
    modality_counts = Counter()

    for i, co in enumerate(companies):
        name = co['name']
        cik = co['cik']
        ticker = co.get('ticker', '')
        status = co.get('status', '')

        print(f"\n[{i+1}/{len(companies)}] {name[:50]}")

        # Analyze company
        pipeline_info = analyze_company(cik, name, ticker if status == 'trading' else None)

        # Merge with existing data
        result = {**co, **pipeline_info}
        result['disease_areas'] = ','.join(pipeline_info['disease_areas'])
        result['targets'] = ','.join(pipeline_info['targets'][:10])  # Limit to 10

        results.append(result)

        # Track counts
        if pipeline_info['primary_disease_area']:
            disease_counts[pipeline_info['primary_disease_area']] += 1
        if pipeline_info['modality_category']:
            modality_counts[pipeline_info['modality_category']] += 1

        # Print summary
        if pipeline_info['disease_areas']:
            print(f"  Disease: {pipeline_info['primary_disease_area']}")
        if pipeline_info['modality_category']:
            print(f"  Modality: {pipeline_info['modality_category']} ({pipeline_info['modality_subtype']})")
        if pipeline_info['lead_stage']:
            print(f"  Lead stage: {pipeline_info['lead_stage']}")

        # Save intermediate results every 50
        if (i + 1) % 50 == 0:
            save_results(results)
            print(f"\n  Saved intermediate results")
            print(f"  Disease areas: {dict(disease_counts.most_common(5))}")
            print(f"  Modalities: {dict(modality_counts)}")

    # Final save
    save_results(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nDisease Area Distribution:")
    for area, count in disease_counts.most_common():
        pct = count / len(companies) * 100
        print(f"  {area:20} {count:4} ({pct:5.1f}%)")

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

    # Biologic subtype distribution
    bio_subtypes = Counter()
    for r in results:
        if r.get('modality_category') == 'biologic' and r.get('modality_subtype'):
            bio_subtypes[r['modality_subtype']] += 1

    print(f"\nBiologic Subtypes:")
    for subtype, count in bio_subtypes.most_common():
        print(f"  {subtype:20} {count:4}")


def save_results(results: list[dict]):
    """Save results to CSV."""
    if not results:
        return

    # Define field order
    key_fields = [
        'name', 'cik', 'ticker', 'founding_date', 's1_filing_date', 'status',
        'primary_disease_area', 'disease_areas', 'modality_category',
        'modality_subtype', 'modality_details', 'targets', 'lead_stage',
        'has_phase_3', 'has_phase_2', 'has_phase_1', 'data_source',
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
