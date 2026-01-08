"""
Add druggability annotations to pan-cancer targets.

Uses multiple sources:
1. ChEMBL API - drugs with known targets
2. Gene family patterns - kinases, GPCRs, ion channels, etc.
3. Protein class annotations

Output: Updated targets CSV with druggability info
"""

import pandas as pd
import requests
import json
import time
from pathlib import Path

DATA_DIR = Path("data")

# Known druggable gene families (patterns in gene names)
DRUGGABLE_FAMILIES = {
    'kinase': ['kinase', 'CDK', 'MAPK', 'JAK', 'SRC', 'ABL', 'EGFR', 'VEGFR', 'PDGFR', 'FGFR',
               'MET', 'ALK', 'RET', 'KIT', 'FLT', 'BTK', 'PIK3', 'AURK', 'PLK', 'CHK', 'WEE',
               'BRAF', 'RAF', 'MEK', 'ERK', 'AKT', 'mTOR', 'GSK3', 'CK1', 'CK2', 'DYRK',
               'PRPF4B', 'SRPK', 'CLK', 'PKC', 'PKA', 'ROCK', 'PKM', 'PKR'],
    'protease': ['protease', 'peptidase', 'cathepsin', 'caspase', 'MMP', 'ADAM', 'USP',
                 'SENP', 'CASP', 'CTSL', 'CTSB', 'CTSD'],
    'phosphatase': ['phosphatase', 'PTP', 'PTPN', 'PTPRA', 'DUSP', 'CDC25', 'PPP', 'PPM'],
    'GPCR': ['GPR', 'ADORA', 'DRD', 'HTR', 'CHRM', 'OPRM', 'ADRB', 'ADRA', 'S1PR', 'CXCR', 'CCR'],
    'ion_channel': ['SCN', 'CACNA', 'KCNA', 'KCNB', 'KCNC', 'KCND', 'KCNH', 'KCNJ', 'KCNQ',
                   'HCN', 'TRPV', 'TRPM', 'TRPC', 'GABRA', 'GRIN', 'GRIA', 'GRIK'],
    'nuclear_receptor': ['NR1', 'NR2', 'NR3', 'NR4', 'NR5', 'ESR', 'AR', 'GR', 'MR', 'PR',
                        'PPARG', 'PPARA', 'RXRA', 'RARA', 'VDR', 'THR'],
    'transporter': ['SLC', 'ABC', 'TFRC', 'ATP1', 'ATP2', 'ATP6', 'ATP7'],
    'epigenetic': ['HDAC', 'HAT', 'KMT', 'KDM', 'DNMT', 'TET', 'BRD', 'SETD', 'EZH', 'DOT1L',
                  'PRMT', 'SIRT', 'EHMT', 'SUV', 'NSD', 'SMYD', 'LSD', 'JMJD', 'PHF', 'ARID'],
    'enzyme_other': ['DHFR', 'TYMS', 'RNR', 'IMPDH', 'DHODH', 'PARP', 'IDH', 'LDHA', 'LDHB',
                    'HK', 'PFKFB', 'PGAM', 'ENO', 'PGK', 'GAPDH', 'TPI', 'ALDOA', 'PKM',
                    'NMT', 'HMGCR', 'HMGCS', 'SCD', 'FASN', 'ACLY', 'ACO', 'GGPS', 'FDPS']
}

# Specific known druggable targets with approved/clinical drugs
KNOWN_DRUGGED_TARGETS = {
    'DHFR': {'drugs': ['methotrexate', 'pemetrexed', 'pralatrexate'], 'status': 'approved'},
    'TYMS': {'drugs': ['5-fluorouracil', 'capecitabine'], 'status': 'approved'},
    'TFRC': {'drugs': ['anti-TFRC antibodies (clinical)'], 'status': 'clinical'},
    'TOP2A': {'drugs': ['doxorubicin', 'etoposide'], 'status': 'approved'},
    'CDK1': {'drugs': ['dinaciclib', 'flavopiridol'], 'status': 'clinical'},
    'KIF11': {'drugs': ['ispinesib', 'filanesib'], 'status': 'clinical'},
    'PLK1': {'drugs': ['volasertib', 'onvansertib'], 'status': 'clinical'},
    'BUB1B': {'drugs': ['paclitaxel (indirect)'], 'status': 'approved'},
    'MCL1': {'drugs': ['S63845', 'AMG-176'], 'status': 'clinical'},
    'PARP1': {'drugs': ['olaparib', 'niraparib', 'rucaparib'], 'status': 'approved'},
    'HDAC1': {'drugs': ['vorinostat', 'romidepsin'], 'status': 'approved'},
    'PRMT5': {'drugs': ['GSK3326595', 'JNJ-64619178'], 'status': 'clinical'},
    'EZH2': {'drugs': ['tazemetostat'], 'status': 'approved'},
    'BRD4': {'drugs': ['JQ1', 'OTX015', 'ABBV-075'], 'status': 'clinical'},
    'NAMPT': {'drugs': ['FK866', 'GMX1778'], 'status': 'clinical'},
    'AURKA': {'drugs': ['alisertib'], 'status': 'clinical'},
    'AURKB': {'drugs': ['barasertib'], 'status': 'clinical'},
    'WEE1': {'drugs': ['adavosertib'], 'status': 'clinical'},
    'CHK1': {'drugs': ['prexasertib', 'rabusertib'], 'status': 'clinical'},
    'HMGCR': {'drugs': ['statins'], 'status': 'approved'},
    'GGPS1': {'drugs': ['bisphosphonates (indirect)'], 'status': 'approved'},
    'ACLY': {'drugs': ['bempedoic acid'], 'status': 'approved'},
    'SCD': {'drugs': ['MK-8245'], 'status': 'clinical'},
    'IDH1': {'drugs': ['ivosidenib'], 'status': 'approved'},
    'IDH2': {'drugs': ['enasidenib'], 'status': 'approved'},
    'DHODH': {'drugs': ['leflunomide', 'brequinar'], 'status': 'approved/clinical'},
    'IMPDH1': {'drugs': ['mycophenolate'], 'status': 'approved'},
    'IMPDH2': {'drugs': ['mycophenolate'], 'status': 'approved'},
    'RRM1': {'drugs': ['gemcitabine'], 'status': 'approved'},
    'RRM2': {'drugs': ['hydroxyurea', 'gemcitabine'], 'status': 'approved'},
    'PSMB5': {'drugs': ['bortezomib', 'carfilzomib'], 'status': 'approved'},
    'USP7': {'drugs': ['P5091', 'GNE-6640'], 'status': 'preclinical'},
    'UBA1': {'drugs': ['TAK-243'], 'status': 'clinical'},
    'SAE1': {'drugs': ['ML-792', 'TAK-981'], 'status': 'clinical'},
}


def query_chembl_for_gene(gene_symbol, max_retries=3):
    """Query ChEMBL for drug targets matching a gene symbol."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={gene_symbol}&limit=10"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                targets = data.get('targets', [])

                # Check if any target matches our gene
                for target in targets:
                    components = target.get('target_components', [])
                    for comp in components:
                        synonyms = comp.get('target_component_synonyms', [])
                        for syn in synonyms:
                            if syn.get('syn_type') == 'GENE_SYMBOL' and syn.get('component_synonym') == gene_symbol:
                                return {
                                    'chembl_id': target.get('target_chembl_id'),
                                    'pref_name': target.get('pref_name'),
                                    'organism': target.get('organism')
                                }
                return None
            elif response.status_code == 429:  # Rate limited
                time.sleep(2 ** attempt)
                continue
            else:
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    return None


def classify_druggability(gene_symbol):
    """Classify a gene's druggability based on family patterns."""
    gene_upper = gene_symbol.upper()

    for family, patterns in DRUGGABLE_FAMILIES.items():
        for pattern in patterns:
            if pattern.upper() in gene_upper or gene_upper.startswith(pattern.upper()):
                return family

    return None


def get_known_drug_info(gene_symbol):
    """Get info about known drugs targeting this gene."""
    return KNOWN_DRUGGED_TARGETS.get(gene_symbol.upper())


def add_druggability_annotations(targets_df):
    """Add druggability annotations to targets dataframe."""
    print("Adding druggability annotations...")

    # Initialize new columns
    targets_df['druggable_family'] = None
    targets_df['known_drugs'] = None
    targets_df['drug_status'] = None
    targets_df['in_chembl'] = False
    targets_df['chembl_id'] = None
    targets_df['druggability_score'] = 0  # 0-3 scale

    total = len(targets_df)
    chembl_hits = 0
    family_hits = 0
    known_drug_hits = 0

    for idx, row in targets_df.iterrows():
        gene_symbol = row['gene_symbol']

        # 1. Check for known drugs
        drug_info = get_known_drug_info(gene_symbol)
        if drug_info:
            targets_df.at[idx, 'known_drugs'] = '; '.join(drug_info['drugs'])
            targets_df.at[idx, 'drug_status'] = drug_info['status']
            targets_df.at[idx, 'druggability_score'] = 3  # Highest - has drugs
            known_drug_hits += 1

        # 2. Classify by gene family
        family = classify_druggability(gene_symbol)
        if family:
            targets_df.at[idx, 'druggable_family'] = family
            if targets_df.at[idx, 'druggability_score'] < 2:
                targets_df.at[idx, 'druggability_score'] = 2  # Medium-high - druggable family
            family_hits += 1

        # 3. Query ChEMBL (rate-limited, do in batches)
        if idx < 100:  # Only query top 100 to avoid rate limiting
            chembl_result = query_chembl_for_gene(gene_symbol)
            if chembl_result:
                targets_df.at[idx, 'in_chembl'] = True
                targets_df.at[idx, 'chembl_id'] = chembl_result['chembl_id']
                if targets_df.at[idx, 'druggability_score'] < 1:
                    targets_df.at[idx, 'druggability_score'] = 1  # In ChEMBL
                chembl_hits += 1

            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{min(100, total)} genes (ChEMBL queries)...")
                time.sleep(0.5)  # Rate limiting

    print(f"\nDruggability summary:")
    print(f"  Known drugs: {known_drug_hits} genes")
    print(f"  Druggable families: {family_hits} genes")
    print(f"  In ChEMBL (top 100): {chembl_hits} genes")

    return targets_df


def main():
    print("=" * 60)
    print("ADDING DRUGGABILITY ANNOTATIONS")
    print("=" * 60)

    # Load existing targets
    targets_file = DATA_DIR / "pan_cancer_targets_v2.csv"
    if not targets_file.exists():
        targets_file = DATA_DIR / "pan_cancer_targets.csv"

    targets_df = pd.read_csv(targets_file)
    print(f"Loaded {len(targets_df)} targets from {targets_file}")

    # Add annotations
    annotated_df = add_druggability_annotations(targets_df)

    # Save results
    output_file = DATA_DIR / "pan_cancer_targets_druggability.csv"
    annotated_df.to_csv(output_file, index=False)
    print(f"\nSaved annotated targets to {output_file}")

    # Print summary of most druggable targets
    print("\n" + "=" * 80)
    print("TOP 30 MOST DRUGGABLE PAN-CANCER TARGETS")
    print("=" * 80)

    # Sort by druggability score, then by selectivity
    druggable = annotated_df[annotated_df['druggability_score'] > 0].copy()
    druggable = druggable.sort_values(['druggability_score', 'selectivity'], ascending=[False, False])

    cols_to_show = ['gene_symbol', 'frac_cancer_essential', 'selectivity',
                    'druggability_score', 'druggable_family', 'known_drugs', 'drug_status']
    print(druggable[cols_to_show].head(30).to_string(index=False))

    # Count by druggability category
    print("\n" + "=" * 60)
    print("DRUGGABILITY BREAKDOWN")
    print("=" * 60)
    print(f"Total targets: {len(annotated_df)}")
    print(f"With known drugs (score=3): {len(annotated_df[annotated_df['druggability_score'] == 3])}")
    print(f"Druggable family (score=2): {len(annotated_df[annotated_df['druggability_score'] == 2])}")
    print(f"In ChEMBL (score=1): {len(annotated_df[annotated_df['druggability_score'] == 1])}")
    print(f"Unknown druggability (score=0): {len(annotated_df[annotated_df['druggability_score'] == 0])}")

    return annotated_df


if __name__ == "__main__":
    results = main()
