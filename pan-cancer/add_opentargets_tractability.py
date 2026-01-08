"""
Add OpenTargets tractability scores to pan-cancer targets.

Fetches tractability data from OpenTargets Platform API for each gene.
Tractability indicates how druggable a target is across different modalities:
- SM: Small Molecule
- AB: Antibody
- PR: PROTAC (protein degrader)
- OC: Other modalities

Output: Updated table with tractability scores
"""

import pandas as pd
import requests
import json
import time
from pathlib import Path

DATA_DIR = Path("data")
API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Query to search for a gene by symbol
SEARCH_QUERY = """
query searchTarget($symbol: String!) {
  search(queryString: $symbol, entityNames: ["target"], page: {size: 1, index: 0}) {
    hits {
      id
      name
    }
  }
}
"""

# Query to get tractability for a target
TRACTABILITY_QUERY = """
query targetTractability($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    tractability {
      label
      modality
      value
    }
  }
}
"""


def search_gene(symbol):
    """Search for a gene by symbol and return Ensembl ID."""
    try:
        response = requests.post(
            API_URL,
            json={"query": SEARCH_QUERY, "variables": {"symbol": symbol}},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            hits = data.get("data", {}).get("search", {}).get("hits", [])
            if hits:
                return hits[0].get("id")
    except Exception as e:
        print(f"  Error searching {symbol}: {e}")
    return None


def get_tractability(ensembl_id):
    """Get tractability data for a target."""
    try:
        response = requests.post(
            API_URL,
            json={"query": TRACTABILITY_QUERY, "variables": {"ensemblId": ensembl_id}},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            target = data.get("data", {}).get("target", {})
            if target:
                return target.get("tractability", [])
    except Exception as e:
        print(f"  Error getting tractability for {ensembl_id}: {e}")
    return []


def summarize_tractability(tractability_list):
    """Summarize tractability into scores per modality."""
    # Define tractability buckets (higher = more tractable)
    clinical_labels = ["Approved Drug", "Advanced Clinical", "Phase 1 Clinical"]

    summary = {
        "SM_clinical": False,  # Has clinical-stage small molecule
        "SM_structural": False,  # Has structural tractability (pocket/ligand)
        "AB_clinical": False,  # Has clinical-stage antibody
        "AB_accessible": False,  # Is cell-surface accessible
        "PR_tractable": False,  # PROTAC tractable
        "OC_clinical": False,  # Other clinical
    }

    sm_score = 0
    ab_score = 0
    pr_score = 0

    for item in tractability_list:
        label = item.get("label", "")
        modality = item.get("modality", "")
        value = item.get("value", False)

        if not value:
            continue

        if modality == "SM":
            if label in clinical_labels:
                summary["SM_clinical"] = True
                sm_score = max(sm_score, 3 - clinical_labels.index(label))  # 3, 2, or 1
            elif "Ligand" in label or "Pocket" in label:
                summary["SM_structural"] = True
                sm_score = max(sm_score, 1)
            elif label == "Druggable Family":
                sm_score = max(sm_score, 0.5)

        elif modality == "AB":
            if label in clinical_labels:
                summary["AB_clinical"] = True
                ab_score = max(ab_score, 3 - clinical_labels.index(label))
            elif "loc" in label.lower() or "SigP" in label or "TMHMM" in label:
                summary["AB_accessible"] = True
                ab_score = max(ab_score, 1)

        elif modality == "PR":
            if label in clinical_labels:
                pr_score = max(pr_score, 3 - clinical_labels.index(label))
            elif value:  # Any PROTAC evidence
                summary["PR_tractable"] = True
                pr_score = max(pr_score, 0.5)

        elif modality == "OC":
            if label in clinical_labels and value:
                summary["OC_clinical"] = True

    summary["SM_score"] = sm_score
    summary["AB_score"] = ab_score
    summary["PR_score"] = pr_score
    summary["total_score"] = sm_score + ab_score + pr_score

    return summary


def main():
    print("=" * 80)
    print("ADDING OPENTARGETS TRACTABILITY SCORES")
    print("=" * 80)

    # Load top 50 genes
    druggability_file = DATA_DIR / "pan_cancer_targets_druggability.csv"
    df = pd.read_csv(druggability_file)
    df = df.sort_values("selectivity", ascending=False).head(50)

    print(f"Processing top 50 genes by selectivity...")

    # Initialize new columns
    results = []

    for idx, row in df.iterrows():
        symbol = row["gene_symbol"]
        print(f"  {symbol}...", end=" ", flush=True)

        # Search for Ensembl ID
        ensembl_id = search_gene(symbol)

        if ensembl_id:
            # Get tractability
            tractability = get_tractability(ensembl_id)
            summary = summarize_tractability(tractability)
            print(f"found (SM:{summary['SM_score']:.1f} AB:{summary['AB_score']:.1f} PR:{summary['PR_score']:.1f})")
        else:
            print("not found")
            summary = {
                "SM_score": 0, "AB_score": 0, "PR_score": 0, "total_score": 0,
                "SM_clinical": False, "SM_structural": False,
                "AB_clinical": False, "AB_accessible": False,
                "PR_tractable": False, "OC_clinical": False
            }
            ensembl_id = ""

        results.append({
            "gene_symbol": symbol,
            "ensembl_id": ensembl_id,
            "selectivity": row["selectivity"],
            "frac_cancer_essential": row["frac_cancer_essential"],
            "druggability_score": row["druggability_score"],
            "SM_score": summary["SM_score"],
            "AB_score": summary["AB_score"],
            "PR_score": summary["PR_score"],
            "total_tractability": summary["total_score"],
            "SM_clinical": summary["SM_clinical"],
            "SM_structural": summary["SM_structural"],
            "AB_clinical": summary["AB_clinical"],
            "AB_accessible": summary["AB_accessible"],
            "PR_tractable": summary["PR_tractable"],
        })

        time.sleep(0.2)  # Rate limiting

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_file = DATA_DIR / "top50_with_tractability.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Print formatted table
    print("\n" + "=" * 120)
    print("TOP 50 CANCER-SELECTIVE TARGETS WITH OPENTARGETS TRACTABILITY")
    print("=" * 120)
    print("\nTractability scores: SM=Small Molecule, AB=Antibody, PR=PROTAC (0-3 scale, higher=more tractable)")
    print("-" * 120)

    header = f"{'Gene':<10} {'Select':>7} {'%Ess':>6} {'SM':>4} {'AB':>4} {'PR':>4} {'Total':>5} {'SM_clin':>7} {'SM_struct':>9} {'AB_clin':>7} {'AB_acc':>6} {'PR':>4}"
    print(header)
    print("-" * 120)

    for _, row in results_df.iterrows():
        line = f"{row['gene_symbol']:<10} {row['selectivity']:>7.3f} {row['frac_cancer_essential']*100:>5.1f}% {row['SM_score']:>4.1f} {row['AB_score']:>4.1f} {row['PR_score']:>4.1f} {row['total_tractability']:>5.1f} {'Yes' if row['SM_clinical'] else '-':>7} {'Yes' if row['SM_structural'] else '-':>9} {'Yes' if row['AB_clinical'] else '-':>7} {'Yes' if row['AB_accessible'] else '-':>6} {'Yes' if row['PR_tractable'] else '-':>4}"
        print(line)

    # Summary stats
    print("\n" + "=" * 80)
    print("TRACTABILITY SUMMARY")
    print("=" * 80)
    print(f"Genes with SM clinical evidence: {results_df['SM_clinical'].sum()}")
    print(f"Genes with SM structural tractability: {results_df['SM_structural'].sum()}")
    print(f"Genes with AB clinical evidence: {results_df['AB_clinical'].sum()}")
    print(f"Genes with AB accessibility: {results_df['AB_accessible'].sum()}")
    print(f"Genes with PROTAC tractability: {results_df['PR_tractable'].sum()}")

    # Best opportunities (high selectivity + high tractability)
    print("\n" + "=" * 80)
    print("BEST OPPORTUNITIES (selectivity > 0.2 AND total_tractability >= 2)")
    print("=" * 80)
    best = results_df[(results_df['selectivity'] > 0.2) & (results_df['total_tractability'] >= 2)]
    best = best.sort_values('total_tractability', ascending=False)
    for _, row in best.iterrows():
        print(f"  {row['gene_symbol']}: selectivity={row['selectivity']:.3f}, tractability={row['total_tractability']:.1f}")

    return results_df


if __name__ == "__main__":
    results = main()
