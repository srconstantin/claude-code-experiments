"""
Pan-Cancer Target Analysis

Identifies genes that are:
1. Essential across many cancer types (pan-essential)
2. NOT essential in non-cancerous cell lines (cancer-selective)
3. Not already "common essential" (essential in ALL cells including normal)

Output: Ranked list of promising pan-cancer drug targets
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
CANCER_DEPENDENCY_THRESHOLD = -0.5  # More negative = more essential
FRACTION_CANCER_LINES_DEPENDENT = 0.5  # Gene must be essential in >50% of cancer lines
NORMAL_DEPENDENCY_THRESHOLD = -0.3  # If essential in normal cells, exclude

def load_data():
    """Load all required datasets."""
    print("Loading data...")

    # Gene dependency scores (negative = essential)
    gene_effect = pd.read_csv(DATA_DIR / "CRISPRGeneEffect.csv", index_col=0)
    print(f"  Gene effect matrix: {gene_effect.shape[0]} cell lines x {gene_effect.shape[1]} genes")

    # Cell line metadata
    model = pd.read_csv(DATA_DIR / "Model.csv")
    print(f"  Model metadata: {len(model)} cell lines")

    # Common essential genes (to exclude - these kill all cells)
    common_essential = pd.read_csv(DATA_DIR / "AchillesCommonEssentialControls.csv", header=None)
    common_essential = set(common_essential[0].tolist())
    print(f"  Common essential genes (to exclude): {len(common_essential)}")

    # Non-essential genes (reference)
    nonessential = pd.read_csv(DATA_DIR / "AchillesNonessentialControls.csv", header=None)
    nonessential = set(nonessential[0].tolist())
    print(f"  Non-essential genes (reference): {len(nonessential)}")

    return gene_effect, model, common_essential, nonessential


def classify_cell_lines(model, gene_effect):
    """Classify cell lines as cancer vs non-cancer."""
    # Non-cancer lines: Fibroblast, Normal, or Non-Cancerous disease
    non_cancer_mask = (
        (model['OncotreeLineage'].isin(['Fibroblast', 'Normal'])) |
        (model['OncotreePrimaryDisease'] == 'Non-Cancerous')
    )

    non_cancer_ids = set(model.loc[non_cancer_mask, 'ModelID'])
    cancer_ids = set(model.loc[~non_cancer_mask, 'ModelID'])

    # Filter to cell lines that are in the gene effect matrix
    available_ids = set(gene_effect.index)
    non_cancer_ids = non_cancer_ids & available_ids
    cancer_ids = cancer_ids & available_ids

    print(f"\nCell line classification:")
    print(f"  Cancer cell lines (in matrix): {len(cancer_ids)}")
    print(f"  Non-cancer cell lines (in matrix): {len(non_cancer_ids)}")

    return cancer_ids, non_cancer_ids


def find_pan_essential_genes(gene_effect, cancer_ids, threshold, min_fraction):
    """Find genes essential in many cancer cell lines."""
    cancer_data = gene_effect.loc[list(cancer_ids)]

    # Count how many cancer lines each gene is essential in
    is_essential = cancer_data < threshold
    essential_counts = is_essential.sum(axis=0)
    essential_fraction = essential_counts / len(cancer_ids)

    # Filter to genes essential in >= min_fraction of cancer lines
    pan_essential = essential_fraction[essential_fraction >= min_fraction]

    print(f"\nPan-essential genes (essential in >={min_fraction*100:.0f}% of cancer lines):")
    print(f"  Found {len(pan_essential)} genes")

    return pan_essential.sort_values(ascending=False)


def filter_cancer_selective(gene_effect, pan_essential, non_cancer_ids, normal_threshold):
    """Remove genes that are also essential in non-cancer cells."""
    if len(non_cancer_ids) == 0:
        print("  Warning: No non-cancer cell lines available for filtering!")
        return pan_essential, pd.Series(dtype=float)

    non_cancer_data = gene_effect.loc[list(non_cancer_ids)]

    # Calculate mean dependency in non-cancer lines
    mean_normal_effect = non_cancer_data.mean(axis=0)

    # Keep genes that are NOT essential in normal cells
    genes_to_check = pan_essential.index.intersection(mean_normal_effect.index)
    cancer_selective_mask = mean_normal_effect[genes_to_check] > normal_threshold

    cancer_selective_genes = pan_essential[genes_to_check[cancer_selective_mask]]
    excluded_genes = pan_essential[genes_to_check[~cancer_selective_mask]]

    print(f"\nCancer-selective filtering (mean normal effect > {normal_threshold}):")
    print(f"  Kept {len(cancer_selective_genes)} genes (not essential in normal)")
    print(f"  Excluded {len(excluded_genes)} genes (also essential in normal)")

    return cancer_selective_genes, mean_normal_effect


def remove_common_essentials(cancer_selective, common_essential):
    """Remove genes that are common essentials (essential everywhere)."""
    # Parse gene names (format: "GENE (ENTREZ)")
    gene_names = [g.split(' ')[0] for g in cancer_selective.index]

    mask = [g not in common_essential for g in gene_names]
    filtered = cancer_selective[mask]

    n_removed = len(cancer_selective) - len(filtered)
    print(f"\nRemoved {n_removed} common essential genes")
    print(f"  Remaining candidates: {len(filtered)}")

    return filtered


def compute_selectivity_score(gene_effect, genes, cancer_ids, non_cancer_ids):
    """Compute selectivity score: cancer_effect - normal_effect (more negative in cancer = better)."""
    cancer_data = gene_effect.loc[list(cancer_ids)]

    cancer_mean = cancer_data[genes.index].mean(axis=0)

    if len(non_cancer_ids) > 0:
        normal_data = gene_effect.loc[list(non_cancer_ids)]
        normal_mean = normal_data[genes.index].mean(axis=0)
        selectivity = normal_mean - cancer_mean  # Positive = more essential in cancer
    else:
        selectivity = -cancer_mean  # Just use cancer essentiality

    return selectivity


def analyze_by_lineage(gene_effect, model, top_genes, cancer_ids):
    """Show which cancer types each gene is most essential in."""
    # Get lineage for each cell line
    lineage_map = model.set_index('ModelID')['OncotreeLineage'].to_dict()

    cancer_data = gene_effect.loc[list(cancer_ids)]
    cancer_data['Lineage'] = cancer_data.index.map(lineage_map)

    results = {}
    for gene in top_genes[:10]:  # Top 10 genes
        lineage_means = cancer_data.groupby('Lineage')[gene].mean()
        most_dependent = lineage_means.nsmallest(5)  # Most negative = most dependent
        results[gene] = most_dependent

    return results


def main():
    print("=" * 60)
    print("PAN-CANCER TARGET ANALYSIS")
    print("=" * 60)

    # Load data
    gene_effect, model, common_essential, nonessential = load_data()

    # Classify cell lines
    cancer_ids, non_cancer_ids = classify_cell_lines(model, gene_effect)

    # Find pan-essential genes
    pan_essential = find_pan_essential_genes(
        gene_effect, cancer_ids,
        threshold=CANCER_DEPENDENCY_THRESHOLD,
        min_fraction=FRACTION_CANCER_LINES_DEPENDENT
    )

    # Filter for cancer selectivity
    cancer_selective, normal_effects = filter_cancer_selective(
        gene_effect, pan_essential, non_cancer_ids,
        normal_threshold=NORMAL_DEPENDENCY_THRESHOLD
    )

    # Remove common essentials
    final_candidates = remove_common_essentials(cancer_selective, common_essential)

    # Compute selectivity scores
    selectivity = compute_selectivity_score(
        gene_effect, final_candidates, cancer_ids, non_cancer_ids
    )

    # Create final output
    results = pd.DataFrame({
        'gene': final_candidates.index,
        'fraction_cancer_dependent': final_candidates.values,
        'selectivity_score': selectivity[final_candidates.index].values,
        'mean_cancer_effect': gene_effect.loc[list(cancer_ids)][final_candidates.index].mean().values,
        'mean_normal_effect': gene_effect.loc[list(non_cancer_ids)][final_candidates.index].mean().values if len(non_cancer_ids) > 0 else np.nan
    })

    # Sort by selectivity (higher = better)
    results = results.sort_values('selectivity_score', ascending=False)

    # Save results
    results.to_csv(DATA_DIR / "pan_cancer_targets.csv", index=False)

    print("\n" + "=" * 60)
    print("TOP 30 PAN-CANCER SELECTIVE TARGETS")
    print("=" * 60)
    print("\nHigher selectivity = more essential in cancer vs normal")
    print(results.head(30).to_string(index=False))

    # Lineage analysis for top genes
    print("\n" + "=" * 60)
    print("LINEAGE ANALYSIS (which cancer types depend most on each gene)")
    print("=" * 60)
    lineage_results = analyze_by_lineage(gene_effect, model, results['gene'].tolist(), cancer_ids)
    for gene, lineages in lineage_results.items():
        gene_name = gene.split(' ')[0]
        print(f"\n{gene_name}:")
        for lineage, effect in lineages.items():
            print(f"  {lineage}: {effect:.3f}")

    print(f"\n\nResults saved to: {DATA_DIR / 'pan_cancer_targets.csv'}")

    return results


if __name__ == "__main__":
    results = main()
