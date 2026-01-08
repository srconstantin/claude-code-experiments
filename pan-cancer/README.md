# Pan-Cancer Target Analysis

Identifying promising broad-spectrum cancer drug targets using DepMap data.

## Goal

Find genes that are:
1. **Pan-essential**: Required for survival across many cancer types
2. **Cancer-selective**: NOT essential in non-cancerous cell lines
3. **Underexplored**: Low publication count relative to their potential
4. **Druggable**: Amenable to small molecule or biologic intervention

## Data Sources

- **DepMap**: Cancer Dependency Map from Broad Institute
  - CRISPR knockout dependency scores (Chronos)
  - Cell line metadata (tissue type, cancer type)
- **ChEMBL**: Druggability annotations
- **PubMed**: Publication counts per gene

## Methods

1. Load DepMap CRISPR dependency scores
2. Identify genes essential across >80% of cancer lines
3. Filter out genes also essential in non-cancerous lines
4. Rank by selectivity (cancer dependency - normal dependency)
5. Cross-reference with druggability and publication data

## Output

A ranked list of promising pan-cancer targets with supporting data.
