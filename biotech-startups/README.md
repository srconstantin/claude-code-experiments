# Biotech Startup IPO Outcome Analysis

What predicts whether a biotech startup succeeds or fails after going public? This project analyzes 803 US biotech companies that filed S-1 registrations since 2000, linking their pipeline characteristics at IPO to their eventual outcomes.

## Data

All data was collected from public sources with no paid databases:

- **SEC EDGAR**: S-1 filings for company identification, founding dates, and pipeline information
- **yfinance**: Stock prices and trading status
- **Claude API**: NLP extraction of disease area, modality, drug targets, and clinical stage from S-1 filing text

### Outcome Categories

Companies were classified into 4 outcome categories:

| Outcome | N | % | Definition |
|---------|---|---|------------|
| Trading- | 319 | 39.7% | Currently trading with negative CAGR |
| Failed | 251 | 31.3% | Bankrupt, delisted, IPO withdrawn, or never traded |
| Acquired | 177 | 22.0% | Acquired by another company |
| Trading+ | 56 | 7.0% | Currently trading with positive CAGR |

Only 7% of biotech IPOs since 2000 are currently trading with positive annualized returns.

## Key Findings

### Multinomial Logistic Regression

A multinomial logistic regression modeled all 4 outcomes simultaneously (reference category: Failed). Predictors included disease area, modality, clinical stage at IPO, geography (state), and founding year.

**Significant predictors (p < 0.05):**

| Predictor | Acquired vs Failed | Trading+ vs Failed | Trading- vs Failed |
|-----------|-------------------|-------------------|-------------------|
| Founding year (per year newer) | 1.04x* | 1.28x*** | 1.12x*** |
| Outside major hubs | 0.40x*** | 0.15x*** | 0.62x* |
| Rare disease | 2.36x* | 5.24x** | -- |
| Immunology | -- | 4.45x* | -- |
| Phase 3 at IPO | 1.90x* | -- | -- |
| "Other" modality (non-drug) | 0.44x* | 0.19x* | -- |

### Detailed Field Analysis (Chi-Square)

Analyzing multi-valued fields (disease areas, modality subtypes, and drug targets) for significantly different outcome distributions:

| Field | Value | N | p-value | Key Pattern |
|-------|-------|---|---------|-------------|
| Disease | Rare disease | 166 | <0.0001 | Lower failure (18.7% vs 31.3%), higher Trading+ (14.5% vs 7.0%) |
| Modality | Antibody | 78 | 0.0005 | Much lower failure (11.5% vs 31.3%), higher Trading+ (14.1% vs 7.0%) |
| Disease | Immunology | 115 | 0.0019 | Lower failure (20.0% vs 31.3%), higher Trading+ (14.8% vs 7.0%) |
| Target | PD-1 | 27 | 0.025 | Much higher acquisition (40.7% vs 22.0%), very low failure (7.4% vs 31.3%) |

### Summary of Results

1. **Geography is the strongest predictor.** Companies outside the major biotech hubs (CA, MA, NY/NJ/PA) have dramatically worse outcomes across all categories. The effect is strongest for Trading+, where non-hub companies have only 0.15x the odds of achieving positive returns.

2. **Rare disease and immunology companies outperform.** Rare disease companies have 5.2x the odds of Trading+ vs Failed, and immunology companies have 4.5x. Both disease areas show lower failure rates and higher positive trading rates than baseline.

3. **Antibody-based therapeutics have the best modality-level outcomes.** Only 11.5% failure rate vs 31.3% baseline, with double the rate of positive trading.

4. **More recently founded companies do better.** Each additional year of founding date increases Trading+ odds by 28% vs Failed, likely reflecting survivorship bias and improving biotech fundamentals over time.

5. **Phase 3 stage at IPO helps acquisition but not stock performance.** Having a Phase 3 asset at IPO nearly doubles acquisition odds (1.90x) but doesn't significantly predict trading outcomes.

6. **Biologic vs small molecule: no significant difference.** After controlling for other factors, the broad modality category (biologic vs small molecule) does not significantly predict outcomes. The signal is in specific subtypes (e.g., antibodies).

7. **PD-1 targeting companies had high acquisition rates** (40.7% vs 22.0% baseline), consistent with the wave of immuno-oncology acquisitions.

## Pipeline

The analysis was built in 4 steps:

### Step 1: Identify Biotech IPOs
- Query SEC EDGAR for S-1 filings in SIC codes 2834 (pharmaceutical) and 2836 (biological products)
- Extract founding dates from filing text via regex patterns
- Recovery script for missing founding dates with expanded patterns
- **Result: 823 companies founded since 2000**

### Step 2: Get Company Status and Prices
- Look up current trading status and stock prices via yfinance
- For companies without tickers, search SEC 8-K filings for exit information (acquisition, bankruptcy, etc.)
- Calculate CAGR (compound annual growth rate) from IPO to present
- **Result: Status determined for all 823 companies**

### Step 3: Extract Pipeline Information
- Download S-1 filing text from SEC EDGAR
- Extract business description section via regex
- Use Claude API (Sonnet) to extract structured pipeline data: disease area, modality, drug targets, clinical stage
- **Result: Pipeline data for ~82% of companies**

### Step 4: Statistical Analysis
- Chi-square tests for categorical predictors vs 4 outcome categories
- Binary logistic regressions (Acquired vs Failed; Trading+ vs Trading-)
- Multinomial logistic regression across all 4 categories
- Detailed analysis of multi-valued fields (disease areas, modality details, targets)

## Files

| File | Description |
|------|-------------|
| `step1_get_biotech_ipos.py` | Query SEC EDGAR for biotech S-1 filings |
| `step1b_recover_founding_dates.py` | Recover missing founding dates with expanded patterns |
| `step2_get_status_and_prices.py` | Get trading status and stock prices via yfinance |
| `step2b_find_exit_info.py` | Search SEC filings for exit info (acquisitions, bankruptcies) |
| `step3_extract_pipeline_claude.py` | Extract pipeline info from S-1 filings using Claude API |
| `step3_extract_pipeline_info.py` | Earlier regex-based extraction (superseded by Claude version) |
| `step4_analysis.py` | Chi-square analysis (original version) |
| `step4_analysis_v2.py` | Chi-square analysis with 4 outcome categories |
| `step4_logistic_regression.py` | Binary logistic regressions (original) |
| `step4_logistic_regression_v2.py` | Binary logistic regressions with corrected modality |
| `step4_analyze_detailed_fields.py` | Analysis of multi-valued fields across outcomes |
| `step4_multinomial_regression.py` | Multinomial logistic regression (all 4 categories) |
| `step3_pipeline_info.csv` | Final merged dataset (823 companies) |
| `step4_logistic_regression_output.txt` | Binary logistic regression results |
| `step4_detailed_field_analysis.txt` | Detailed field analysis results |
| `step4_multinomial_regression_output.txt` | Multinomial regression results |

## Requirements

```
requests
pandas
numpy
scipy
statsmodels
yfinance
anthropic
```

## Notes

- All data is from public sources (SEC EDGAR, Yahoo Finance)
- Pipeline extraction used Claude Sonnet via the Anthropic API
- The analysis covers companies that filed S-1 registrations, not all biotech companies
- CAGR is calculated from IPO price to most recent price, annualized
- "Failed" includes bankrupt, delisted, IPO withdrawn, IPO incomplete, and never-traded companies
- Companies with unknown trading status or missing CAGR data were excluded from analysis
