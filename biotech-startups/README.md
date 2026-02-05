# Biotech Startup IPO Outcome Analysis

What predicts whether a biotech startup succeeds or fails after going public? This project analyzes 803 US biotech companies that filed S-1 registrations since 2000, linking their pipeline characteristics at IPO to their eventual outcomes.

## Data

All data was collected from public sources:

- **SEC EDGAR**: S-1 filings for company identification, founding dates, and pipeline information
- **yfinance**: Stock prices and trading status
- **Claude API**:  extraction of disease area, modality, drug targets, and clinical stage from S-1 filing text

### Outcome Categories

Companies were classified into 4 outcome categories:

| Outcome | N | % | Definition |
|---------|---|---|------------|
| Trading- | 319 | 39.7% | Currently trading with negative CAGR |
| Failed | 251 | 31.3% | Bankrupt, delisted, IPO withdrawn, or never traded |
| Acquired | 177 | 22.0% | Acquired by another company |
| Trading+ | 56 | 7.0% | Currently trading with positive CAGR |


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


