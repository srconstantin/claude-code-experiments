#!/usr/bin/env python3
"""
Step 4: Correlational analysis of biotech company outcomes.

Categories:
- Acquired: Companies that were acquired
- Failed: Bankrupt, IPO withdrawn, delisted, IPO incomplete
- Trading (positive CAGR): Still trading with positive annualized return
- Trading (negative CAGR): Still trading with negative annualized return

Predictors:
- Disease area
- Modality category
- Modality subtype
- Lead stage at IPO
- Founding year / IPO timing
- State (geography)
- Targets
"""

import csv
from collections import Counter, defaultdict
from scipy import stats
import numpy as np

def load_data():
    with open('step3_pipeline_info.csv', 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def categorize_outcome(row):
    """Categorize company into one of 4 outcomes."""
    status = row.get('status', '')
    cagr = row.get('cagr_pct', '')

    if status == 'acquired':
        return 'Acquired'
    elif status in ['bankrupt', 'ipo_withdrawn', 'delisted', 'ipo_incomplete', 'unknown']:
        return 'Failed'
    elif status == 'trading':
        if cagr:
            try:
                cagr_val = float(cagr)
                if cagr_val > 0:
                    return 'Trading (positive)'
                else:
                    return 'Trading (negative)'
            except:
                return 'Trading (unknown)'
        else:
            return 'Trading (unknown)'
    elif status == 'never_traded':
        return 'Failed'  # Treat as failed
    else:
        return 'Unknown'

def chi_square_test(rows, predictor_field, outcome_categories):
    """
    Perform chi-square test for categorical predictor vs outcome.
    Returns chi2, p-value, and contingency table.
    """
    # Build contingency table
    contingency = defaultdict(lambda: defaultdict(int))

    for row in rows:
        outcome = categorize_outcome(row)
        if outcome not in outcome_categories:
            continue

        predictor = row.get(predictor_field, '')
        if not predictor or predictor in ['', 'unknown', 'Unknown']:
            continue

        contingency[predictor][outcome] += 1

    # Convert to matrix
    predictors = sorted(contingency.keys())
    outcomes = outcome_categories

    # Filter to predictors with at least 10 companies
    predictors = [p for p in predictors if sum(contingency[p].values()) >= 10]

    if len(predictors) < 2:
        return None, None, None, None

    matrix = []
    for p in predictors:
        row = [contingency[p][o] for o in outcomes]
        matrix.append(row)

    matrix = np.array(matrix)

    # Chi-square test
    try:
        chi2, p, dof, expected = stats.chi2_contingency(matrix)
        return chi2, p, predictors, contingency
    except:
        return None, None, None, None

def analyze_predictor(rows, predictor_field, predictor_name, outcome_categories):
    """Analyze a single predictor."""
    print(f"\n{'='*70}")
    print(f"PREDICTOR: {predictor_name}")
    print(f"{'='*70}")

    chi2, p, predictors, contingency = chi_square_test(rows, predictor_field, outcome_categories)

    if chi2 is None:
        print("  Insufficient data for analysis")
        return None

    print(f"\nChi-square: {chi2:.2f}, p-value: {p:.4f}")
    if p < 0.001:
        print("  *** Highly significant (p < 0.001)")
    elif p < 0.01:
        print("  ** Significant (p < 0.01)")
    elif p < 0.05:
        print("  * Significant (p < 0.05)")
    else:
        print("  Not significant (p >= 0.05)")

    # Print contingency table
    print(f"\n{'Predictor':<25} {'Acquired':>10} {'Failed':>10} {'Trade(+)':>10} {'Trade(-)':>10} {'Total':>8} {'Acq%':>7} {'Fail%':>7}")
    print("-" * 100)

    results = []
    for pred in predictors:
        total = sum(contingency[pred].values())
        acq = contingency[pred].get('Acquired', 0)
        fail = contingency[pred].get('Failed', 0)
        pos = contingency[pred].get('Trading (positive)', 0)
        neg = contingency[pred].get('Trading (negative)', 0)

        acq_pct = acq / total * 100 if total > 0 else 0
        fail_pct = fail / total * 100 if total > 0 else 0

        print(f"{pred[:25]:<25} {acq:>10} {fail:>10} {pos:>10} {neg:>10} {total:>8} {acq_pct:>6.1f}% {fail_pct:>6.1f}%")
        results.append({
            'predictor': pred,
            'acquired': acq,
            'failed': fail,
            'trading_pos': pos,
            'trading_neg': neg,
            'total': total,
            'acq_pct': acq_pct,
            'fail_pct': fail_pct,
        })

    # Sort by acquisition rate and show best/worst
    results.sort(key=lambda x: x['acq_pct'], reverse=True)
    print(f"\nHighest acquisition rate: {results[0]['predictor']} ({results[0]['acq_pct']:.1f}%)")
    print(f"Lowest acquisition rate: {results[-1]['predictor']} ({results[-1]['acq_pct']:.1f}%)")

    results.sort(key=lambda x: x['fail_pct'])
    print(f"Lowest failure rate: {results[0]['predictor']} ({results[0]['fail_pct']:.1f}%)")
    print(f"Highest failure rate: {results[-1]['predictor']} ({results[-1]['fail_pct']:.1f}%)")

    return {'chi2': chi2, 'p': p, 'significant': p < 0.05}

def analyze_founding_year(rows, outcome_categories):
    """Analyze founding year as a predictor."""
    print(f"\n{'='*70}")
    print(f"PREDICTOR: Founding Year")
    print(f"{'='*70}")

    # Group by founding year
    by_year = defaultdict(lambda: defaultdict(int))

    for row in rows:
        outcome = categorize_outcome(row)
        if outcome not in outcome_categories:
            continue

        founding = row.get('founding_date', '')
        if not founding:
            continue

        try:
            year = int(founding[:4])
            if 2000 <= year <= 2024:
                by_year[year][outcome] += 1
        except:
            continue

    # Print by year
    print(f"\n{'Year':<8} {'Acquired':>10} {'Failed':>10} {'Trade(+)':>10} {'Trade(-)':>10} {'Total':>8} {'Acq%':>7} {'Fail%':>7}")
    print("-" * 90)

    years_data = []
    for year in sorted(by_year.keys()):
        total = sum(by_year[year].values())
        if total < 5:
            continue
        acq = by_year[year].get('Acquired', 0)
        fail = by_year[year].get('Failed', 0)
        pos = by_year[year].get('Trading (positive)', 0)
        neg = by_year[year].get('Trading (negative)', 0)

        acq_pct = acq / total * 100 if total > 0 else 0
        fail_pct = fail / total * 100 if total > 0 else 0

        print(f"{year:<8} {acq:>10} {fail:>10} {pos:>10} {neg:>10} {total:>8} {acq_pct:>6.1f}% {fail_pct:>6.1f}%")
        years_data.append((year, acq_pct, fail_pct, total))

    # Correlation analysis
    if len(years_data) >= 5:
        years = [y[0] for y in years_data]
        acq_pcts = [y[1] for y in years_data]
        fail_pcts = [y[2] for y in years_data]

        # Spearman correlation
        r_acq, p_acq = stats.spearmanr(years, acq_pcts)
        r_fail, p_fail = stats.spearmanr(years, fail_pcts)

        print(f"\nCorrelation with founding year:")
        print(f"  Acquisition rate: r={r_acq:.3f}, p={p_acq:.4f} {'*' if p_acq < 0.05 else ''}")
        print(f"  Failure rate: r={r_fail:.3f}, p={p_fail:.4f} {'*' if p_fail < 0.05 else ''}")

def analyze_state(rows, outcome_categories):
    """Analyze state/geography as a predictor."""
    print(f"\n{'='*70}")
    print(f"PREDICTOR: State (Geography)")
    print(f"{'='*70}")

    # Consolidate states into regions
    state_to_region = {
        'CA': 'California',
        'MA': 'Massachusetts',
        'NJ': 'NJ/PA/NY Metro',
        'NY': 'NJ/PA/NY Metro',
        'PA': 'NJ/PA/NY Metro',
        'TX': 'Texas',
        'MD': 'DC Metro',
        'DC': 'DC Metro',
        'VA': 'DC Metro',
    }

    # Also analyze top states individually
    by_state = defaultdict(lambda: defaultdict(int))

    for row in rows:
        outcome = categorize_outcome(row)
        if outcome not in outcome_categories:
            continue

        state = row.get('state', '')
        if not state:
            continue

        by_state[state][outcome] += 1

    # Filter to states with at least 15 companies
    states = [(s, sum(by_state[s].values())) for s in by_state]
    states = [(s, n) for s, n in states if n >= 15]
    states.sort(key=lambda x: -x[1])

    print(f"\n{'State':<8} {'Acquired':>10} {'Failed':>10} {'Trade(+)':>10} {'Trade(-)':>10} {'Total':>8} {'Acq%':>7} {'Fail%':>7}")
    print("-" * 90)

    for state, total in states[:15]:
        acq = by_state[state].get('Acquired', 0)
        fail = by_state[state].get('Failed', 0)
        pos = by_state[state].get('Trading (positive)', 0)
        neg = by_state[state].get('Trading (negative)', 0)

        acq_pct = acq / total * 100
        fail_pct = fail / total * 100

        print(f"{state:<8} {acq:>10} {fail:>10} {pos:>10} {neg:>10} {total:>8} {acq_pct:>6.1f}% {fail_pct:>6.1f}%")

    # Chi-square on top states
    chi2, p, _, _ = chi_square_test(rows, 'state', outcome_categories)
    if chi2:
        print(f"\nChi-square (all states with n>=10): {chi2:.2f}, p-value: {p:.4f}")

def analyze_targets(rows, outcome_categories):
    """Analyze drug targets as predictors."""
    print(f"\n{'='*70}")
    print(f"PREDICTOR: Drug Targets")
    print(f"{'='*70}")

    # Count target frequency by outcome
    target_outcomes = defaultdict(lambda: defaultdict(int))

    for row in rows:
        outcome = categorize_outcome(row)
        if outcome not in outcome_categories:
            continue

        targets = row.get('targets', '')
        if not targets:
            continue

        for target in targets.split(','):
            target = target.strip().upper()
            if target and len(target) > 1:
                target_outcomes[target][outcome] += 1

    # Filter to targets with at least 10 companies
    targets = [(t, sum(target_outcomes[t].values())) for t in target_outcomes]
    targets = [(t, n) for t, n in targets if n >= 10]
    targets.sort(key=lambda x: -x[1])

    print(f"\n{'Target':<15} {'Acquired':>10} {'Failed':>10} {'Trade(+)':>10} {'Trade(-)':>10} {'Total':>8} {'Acq%':>7} {'Fail%':>7}")
    print("-" * 95)

    results = []
    for target, total in targets[:20]:
        acq = target_outcomes[target].get('Acquired', 0)
        fail = target_outcomes[target].get('Failed', 0)
        pos = target_outcomes[target].get('Trading (positive)', 0)
        neg = target_outcomes[target].get('Trading (negative)', 0)

        acq_pct = acq / total * 100
        fail_pct = fail / total * 100

        print(f"{target:<15} {acq:>10} {fail:>10} {pos:>10} {neg:>10} {total:>8} {acq_pct:>6.1f}% {fail_pct:>6.1f}%")
        results.append({'target': target, 'acq_pct': acq_pct, 'fail_pct': fail_pct, 'total': total})

    # Best and worst targets
    results.sort(key=lambda x: x['acq_pct'], reverse=True)
    print(f"\nHighest acquisition rate targets (n>=10):")
    for r in results[:5]:
        print(f"  {r['target']}: {r['acq_pct']:.1f}% acquired (n={r['total']})")

def main():
    print("=" * 70)
    print("BIOTECH STARTUP OUTCOME ANALYSIS")
    print("=" * 70)

    rows = load_data()
    print(f"\nTotal companies: {len(rows)}")

    # Define outcome categories
    outcome_categories = ['Acquired', 'Failed', 'Trading (positive)', 'Trading (negative)']

    # Categorize all companies
    outcomes = Counter(categorize_outcome(row) for row in rows)

    print("\nOutcome Distribution:")
    print("-" * 40)
    for outcome, count in outcomes.most_common():
        pct = count / len(rows) * 100
        print(f"  {outcome:<25} {count:>5} ({pct:5.1f}%)")

    # Analyze each predictor
    results = {}

    # 1. Disease Area
    results['disease'] = analyze_predictor(rows, 'primary_disease_area', 'Disease Area', outcome_categories)

    # 2. Modality Category
    results['modality'] = analyze_predictor(rows, 'modality_category', 'Modality Category', outcome_categories)

    # 3. Modality Subtype
    results['modality_sub'] = analyze_predictor(rows, 'modality_subtype', 'Modality Subtype', outcome_categories)

    # 4. Lead Stage at IPO
    results['stage'] = analyze_predictor(rows, 'lead_stage', 'Lead Stage at IPO', outcome_categories)

    # 5. Founding Year
    analyze_founding_year(rows, outcome_categories)

    # 6. State/Geography
    analyze_state(rows, outcome_categories)

    # 7. Drug Targets
    analyze_targets(rows, outcome_categories)

    # Summary of significant predictors
    print("\n" + "=" * 70)
    print("SUMMARY OF SIGNIFICANT PREDICTORS")
    print("=" * 70)

    for name, result in results.items():
        if result and result.get('significant'):
            print(f"  {name}: Chi2={result['chi2']:.1f}, p={result['p']:.4f} *")
        elif result:
            print(f"  {name}: Chi2={result['chi2']:.1f}, p={result['p']:.4f}")
        else:
            print(f"  {name}: Insufficient data")

if __name__ == "__main__":
    main()
