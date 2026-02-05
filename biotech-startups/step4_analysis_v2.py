#!/usr/bin/env python3
"""
Step 4 v2: Analysis with 4 distinct outcome categories.

Categories:
1. Acquired
2. Failed (bankrupt, IPO withdrawn, delisted, etc.)
3. Trading (positive CAGR)
4. Trading (negative CAGR)
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
    elif status in ['bankrupt', 'ipo_withdrawn', 'delisted', 'ipo_incomplete', 'unknown', 'never_traded']:
        return 'Failed'
    elif status == 'trading':
        if cagr:
            try:
                cagr_val = float(cagr)
                if cagr_val > 0:
                    return 'Trading+'
                else:
                    return 'Trading-'
            except:
                return None
        else:
            return None
    else:
        return None

def chi_square_test(rows, predictor_field, outcome_categories):
    """Perform chi-square test."""
    contingency = defaultdict(lambda: defaultdict(int))

    for row in rows:
        outcome = categorize_outcome(row)
        if outcome not in outcome_categories:
            continue

        predictor = row.get(predictor_field, '')
        if not predictor or predictor in ['', 'unknown', 'Unknown']:
            continue

        contingency[predictor][outcome] += 1

    predictors = [p for p in contingency.keys() if sum(contingency[p].values()) >= 10]

    if len(predictors) < 2:
        return None, None, None, None

    matrix = []
    for p in sorted(predictors):
        row = [contingency[p][o] for o in outcome_categories]
        matrix.append(row)

    matrix = np.array(matrix)

    try:
        chi2, p, dof, expected = stats.chi2_contingency(matrix)
        return chi2, p, sorted(predictors), contingency
    except:
        return None, None, None, None

def analyze_predictor(rows, predictor_field, predictor_name, outcome_categories):
    """Analyze a single predictor with all 4 categories."""
    print(f"\n{'='*80}")
    print(f"PREDICTOR: {predictor_name}")
    print(f"{'='*80}")

    chi2, p, predictors, contingency = chi_square_test(rows, predictor_field, outcome_categories)

    if chi2 is None:
        print("  Insufficient data for analysis")
        return None

    sig = ""
    if p < 0.001:
        sig = "*** (p < 0.001)"
    elif p < 0.01:
        sig = "** (p < 0.01)"
    elif p < 0.05:
        sig = "* (p < 0.05)"

    print(f"\nChi-square: {chi2:.2f}, p-value: {p:.6f} {sig}")

    # Print contingency table with percentages
    print(f"\n{'Predictor':<22} {'Acquired':>9} {'Failed':>9} {'Trade+':>9} {'Trade-':>9} {'Total':>7}")
    print("-" * 70)

    results = []
    for pred in predictors:
        total = sum(contingency[pred].values())
        acq = contingency[pred].get('Acquired', 0)
        fail = contingency[pred].get('Failed', 0)
        pos = contingency[pred].get('Trading+', 0)
        neg = contingency[pred].get('Trading-', 0)

        print(f"{pred[:22]:<22} {acq:>4} ({acq/total*100:4.1f}%) {fail:>4} ({fail/total*100:4.1f}%) {pos:>4} ({pos/total*100:4.1f}%) {neg:>4} ({neg/total*100:4.1f}%) {total:>7}")

        results.append({
            'predictor': pred,
            'acquired': acq, 'acq_pct': acq/total*100,
            'failed': fail, 'fail_pct': fail/total*100,
            'trading_pos': pos, 'pos_pct': pos/total*100,
            'trading_neg': neg, 'neg_pct': neg/total*100,
            'total': total,
        })

    # Find best/worst for each outcome
    print(f"\nBest/Worst by Outcome:")

    results_sorted = sorted(results, key=lambda x: x['acq_pct'], reverse=True)
    print(f"  Highest Acquired%:  {results_sorted[0]['predictor']} ({results_sorted[0]['acq_pct']:.1f}%)")
    print(f"  Lowest Acquired%:   {results_sorted[-1]['predictor']} ({results_sorted[-1]['acq_pct']:.1f}%)")

    results_sorted = sorted(results, key=lambda x: x['fail_pct'], reverse=True)
    print(f"  Highest Failed%:    {results_sorted[0]['predictor']} ({results_sorted[0]['fail_pct']:.1f}%)")
    print(f"  Lowest Failed%:     {results_sorted[-1]['predictor']} ({results_sorted[-1]['fail_pct']:.1f}%)")

    results_sorted = sorted(results, key=lambda x: x['pos_pct'], reverse=True)
    print(f"  Highest Trading+%:  {results_sorted[0]['predictor']} ({results_sorted[0]['pos_pct']:.1f}%)")
    print(f"  Lowest Trading+%:   {results_sorted[-1]['predictor']} ({results_sorted[-1]['pos_pct']:.1f}%)")

    results_sorted = sorted(results, key=lambda x: x['neg_pct'], reverse=True)
    print(f"  Highest Trading-%:  {results_sorted[0]['predictor']} ({results_sorted[0]['neg_pct']:.1f}%)")
    print(f"  Lowest Trading-%:   {results_sorted[-1]['predictor']} ({results_sorted[-1]['neg_pct']:.1f}%)")

    return {'chi2': chi2, 'p': p, 'significant': p < 0.05, 'results': results}

def analyze_founding_year(rows, outcome_categories):
    """Analyze founding year."""
    print(f"\n{'='*80}")
    print(f"PREDICTOR: Founding Year")
    print(f"{'='*80}")

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

    print(f"\n{'Year':<6} {'Acquired':>9} {'Failed':>9} {'Trade+':>9} {'Trade-':>9} {'Total':>7}")
    print("-" * 55)

    years_data = []
    for year in sorted(by_year.keys()):
        total = sum(by_year[year].values())
        if total < 5:
            continue
        acq = by_year[year].get('Acquired', 0)
        fail = by_year[year].get('Failed', 0)
        pos = by_year[year].get('Trading+', 0)
        neg = by_year[year].get('Trading-', 0)

        print(f"{year:<6} {acq:>4} ({acq/total*100:4.1f}%) {fail:>4} ({fail/total*100:4.1f}%) {pos:>4} ({pos/total*100:4.1f}%) {neg:>4} ({neg/total*100:4.1f}%) {total:>7}")
        years_data.append({
            'year': year,
            'acq_pct': acq/total*100,
            'fail_pct': fail/total*100,
            'pos_pct': pos/total*100,
            'neg_pct': neg/total*100,
            'total': total
        })

    # Correlation analysis
    if len(years_data) >= 5:
        years = [y['year'] for y in years_data]

        print(f"\nSpearman correlations with founding year:")
        for outcome, key in [('Acquired', 'acq_pct'), ('Failed', 'fail_pct'),
                             ('Trading+', 'pos_pct'), ('Trading-', 'neg_pct')]:
            vals = [y[key] for y in years_data]
            r, p = stats.spearmanr(years, vals)
            sig = '*' if p < 0.05 else ''
            print(f"  {outcome:12} r={r:+.3f}, p={p:.4f} {sig}")

def analyze_state(rows, outcome_categories):
    """Analyze state/geography."""
    print(f"\n{'='*80}")
    print(f"PREDICTOR: State (Geography)")
    print(f"{'='*80}")

    by_state = defaultdict(lambda: defaultdict(int))

    for row in rows:
        outcome = categorize_outcome(row)
        if outcome not in outcome_categories:
            continue

        state = row.get('state', '')
        if not state or len(state) > 5:  # Filter out weird values
            continue

        by_state[state][outcome] += 1

    states = [(s, sum(by_state[s].values())) for s in by_state]
    states = [(s, n) for s, n in states if n >= 15]
    states.sort(key=lambda x: -x[1])

    print(f"\n{'State':<6} {'Acquired':>9} {'Failed':>9} {'Trade+':>9} {'Trade-':>9} {'Total':>7}")
    print("-" * 55)

    for state, total in states[:12]:
        acq = by_state[state].get('Acquired', 0)
        fail = by_state[state].get('Failed', 0)
        pos = by_state[state].get('Trading+', 0)
        neg = by_state[state].get('Trading-', 0)

        print(f"{state:<6} {acq:>4} ({acq/total*100:4.1f}%) {fail:>4} ({fail/total*100:4.1f}%) {pos:>4} ({pos/total*100:4.1f}%) {neg:>4} ({neg/total*100:4.1f}%) {total:>7}")

    chi2, p, _, _ = chi_square_test(rows, 'state', outcome_categories)
    if chi2:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"\nChi-square: {chi2:.2f}, p-value: {p:.6f} {sig}")

def analyze_targets(rows, outcome_categories):
    """Analyze drug targets."""
    print(f"\n{'='*80}")
    print(f"PREDICTOR: Drug Targets")
    print(f"{'='*80}")

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
            if target and len(target) > 1 and len(target) < 20:
                target_outcomes[target][outcome] += 1

    targets = [(t, sum(target_outcomes[t].values())) for t in target_outcomes]
    targets = [(t, n) for t, n in targets if n >= 10]
    targets.sort(key=lambda x: -x[1])

    print(f"\n{'Target':<12} {'Acquired':>9} {'Failed':>9} {'Trade+':>9} {'Trade-':>9} {'Total':>7}")
    print("-" * 60)

    results = []
    for target, total in targets[:15]:
        acq = target_outcomes[target].get('Acquired', 0)
        fail = target_outcomes[target].get('Failed', 0)
        pos = target_outcomes[target].get('Trading+', 0)
        neg = target_outcomes[target].get('Trading-', 0)

        print(f"{target:<12} {acq:>4} ({acq/total*100:4.1f}%) {fail:>4} ({fail/total*100:4.1f}%) {pos:>4} ({pos/total*100:4.1f}%) {neg:>4} ({neg/total*100:4.1f}%) {total:>7}")
        results.append({
            'target': target,
            'acq_pct': acq/total*100,
            'fail_pct': fail/total*100,
            'pos_pct': pos/total*100,
            'total': total
        })

    # Best targets
    print(f"\nTop targets by Acquired%:")
    for r in sorted(results, key=lambda x: x['acq_pct'], reverse=True)[:5]:
        print(f"  {r['target']}: {r['acq_pct']:.1f}% (n={r['total']})")

    print(f"\nTop targets by Trading+%:")
    for r in sorted(results, key=lambda x: x['pos_pct'], reverse=True)[:5]:
        print(f"  {r['target']}: {r['pos_pct']:.1f}% (n={r['total']})")

def main():
    print("=" * 80)
    print("BIOTECH STARTUP OUTCOME ANALYSIS (4 Categories)")
    print("=" * 80)

    rows = load_data()
    print(f"\nTotal companies: {len(rows)}")

    outcome_categories = ['Acquired', 'Failed', 'Trading+', 'Trading-']

    # Categorize all companies
    outcomes = Counter(categorize_outcome(row) for row in rows)

    print("\nOutcome Distribution:")
    print("-" * 40)
    total_valid = sum(outcomes[o] for o in outcome_categories)
    for outcome in outcome_categories:
        count = outcomes[outcome]
        pct = count / total_valid * 100
        print(f"  {outcome:<20} {count:>5} ({pct:5.1f}%)")
    print(f"  {'TOTAL':<20} {total_valid:>5}")

    # Analyze each predictor
    results = {}
    results['disease'] = analyze_predictor(rows, 'primary_disease_area', 'Disease Area', outcome_categories)
    results['modality'] = analyze_predictor(rows, 'modality_category', 'Modality Category', outcome_categories)
    results['modality_sub'] = analyze_predictor(rows, 'modality_subtype', 'Modality Subtype', outcome_categories)
    results['stage'] = analyze_predictor(rows, 'lead_stage', 'Lead Stage at IPO', outcome_categories)

    analyze_founding_year(rows, outcome_categories)
    analyze_state(rows, outcome_categories)
    analyze_targets(rows, outcome_categories)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: SIGNIFICANT PREDICTORS (p < 0.05)")
    print("=" * 80)

    for name, result in results.items():
        if result:
            sig = "***" if result['p'] < 0.001 else "**" if result['p'] < 0.01 else "*" if result['p'] < 0.05 else ""
            print(f"  {name:15} Chi2={result['chi2']:6.1f}, p={result['p']:.6f} {sig}")

if __name__ == "__main__":
    main()
