#!/usr/bin/env python3
"""
Analyze disease_areas, modality_details, and targets fields
across 4 outcome categories.
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
                if float(cagr) > 0:
                    return 'Trading+'
                else:
                    return 'Trading-'
            except:
                return None
        return None
    return None

def parse_list_field(value):
    """Parse comma-separated field into list of normalized values."""
    if not value:
        return []
    items = [item.strip().lower() for item in value.split(',')]
    return [item for item in items if item and len(item) > 1]

def analyze_field(rows, field_name, field_label, min_count=15):
    """Analyze a multi-valued field across outcome categories."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {field_label}")
    print(f"{'='*80}")

    outcome_categories = ['Acquired', 'Failed', 'Trading+', 'Trading-']

    # Count occurrences by value and outcome
    value_outcomes = defaultdict(lambda: defaultdict(int))
    total_by_outcome = Counter()

    for row in rows:
        outcome = categorize_outcome(row)
        if outcome not in outcome_categories:
            continue

        total_by_outcome[outcome] += 1
        values = parse_list_field(row.get(field_name, ''))

        for value in values:
            value_outcomes[value][outcome] += 1

    # Filter to values with minimum count
    values_to_analyze = []
    for value, outcomes in value_outcomes.items():
        total = sum(outcomes.values())
        if total >= min_count:
            values_to_analyze.append((value, total, outcomes))

    values_to_analyze.sort(key=lambda x: -x[1])

    print(f"\nTotal companies by outcome: {dict(total_by_outcome)}")
    print(f"Values with n >= {min_count}: {len(values_to_analyze)}")

    # Analyze each value
    results = []

    for value, total, outcomes in values_to_analyze:
        # Calculate percentages
        acq = outcomes.get('Acquired', 0)
        fail = outcomes.get('Failed', 0)
        pos = outcomes.get('Trading+', 0)
        neg = outcomes.get('Trading-', 0)

        # Chi-square test: does this value have different outcome distribution than overall?
        # Compare observed vs expected based on overall rates
        observed = [acq, fail, pos, neg]

        # Expected counts based on overall distribution
        total_all = sum(total_by_outcome.values())
        expected = [
            total * total_by_outcome['Acquired'] / total_all,
            total * total_by_outcome['Failed'] / total_all,
            total * total_by_outcome['Trading+'] / total_all,
            total * total_by_outcome['Trading-'] / total_all,
        ]

        # Chi-square test
        try:
            # Only test if expected counts are reasonable
            if min(expected) >= 1:
                chi2, p = stats.chisquare(observed, expected)
            else:
                chi2, p = None, None
        except:
            chi2, p = None, None

        results.append({
            'value': value,
            'total': total,
            'acquired': acq,
            'acq_pct': acq/total*100,
            'failed': fail,
            'fail_pct': fail/total*100,
            'trading_pos': pos,
            'pos_pct': pos/total*100,
            'trading_neg': neg,
            'neg_pct': neg/total*100,
            'chi2': chi2,
            'p': p,
        })

    # Print table header
    print(f"\n{'Value':<25} {'N':>5} {'Acq%':>7} {'Fail%':>7} {'Tr+%':>7} {'Tr-%':>7} {'Chi2':>8} {'p':>10}")
    print("-" * 90)

    # Sort by p-value for display
    results_by_p = sorted([r for r in results if r['p'] is not None], key=lambda x: x['p'])
    results_no_p = [r for r in results if r['p'] is None]

    for r in results_by_p + results_no_p:
        sig = ''
        if r['p'] is not None:
            if r['p'] < 0.001:
                sig = '***'
            elif r['p'] < 0.01:
                sig = '**'
            elif r['p'] < 0.05:
                sig = '*'
            elif r['p'] < 0.1:
                sig = '.'

        chi2_str = f"{r['chi2']:.2f}" if r['chi2'] is not None else "N/A"
        p_str = f"{r['p']:.4f}" if r['p'] is not None else "N/A"

        print(f"{r['value'][:25]:<25} {r['total']:>5} {r['acq_pct']:>6.1f}% {r['fail_pct']:>6.1f}% {r['pos_pct']:>6.1f}% {r['neg_pct']:>6.1f}% {chi2_str:>8} {p_str:>10} {sig}")

    # Overall baseline rates
    total_all = sum(total_by_outcome.values())
    print(f"\n{'BASELINE (all companies)':<25} {total_all:>5} {total_by_outcome['Acquired']/total_all*100:>6.1f}% {total_by_outcome['Failed']/total_all*100:>6.1f}% {total_by_outcome['Trading+']/total_all*100:>6.1f}% {total_by_outcome['Trading-']/total_all*100:>6.1f}%")

    # Highlight significant findings
    significant = [r for r in results if r['p'] is not None and r['p'] < 0.05]

    if significant:
        print(f"\n{'SIGNIFICANT FINDINGS (p < 0.05)':^80}")
        print("-" * 80)

        for r in significant:
            # Determine what makes this significant
            baseline_acq = total_by_outcome['Acquired'] / total_all * 100
            baseline_fail = total_by_outcome['Failed'] / total_all * 100
            baseline_pos = total_by_outcome['Trading+'] / total_all * 100
            baseline_neg = total_by_outcome['Trading-'] / total_all * 100

            deviations = []
            if abs(r['acq_pct'] - baseline_acq) > 10:
                direction = "higher" if r['acq_pct'] > baseline_acq else "lower"
                deviations.append(f"Acquired {r['acq_pct']:.1f}% vs {baseline_acq:.1f}% baseline ({direction})")
            if abs(r['fail_pct'] - baseline_fail) > 10:
                direction = "higher" if r['fail_pct'] > baseline_fail else "lower"
                deviations.append(f"Failed {r['fail_pct']:.1f}% vs {baseline_fail:.1f}% baseline ({direction})")
            if abs(r['pos_pct'] - baseline_pos) > 5:
                direction = "higher" if r['pos_pct'] > baseline_pos else "lower"
                deviations.append(f"Trading+ {r['pos_pct']:.1f}% vs {baseline_pos:.1f}% baseline ({direction})")
            if abs(r['neg_pct'] - baseline_neg) > 10:
                direction = "higher" if r['neg_pct'] > baseline_neg else "lower"
                deviations.append(f"Trading- {r['neg_pct']:.1f}% vs {baseline_neg:.1f}% baseline ({direction})")

            print(f"\n  {r['value'].upper()} (n={r['total']}, p={r['p']:.4f}):")
            for dev in deviations:
                print(f"    • {dev}")

    return results

def main():
    print("=" * 80)
    print("DETAILED FIELD ANALYSIS ACROSS 4 OUTCOME CATEGORIES")
    print("=" * 80)

    rows = load_data()

    # Analyze disease_areas
    disease_results = analyze_field(rows, 'disease_areas', 'Disease Areas', min_count=15)

    # Analyze modality_details
    modality_results = analyze_field(rows, 'modality_details', 'Modality Details', min_count=15)

    # Analyze targets
    target_results = analyze_field(rows, 'targets', 'Drug Targets', min_count=10)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL SIGNIFICANT FINDINGS (p < 0.05)")
    print("=" * 80)

    all_significant = []
    for field_name, results in [('Disease', disease_results), ('Modality', modality_results), ('Target', target_results)]:
        for r in results:
            if r['p'] is not None and r['p'] < 0.05:
                all_significant.append({
                    'field': field_name,
                    'value': r['value'],
                    'n': r['total'],
                    'p': r['p'],
                    'acq_pct': r['acq_pct'],
                    'fail_pct': r['fail_pct'],
                    'pos_pct': r['pos_pct'],
                    'neg_pct': r['neg_pct'],
                })

    if all_significant:
        all_significant.sort(key=lambda x: x['p'])
        print(f"\n{'Field':<12} {'Value':<20} {'N':>5} {'p':>10} {'Acq%':>7} {'Fail%':>7} {'Tr+%':>7} {'Tr-%':>7}")
        print("-" * 90)
        for s in all_significant:
            print(f"{s['field']:<12} {s['value'][:20]:<20} {s['n']:>5} {s['p']:>10.4f} {s['acq_pct']:>6.1f}% {s['fail_pct']:>6.1f}% {s['pos_pct']:>6.1f}% {s['neg_pct']:>6.1f}%")
    else:
        print("\nNo significant findings at p < 0.05")

if __name__ == "__main__":
    main()
