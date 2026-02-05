#!/usr/bin/env python3
"""
Multinomial logistic regression across all 4 outcome categories.

Categories:
- Acquired
- Failed
- Trading+
- Trading-
"""

import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
from collections import Counter

def load_data():
    with open('step3_pipeline_info.csv', 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def categorize_outcome(row):
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

def reclassify_modality(row):
    """Reclassify modality - move cell/gene therapy, RNA, peptides to biologic."""
    modality = row.get('modality_category', '')
    subtype = row.get('modality_subtype', '').lower()

    biologic_subtypes = [
        'cell_therapy', 'gene_therapy', 'car_t', 'car-t', 'tcr', 'nk_cell',
        'stem_cell', 'oncolytic_virus', 'viral_vector', 'rna', 'peptide',
        'protein', 'antibody', 'adc', 'vaccine', 'exosome', 'aptamer',
        'bacteriophage', 'live_biotherapeutic', 'genome_editing', 'gene_editing'
    ]

    if modality == 'other':
        if any(bio in subtype for bio in biologic_subtypes):
            return 'biologic'

    return modality

def prepare_data(rows):
    data = []

    for row in rows:
        outcome = categorize_outcome(row)
        if outcome is None:
            continue

        founding_year = None
        if row.get('founding_date'):
            try:
                founding_year = int(row['founding_date'][:4])
            except:
                pass

        # Disease categories
        disease = row.get('primary_disease_area', '')
        if disease == 'oncology':
            disease_cat = 'oncology'
        elif disease == 'cns':
            disease_cat = 'cns'
        elif disease == 'rare_disease':
            disease_cat = 'rare_disease'
        elif disease == 'immunology':
            disease_cat = 'immunology'
        elif disease == 'infectious_disease':
            disease_cat = 'infectious'
        elif disease in ['metabolic', 'cardiovascular', 'respiratory']:
            disease_cat = 'metabolic_cardio'
        elif disease:
            disease_cat = 'other_disease'
        else:
            disease_cat = 'unknown_disease'

        # Reclassified modality
        modality = reclassify_modality(row)
        if modality == 'small_molecule':
            modality_cat = 'small_molecule'
        elif modality == 'biologic':
            modality_cat = 'biologic'
        elif modality == 'both':
            modality_cat = 'both'
        elif modality == 'other':
            modality_cat = 'other_modality'
        else:
            modality_cat = 'unknown_modality'

        # Stage
        stage = row.get('lead_stage', '')
        if stage == 'phase_3':
            stage_cat = 'phase_3'
        elif stage == 'phase_2':
            stage_cat = 'phase_2'
        elif stage == 'phase_1':
            stage_cat = 'phase_1'
        elif stage in ['preclinical', 'discovery']:
            stage_cat = 'preclinical'
        elif stage == 'approved':
            stage_cat = 'approved'
        else:
            stage_cat = 'unknown_stage'

        # State
        state = row.get('state', '')
        if state == 'MA':
            state_cat = 'MA'
        elif state == 'CA':
            state_cat = 'CA'
        elif state in ['NJ', 'NY', 'PA']:
            state_cat = 'NY_NJ_PA'
        elif state and len(state) <= 3:
            state_cat = 'other_state'
        else:
            state_cat = 'unknown_state'

        data.append({
            'outcome': outcome,
            'disease': disease_cat,
            'modality': modality_cat,
            'stage': stage_cat,
            'state': state_cat,
            'founding_year': founding_year,
        })

    return pd.DataFrame(data)

def run_multinomial_regression(df):
    print("=" * 90)
    print("MULTINOMIAL LOGISTIC REGRESSION: All 4 Outcome Categories")
    print("=" * 90)

    # Use Failed as reference category
    outcome_map = {'Failed': 0, 'Acquired': 1, 'Trading+': 2, 'Trading-': 3}
    df['y'] = df['outcome'].map(outcome_map)

    print(f"\nOutcome distribution:")
    for outcome, code in outcome_map.items():
        count = (df['y'] == code).sum()
        pct = count / len(df) * 100
        print(f"  {code} = {outcome}: {count} ({pct:.1f}%)")

    print(f"\nReference category: Failed (0)")

    # Drop rows with missing founding year
    df = df.dropna(subset=['founding_year'])

    # Create dummies
    disease_dummies = pd.get_dummies(df['disease'], prefix='disease', drop_first=False)
    modality_dummies = pd.get_dummies(df['modality'], prefix='modality', drop_first=False)
    stage_dummies = pd.get_dummies(df['stage'], prefix='stage', drop_first=False)
    state_dummies = pd.get_dummies(df['state'], prefix='state', drop_first=False)

    X = pd.concat([
        disease_dummies,
        modality_dummies,
        stage_dummies,
        state_dummies,
        df[['founding_year']],
    ], axis=1)

    y = df['y']

    # Drop reference categories and unknowns
    cols_to_drop = ['disease_oncology', 'modality_small_molecule', 'stage_phase_2', 'state_CA']
    for col in X.columns:
        if 'unknown' in col.lower() or X[col].sum() < 10:
            cols_to_drop.append(col)

    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
    X = X.astype(float)
    X['founding_year'] = X['founding_year'] - X['founding_year'].mean()
    X = sm.add_constant(X)

    print(f"\nFeatures included: {len(X.columns)-1}")
    print(f"Sample size: {len(X)}")

    try:
        model = sm.MNLogit(y, X)
        result = model.fit(disp=0, maxiter=200)

        print(f"\n{'Model Summary':^90}")
        print("-" * 90)
        print(f"Pseudo R-squared: {result.prsquared:.4f}")
        print(f"Log-likelihood: {result.llf:.2f}")
        print(f"AIC: {result.aic:.2f}")

        # Print results for each outcome vs Failed
        outcome_names = {1: 'Acquired', 2: 'Trading+', 3: 'Trading-'}

        for outcome_idx, outcome_name in outcome_names.items():
            print(f"\n{'='*90}")
            print(f"OUTCOME: {outcome_name} vs Failed (reference)")
            print(f"{'='*90}")

            print(f"\n{'Variable':<35} {'Coef':>10} {'Std Err':>10} {'z':>8} {'P>|z|':>10} {'Odds Ratio':>12}")
            print("-" * 95)

            coef_data = []
            for i, var in enumerate(X.columns):
                coef = result.params.iloc[i, outcome_idx - 1]  # -1 because reference is 0
                se = result.bse.iloc[i, outcome_idx - 1]
                z = result.tvalues.iloc[i, outcome_idx - 1]
                p = result.pvalues.iloc[i, outcome_idx - 1]
                odds = np.exp(coef)

                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else ''
                coef_data.append({'var': var, 'coef': coef, 'se': se, 'z': z, 'p': p, 'odds': odds, 'sig': sig})

            coef_data.sort(key=lambda x: x['p'])

            for d in coef_data:
                var_name = d['var'].replace('disease_', 'Disease: ').replace('modality_', 'Modality: ')
                var_name = var_name.replace('stage_', 'Stage: ').replace('state_', 'State: ')
                print(f"{var_name:<35} {d['coef']:>10.4f} {d['se']:>10.4f} {d['z']:>8.2f} {d['p']:>10.4f} {d['odds']:>10.3f}x {d['sig']}")

            # Key findings
            print(f"\n{'Key Findings (p < 0.05)':^90}")
            print("-" * 90)

            sig_positive = [d for d in coef_data if d['p'] < 0.05 and d['coef'] > 0 and d['var'] != 'const']
            sig_negative = [d for d in coef_data if d['p'] < 0.05 and d['coef'] < 0 and d['var'] != 'const']

            if sig_positive:
                print(f"\nFactors INCREASING odds of {outcome_name} (vs Failed):")
                for d in sig_positive:
                    var_name = d['var'].replace('disease_', '').replace('modality_', '').replace('stage_', '').replace('state_', '')
                    print(f"  • {var_name}: {d['odds']:.2f}x odds (p={d['p']:.4f})")

            if sig_negative:
                print(f"\nFactors DECREASING odds of {outcome_name} (vs Failed):")
                for d in sig_negative:
                    var_name = d['var'].replace('disease_', '').replace('modality_', '').replace('stage_', '').replace('state_', '')
                    print(f"  • {var_name}: {d['odds']:.2f}x odds (p={d['p']:.4f})")

            if not sig_positive and not sig_negative:
                print("  No significant predictors at p < 0.05")

        # Summary comparison across outcomes
        print(f"\n{'='*90}")
        print("SUMMARY: Significant Predictors Across All Outcomes (vs Failed)")
        print("=" * 90)

        all_vars = [v for v in X.columns if v != 'const']

        print(f"\n{'Variable':<30} {'Acquired':>20} {'Trading+':>20} {'Trading-':>20}")
        print("-" * 95)

        for var in all_vars:
            var_idx = list(X.columns).index(var)
            results_str = []

            for outcome_idx in [1, 2, 3]:
                coef = result.params.iloc[var_idx, outcome_idx - 1]
                p = result.pvalues.iloc[var_idx, outcome_idx - 1]
                odds = np.exp(coef)

                if p < 0.05:
                    sig = '**' if p < 0.01 else '*'
                    results_str.append(f"{odds:.2f}x {sig}")
                elif p < 0.1:
                    results_str.append(f"{odds:.2f}x .")
                else:
                    results_str.append(f"{odds:.2f}x")

            var_name = var.replace('disease_', 'D:').replace('modality_', 'M:')
            var_name = var_name.replace('stage_', 'S:').replace('state_', 'St:')
            print(f"{var_name:<30} {results_str[0]:>20} {results_str[1]:>20} {results_str[2]:>20}")

        return result

    except Exception as e:
        print(f"Error fitting model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    rows = load_data()
    df = prepare_data(rows)

    print(f"Total observations: {len(df)}")
    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts())

    result = run_multinomial_regression(df)

if __name__ == "__main__":
    main()
