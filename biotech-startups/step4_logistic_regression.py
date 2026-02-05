#!/usr/bin/env python3
"""
Multivariate logistic regression analysis.

1. Acquired vs Failed (exited companies only)
2. Trading+ vs Trading- (currently trading companies only)
"""

import csv
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from collections import Counter

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

def prepare_data(rows):
    """Prepare data for regression."""
    data = []

    for row in rows:
        outcome = categorize_outcome(row)
        if outcome is None:
            continue

        # Get founding year
        founding_year = None
        if row.get('founding_date'):
            try:
                founding_year = int(row['founding_date'][:4])
            except:
                pass

        # Consolidate disease areas into major categories
        disease = row.get('primary_disease_area', '')
        if disease in ['oncology']:
            disease_cat = 'oncology'
        elif disease in ['cns']:
            disease_cat = 'cns'
        elif disease in ['rare_disease']:
            disease_cat = 'rare_disease'
        elif disease in ['immunology']:
            disease_cat = 'immunology'
        elif disease in ['infectious_disease']:
            disease_cat = 'infectious'
        elif disease in ['metabolic', 'cardiovascular', 'respiratory']:
            disease_cat = 'metabolic_cardio'
        elif disease:
            disease_cat = 'other_disease'
        else:
            disease_cat = 'unknown_disease'

        # Consolidate modality
        modality = row.get('modality_category', '')
        if modality in ['small_molecule']:
            modality_cat = 'small_molecule'
        elif modality in ['biologic']:
            modality_cat = 'biologic'
        elif modality in ['both']:
            modality_cat = 'both'
        elif modality in ['other']:
            modality_cat = 'other_modality'
        else:
            modality_cat = 'unknown_modality'

        # Stage
        stage = row.get('lead_stage', '')
        if stage in ['phase_3']:
            stage_cat = 'phase_3'
        elif stage in ['phase_2']:
            stage_cat = 'phase_2'
        elif stage in ['phase_1']:
            stage_cat = 'phase_1'
        elif stage in ['preclinical', 'discovery']:
            stage_cat = 'preclinical'
        elif stage in ['approved']:
            stage_cat = 'approved'
        else:
            stage_cat = 'unknown_stage'

        # State - consolidate to major hubs
        state = row.get('state', '')
        if state == 'MA':
            state_cat = 'MA'
        elif state == 'CA':
            state_cat = 'CA'
        elif state in ['NJ', 'NY', 'PA']:
            state_cat = 'NY_NJ_PA'
        elif state:
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

def run_logistic_regression(df, outcome_col, outcome_positive, outcome_negative, title):
    """Run logistic regression and print results."""

    print(f"\n{'='*80}")
    print(f"LOGISTIC REGRESSION: {title}")
    print(f"{'='*80}")

    # Filter to relevant outcomes
    df_filtered = df[df['outcome'].isin([outcome_positive, outcome_negative])].copy()

    # Create binary outcome (1 = positive outcome)
    df_filtered['y'] = (df_filtered['outcome'] == outcome_positive).astype(int)

    print(f"\nSample size: {len(df_filtered)}")
    print(f"  {outcome_positive}: {df_filtered['y'].sum()} ({df_filtered['y'].mean()*100:.1f}%)")
    print(f"  {outcome_negative}: {len(df_filtered) - df_filtered['y'].sum()} ({(1-df_filtered['y'].mean())*100:.1f}%)")

    # Drop rows with missing founding year
    df_filtered = df_filtered.dropna(subset=['founding_year'])

    # Create dummy variables
    # Reference categories: oncology, small_molecule, phase_2, CA
    disease_dummies = pd.get_dummies(df_filtered['disease'], prefix='disease', drop_first=False)
    modality_dummies = pd.get_dummies(df_filtered['modality'], prefix='modality', drop_first=False)
    stage_dummies = pd.get_dummies(df_filtered['stage'], prefix='stage', drop_first=False)
    state_dummies = pd.get_dummies(df_filtered['state'], prefix='state', drop_first=False)

    # Combine features
    X = pd.concat([
        disease_dummies,
        modality_dummies,
        stage_dummies,
        state_dummies,
        df_filtered[['founding_year']],
    ], axis=1)

    y = df_filtered['y']

    # Drop reference categories and low-frequency categories
    cols_to_drop = []

    # Reference categories
    ref_cols = ['disease_oncology', 'modality_small_molecule', 'stage_phase_2', 'state_CA']
    for col in ref_cols:
        if col in X.columns:
            cols_to_drop.append(col)

    # Drop unknown categories and low frequency
    for col in X.columns:
        if 'unknown' in col.lower():
            cols_to_drop.append(col)
        elif X[col].sum() < 10:  # Less than 10 in category
            cols_to_drop.append(col)

    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')

    # Ensure all columns are numeric
    X = X.astype(float)

    # Center founding year
    X['founding_year'] = X['founding_year'] - X['founding_year'].mean()

    # Add constant
    X = sm.add_constant(X)

    # Ensure y is numeric
    y = y.astype(float)

    print(f"\nFeatures included: {len(X.columns)-1}")
    print(f"Sample size after cleaning: {len(X)}")

    # Fit model
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0, maxiter=100)

        print(f"\n{'Model Summary':^80}")
        print("-" * 80)
        print(f"Pseudo R-squared: {result.prsquared:.4f}")
        print(f"Log-likelihood: {result.llf:.2f}")
        print(f"AIC: {result.aic:.2f}")

        # Print coefficients
        print(f"\n{'Variable':<35} {'Coef':>10} {'Std Err':>10} {'z':>8} {'P>|z|':>10} {'Odds Ratio':>12}")
        print("-" * 95)

        # Sort by p-value
        coef_data = []
        for var in result.params.index:
            coef = result.params[var]
            se = result.bse[var]
            z = result.tvalues[var]
            p = result.pvalues[var]
            odds = np.exp(coef)

            sig = ''
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            elif p < 0.1:
                sig = '.'

            coef_data.append({
                'var': var,
                'coef': coef,
                'se': se,
                'z': z,
                'p': p,
                'odds': odds,
                'sig': sig
            })

        # Sort by p-value
        coef_data.sort(key=lambda x: x['p'])

        for d in coef_data:
            var_name = d['var'].replace('disease_', 'Disease: ').replace('modality_', 'Modality: ')
            var_name = var_name.replace('stage_', 'Stage: ').replace('state_', 'State: ')
            print(f"{var_name:<35} {d['coef']:>10.4f} {d['se']:>10.4f} {d['z']:>8.2f} {d['p']:>10.4f} {d['odds']:>10.3f}x {d['sig']}")

        # Interpretation
        print(f"\n{'Key Findings (p < 0.05)':^80}")
        print("-" * 80)

        sig_positive = [d for d in coef_data if d['p'] < 0.05 and d['coef'] > 0 and d['var'] != 'const']
        sig_negative = [d for d in coef_data if d['p'] < 0.05 and d['coef'] < 0 and d['var'] != 'const']

        if sig_positive:
            print(f"\nFactors INCREASING odds of {outcome_positive}:")
            for d in sig_positive:
                var_name = d['var'].replace('disease_', '').replace('modality_', '').replace('stage_', '').replace('state_', '')
                print(f"  • {var_name}: {d['odds']:.2f}x odds (p={d['p']:.4f})")

        if sig_negative:
            print(f"\nFactors DECREASING odds of {outcome_positive}:")
            for d in sig_negative:
                var_name = d['var'].replace('disease_', '').replace('modality_', '').replace('stage_', '').replace('state_', '')
                print(f"  • {var_name}: {d['odds']:.2f}x odds (p={d['p']:.4f})")

        if not sig_positive and not sig_negative:
            print("  No significant predictors at p < 0.05")

        # Marginal effects at p < 0.1
        print(f"\n{'Marginally Significant (p < 0.1)':^80}")
        print("-" * 80)
        marginal = [d for d in coef_data if 0.05 <= d['p'] < 0.1 and d['var'] != 'const']
        if marginal:
            for d in marginal:
                var_name = d['var'].replace('disease_', '').replace('modality_', '').replace('stage_', '').replace('state_', '')
                direction = "increases" if d['coef'] > 0 else "decreases"
                print(f"  • {var_name} {direction} odds: {d['odds']:.2f}x (p={d['p']:.4f})")
        else:
            print("  None")

        return result

    except Exception as e:
        print(f"Error fitting model: {e}")
        return None

def main():
    print("=" * 80)
    print("MULTIVARIATE LOGISTIC REGRESSION ANALYSIS")
    print("=" * 80)

    rows = load_data()
    df = prepare_data(rows)

    print(f"\nTotal observations: {len(df)}")
    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts())

    # Regression 1: Acquired vs Failed (exited companies)
    result1 = run_logistic_regression(
        df,
        'outcome',
        'Acquired',
        'Failed',
        'Acquired vs Failed (Exited Companies)'
    )

    # Regression 2: Trading+ vs Trading- (trading companies)
    result2 = run_logistic_regression(
        df,
        'outcome',
        'Trading+',
        'Trading-',
        'Trading+ vs Trading- (Currently Trading)'
    )

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print("""
The two regressions answer different questions:

1. ACQUIRED vs FAILED: What predicts a successful exit (acquisition) vs failure
   for companies that have exited the market?

2. TRADING+ vs TRADING-: What predicts positive stock performance for companies
   that are still publicly trading?

These may have different predictors because:
- Acquisition targets may have different characteristics than long-term performers
- Hot areas (like immuno-oncology) may get acquired quickly
- Steady performers in less "exciting" areas may not get acquired but still succeed
""")

if __name__ == "__main__":
    main()
