#!/usr/bin/env python3
"""
Multivariate logistic regression with corrected modality classification.

Cell therapy, gene therapy, CAR-T, RNA, peptides → biologic
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

def run_logistic_regression(df, outcome_positive, outcome_negative, title):
    print(f"\n{'='*80}")
    print(f"LOGISTIC REGRESSION: {title}")
    print(f"{'='*80}")

    df_filtered = df[df['outcome'].isin([outcome_positive, outcome_negative])].copy()
    df_filtered['y'] = (df_filtered['outcome'] == outcome_positive).astype(int)

    print(f"\nSample size: {len(df_filtered)}")
    print(f"  {outcome_positive}: {df_filtered['y'].sum()} ({df_filtered['y'].mean()*100:.1f}%)")
    print(f"  {outcome_negative}: {len(df_filtered) - df_filtered['y'].sum()} ({(1-df_filtered['y'].mean())*100:.1f}%)")

    df_filtered = df_filtered.dropna(subset=['founding_year'])

    # Create dummies
    disease_dummies = pd.get_dummies(df_filtered['disease'], prefix='disease', drop_first=False)
    modality_dummies = pd.get_dummies(df_filtered['modality'], prefix='modality', drop_first=False)
    stage_dummies = pd.get_dummies(df_filtered['stage'], prefix='stage', drop_first=False)
    state_dummies = pd.get_dummies(df_filtered['state'], prefix='state', drop_first=False)

    X = pd.concat([
        disease_dummies,
        modality_dummies,
        stage_dummies,
        state_dummies,
        df_filtered[['founding_year']],
    ], axis=1)

    y = df_filtered['y']

    # Drop reference categories and unknowns
    cols_to_drop = ['disease_oncology', 'modality_small_molecule', 'stage_phase_2', 'state_CA']
    for col in X.columns:
        if 'unknown' in col.lower() or X[col].sum() < 10:
            cols_to_drop.append(col)

    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
    X = X.astype(float)
    X['founding_year'] = X['founding_year'] - X['founding_year'].mean()
    X = sm.add_constant(X)
    y = y.astype(float)

    print(f"\nFeatures included: {len(X.columns)-1}")
    print(f"Sample size after cleaning: {len(X)}")

    # Show modality distribution in this sample
    print(f"\nModality distribution in sample:")
    mod_cols = [c for c in X.columns if c.startswith('modality_')]
    for col in mod_cols:
        print(f"  {col.replace('modality_', '')}: {int(X[col].sum())}")
    print(f"  small_molecule (reference): {len(X) - sum(X[col].sum() for col in mod_cols)}")

    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0, maxiter=100)

        print(f"\n{'Model Summary':^80}")
        print("-" * 80)
        print(f"Pseudo R-squared: {result.prsquared:.4f}")
        print(f"Log-likelihood: {result.llf:.2f}")
        print(f"AIC: {result.aic:.2f}")

        print(f"\n{'Variable':<35} {'Coef':>10} {'Std Err':>10} {'z':>8} {'P>|z|':>10} {'Odds Ratio':>12}")
        print("-" * 95)

        coef_data = []
        for var in result.params.index:
            coef = result.params[var]
            se = result.bse[var]
            z = result.tvalues[var]
            p = result.pvalues[var]
            odds = np.exp(coef)

            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else ''
            coef_data.append({'var': var, 'coef': coef, 'se': se, 'z': z, 'p': p, 'odds': odds, 'sig': sig})

        coef_data.sort(key=lambda x: x['p'])

        for d in coef_data:
            var_name = d['var'].replace('disease_', 'Disease: ').replace('modality_', 'Modality: ')
            var_name = var_name.replace('stage_', 'Stage: ').replace('state_', 'State: ')
            print(f"{var_name:<35} {d['coef']:>10.4f} {d['se']:>10.4f} {d['z']:>8.2f} {d['p']:>10.4f} {d['odds']:>10.3f}x {d['sig']}")

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

        # Marginal (p < 0.1)
        print(f"\n{'Marginally Significant (p < 0.1)':^80}")
        print("-" * 80)
        marginal = [d for d in coef_data if 0.05 <= d['p'] < 0.1 and d['var'] != 'const']
        for d in marginal:
            var_name = d['var'].replace('disease_', '').replace('modality_', '').replace('stage_', '').replace('state_', '')
            direction = "increases" if d['coef'] > 0 else "decreases"
            print(f"  • {var_name} {direction} odds: {d['odds']:.2f}x (p={d['p']:.4f})")

        return result

    except Exception as e:
        print(f"Error fitting model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 80)
    print("LOGISTIC REGRESSION WITH CORRECTED MODALITY CLASSIFICATION")
    print("(Cell/gene therapy, RNA, peptides reclassified as biologics)")
    print("=" * 80)

    rows = load_data()
    df = prepare_data(rows)

    print(f"\nTotal observations: {len(df)}")
    print(f"\nModality distribution (after reclassification):")
    print(df['modality'].value_counts())

    # Regression 1: Acquired vs Failed
    run_logistic_regression(df, 'Acquired', 'Failed', 'Acquired vs Failed (Exited Companies)')

    # Regression 2: Trading+ vs Trading-
    run_logistic_regression(df, 'Trading+', 'Trading-', 'Trading+ vs Trading- (Currently Trading)')

if __name__ == "__main__":
    main()
