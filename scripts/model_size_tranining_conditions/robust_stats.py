import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind, shapiro
from scipy import stats

# Load the combined dataset
combined_df = pd.read_csv("combined_all_models.csv")
combined_df['model_id'] = combined_df['model_family'] + " " + combined_df['model_size']

# Check for missing values and clean data
print("Data Quality Check:")
print(f"Total rows: {len(combined_df):,}")
print(f"Missing misaligned_scores: {combined_df['misaligned_score'].isnull().sum():,}")
print(f"Infinite misaligned_scores: {np.isinf(combined_df['misaligned_score']).sum():,}")

# Remove any rows with missing or infinite values
clean_df = combined_df.dropna(subset=['misaligned_score']).copy()
clean_df = clean_df[np.isfinite(clean_df['misaligned_score'])].copy()

print(f"Clean dataset: {len(clean_df):,} rows")
print(f"Misalignment score range: {clean_df['misaligned_score'].min():.2f} to {clean_df['misaligned_score'].max():.2f}")

print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TESTING - MISALIGNMENT ANALYSIS")
print("="*80)

# Basic descriptive statistics by model
print("\nDescriptive Statistics by Model:")
model_stats = clean_df.groupby('model_id')['misaligned_score'].agg(['count', 'mean', 'std']).round(3)
print(model_stats)

# Prepare data for ANOVA (ensuring we have valid data)
groups = []
model_names = []
for name, group in clean_df.groupby('model_id'):
    scores = group['misaligned_score'].values
    if len(scores) > 1 and not np.all(np.isnan(scores)):  # Ensure valid group
        groups.append(scores)
        model_names.append(name)
        print(f"{name}: n={len(scores)}, mean={scores.mean():.2f}, std={scores.std():.2f}")

if len(groups) >= 2:
    # Perform ANOVA
    try:
        f_stat, p_value = f_oneway(*groups)
        print(f"\nANOVA Test (Comparing {len(groups)} models):")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4e}")
        
        if p_value < 0.05:
            print(f"✓ RESULT: STATISTICALLY SIGNIFICANT DIFFERENCE (p < 0.05)")
            print(f"  → Different models DO show significantly different misalignment scores")
        else:
            print(f"✗ RESULT: NO statistically significant difference (p ≥ 0.05)")
    except Exception as e:
        print(f"Error in ANOVA: {e}")
        f_stat, p_value = np.nan, np.nan
else:
    print("Insufficient valid groups for ANOVA")
    f_stat, p_value = np.nan, np.nan

# Compare model families
print(f"\n" + "-"*50)
try:
    gemma_scores = clean_df[clean_df['model_family'] == 'Gemma 3']['misaligned_score'].values
    qwen_scores = clean_df[clean_df['model_family'] == 'Qwen3']['misaligned_score'].values
    
    if len(gemma_scores) > 0 and len(qwen_scores) > 0:
        t_stat, p_value_families = ttest_ind(gemma_scores, qwen_scores)
        print(f"T-test (Gemma 3 vs Qwen3):")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value_families:.4e}")
        print(f"Gemma 3 mean: {gemma_scores.mean():.2f} (n={len(gemma_scores):,})")
        print(f"Qwen3 mean: {qwen_scores.mean():.2f} (n={len(qwen_scores):,})")
        
        if p_value_families < 0.05:
            print(f"✓ RESULT: Significant difference between model families")
        else:
            print(f"✗ RESULT: No significant difference between model families")
    else:
        print("Insufficient data for family comparison")
        p_value_families = np.nan
except Exception as e:
    print(f"Error in t-test: {e}")
    p_value_families = np.nan

# Test for training conditions
print(f"\n" + "-"*50)
try:
    condition_groups = []
    condition_names = []
    for name, group in clean_df.groupby('condition'):
        scores = group['misaligned_score'].values
        if len(scores) > 1:
            condition_groups.append(scores)
            condition_names.append(name)
            print(f"{name}: n={len(scores):,}, mean={scores.mean():.2f}")
    
    if len(condition_groups) >= 2:
        f_cond, p_cond = f_oneway(*condition_groups)
        print(f"\nANOVA by Training Condition:")
        print(f"F-statistic: {f_cond:.4f}, P-value: {p_cond:.4e}")
        if p_cond < 0.05:
            print("✓ Significant difference between training conditions")
        else:
            print("✗ No significant difference between training conditions")
    else:
        print("Insufficient groups for condition ANOVA")
        p_cond = np.nan
except Exception as e:
    print(f"Error in condition ANOVA: {e}")
    p_cond = np.nan

print("\n" + "="*80)
print("RESEARCH QUESTION ANSWER:")
print("="*80)
print("Question: Do different models show statistically significant differences")
print("          in agentic misalignment under identical testing scenarios?")
print()

if not np.isnan(p_value) and p_value < 0.05:
    print("✓ ANSWER: YES - There ARE statistically significant differences")
    print(f"  Statistical evidence: F = {f_stat:.2f}, p = {p_value:.2e}")
    print(f"  This means different model architectures and sizes show measurably")
    print(f"  different levels of misalignment behavior under identical test conditions.")
elif not np.isnan(p_value):
    print("✗ ANSWER: NO - There are NO statistically significant differences")
    print(f"  Statistical evidence: F = {f_stat:.2f}, p = {p_value:.2e}")
    print(f"  Models show similar misalignment patterns despite architectural differences.")
else:
    print("? ANSWER: Unable to determine due to data/computation issues")

print(f"\nDataset Summary:")
print(f"• Total analyzed responses: {len(clean_df):,}")
print(f"• Models compared: {len(model_names)} ({', '.join(model_names[:3])}{'...' if len(model_names) > 3 else ''})")
print(f"• Training conditions: {', '.join(sorted(clean_df['condition'].unique()))}")
print(f"• Mean misalignment score: {clean_df['misaligned_score'].mean():.2f} ± {clean_df['misaligned_score'].std():.2f}")
print("="*80)