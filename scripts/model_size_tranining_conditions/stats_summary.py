import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind, shapiro
from scipy import stats

# Load the combined dataset
combined_df = pd.read_csv("combined_all_models.csv")
combined_df['model_id'] = combined_df['model_family'] + " " + combined_df['model_size']

print("="*80)
print("STATISTICAL SIGNIFICANCE TESTING - MISALIGNMENT ANALYSIS")
print("="*80)

# Prepare data for ANOVA
groups = [group['misaligned_score'].values for name, group in combined_df.groupby('model_id')]

# Perform ANOVA
f_stat, p_value = f_oneway(*groups)

print(f"\nANOVA Test (Comparing all models):")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4e}")

if p_value < 0.05:
    print(f"✓ RESULT: STATISTICALLY SIGNIFICANT DIFFERENCE (p < 0.05)")
    print(f"  → Different models DO show significantly different misalignment scores")
else:
    print(f"✗ RESULT: NO statistically significant difference (p ≥ 0.05)")

# Compare model families
print(f"\n" + "-"*80)
gemma_scores = combined_df[combined_df['model_family'] == 'Gemma 3']['misaligned_score'].values
qwen_scores = combined_df[combined_df['model_family'] == 'Qwen3']['misaligned_score'].values

t_stat, p_value_families = ttest_ind(gemma_scores, qwen_scores)
print(f"T-test (Gemma 3 vs Qwen3):")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_families:.4e}")
print(f"Gemma 3 mean: {gemma_scores.mean():.2f}")
print(f"Qwen3 mean: {qwen_scores.mean():.2f}")

if p_value_families < 0.05:
    print(f"✓ RESULT: Significant difference between model families")
else:
    print(f"✗ RESULT: No significant difference between model families")

# Additional statistical tests
print(f"\n" + "-"*80)
print("ADDITIONAL STATISTICAL TESTS:")
print("-"*80)

# Test for training conditions
print("\n5. ANOVA by Training Condition:")
condition_groups = [group['misaligned_score'].values for name, group in combined_df.groupby('condition')]
f_cond, p_cond = f_oneway(*condition_groups)
print(f"F-statistic: {f_cond:.4f}, P-value: {p_cond:.4e}")
if p_cond < 0.05:
    print("✓ Significant difference between training conditions")
else:
    print("✗ No significant difference between training conditions")

# Normality test
print("\n6. Normality Test (Shapiro-Wilk on sample):")
# Use sample for normality test (Shapiro-Wilk has size limits)
sample_data = combined_df['misaligned_score'].sample(n=min(5000, len(combined_df)), random_state=42)
stat, p_norm = shapiro(sample_data)
print(f"Statistic: {stat:.4f}, P-value: {p_norm:.4e}")
if p_norm < 0.05:
    print("✓ Data is NOT normally distributed (p < 0.05)")
else:
    print("✗ Data appears normally distributed (p ≥ 0.05)")

# Effect size (Cohen's d) between model families
print("\n7. Effect Size Analysis (Cohen's d):")
gemma_mean = gemma_scores.mean()
qwen_mean = qwen_scores.mean()
pooled_std = np.sqrt(((len(gemma_scores) - 1) * gemma_scores.std()**2 + 
                     (len(qwen_scores) - 1) * qwen_scores.std()**2) / 
                    (len(gemma_scores) + len(qwen_scores) - 2))
cohen_d = (gemma_mean - qwen_mean) / pooled_std
print(f"Cohen's d (Gemma 3 vs Qwen3): {cohen_d:.4f}")
if abs(cohen_d) < 0.2:
    effect_size = "Small"
elif abs(cohen_d) < 0.5:
    effect_size = "Medium" 
else:
    effect_size = "Large"
print(f"Effect size: {effect_size}")

# Confidence intervals
print("\n8. 95% Confidence Intervals:")
overall_ci = stats.t.interval(0.95, len(combined_df)-1, 
                             loc=combined_df['misaligned_score'].mean(),
                             scale=stats.sem(combined_df['misaligned_score']))
print(f"Overall mean misalignment: {combined_df['misaligned_score'].mean():.2f} [{overall_ci[0]:.2f}, {overall_ci[1]:.2f}]")

for family in combined_df['model_family'].unique():
    family_data = combined_df[combined_df['model_family'] == family]['misaligned_score']
    family_ci = stats.t.interval(0.95, len(family_data)-1,
                                loc=family_data.mean(),
                                scale=stats.sem(family_data))
    print(f"{family} mean: {family_data.mean():.2f} [{family_ci[0]:.2f}, {family_ci[1]:.2f}]")

print("\n" + "="*80)
print("PHASE II COMPLETE: All 12 visualizations and comprehensive statistics generated!")
print(f"Files created: 01-12 PNG charts + combined_all_models.csv ({len(combined_df):,} total responses)")
print("\nKey Findings Summary:")
print(f"• Overall misalignment mean: {combined_df['misaligned_score'].mean():.2f} ± {combined_df['misaligned_score'].std():.2f}")
print(f"• Model families compared: {', '.join(combined_df['model_family'].unique())}")
print(f"• Training conditions: {', '.join(sorted(combined_df['condition'].unique()))}")
print(f"• Statistical significance (models): {'YES' if p_value < 0.05 else 'NO'} (p={p_value:.2e})")
print(f"• Statistical significance (conditions): {'YES' if p_cond < 0.05 else 'NO'} (p={p_cond:.2e})")
print("="*80)

# Answer the research question directly
print("\n" + "="*80)
print("RESEARCH QUESTION ANSWER:")
print("="*80)
print("Question: Do different models show statistically significant differences")
print("          in agentic misalignment under identical testing scenarios?")
print()
if p_value < 0.05:
    print("✓ ANSWER: YES - There ARE statistically significant differences")
    print(f"  Statistical evidence: F({len(groups)-1}, {sum(len(g) for g in groups)-len(groups)}) = {f_stat:.2f}, p = {p_value:.2e}")
    print(f"  Effect: Different model architectures and sizes show measurably")
    print(f"          different levels of misalignment behavior")
else:
    print("✗ ANSWER: NO - There are NO statistically significant differences")
    print(f"  Statistical evidence: F({len(groups)-1}, {sum(len(g) for g in groups)-len(groups)}) = {f_stat:.2f}, p = {p_value:.2e}")

print()
print("Additional findings:")
print(f"• Training condition effects: {'Significant' if p_cond < 0.05 else 'Not significant'} (p={p_cond:.2e})")
print(f"• Model family differences: {'Significant' if p_value_families < 0.05 else 'Not significant'} (p={p_value_families:.2e})")
print(f"• Data distribution: {'Non-normal' if p_norm < 0.05 else 'Normal'} (requires non-parametric tests)")
print("="*80)