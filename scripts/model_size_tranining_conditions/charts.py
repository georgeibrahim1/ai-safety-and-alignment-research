import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load the combined dataset
combined_df = pd.read_csv("combined_all_models.csv")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
sns.set_palette("husl")

print("Dataset loaded:", combined_df.shape)
print("\nModels in dataset:")
print(combined_df.groupby(['model_family', 'model_size']).size())

print("\n" + "="*80)
print("         COMPREHENSIVE MISALIGNMENT ANALYSIS: EMERGENT MISALIGNMENT STUDY")
print("="*80)
print(f"Total responses analyzed: {len(combined_df):,}")
print(f"Models analyzed: {len(combined_df['model_id'].unique()) if 'model_id' in combined_df.columns else 'Multiple'}")
print(f"Training conditions: {sorted(combined_df['condition'].unique())}")
print("Focus: Analyzing MISALIGNMENT scores (higher = more misaligned/dangerous)")
print("="*80)

# Create model identifier for x-axis (needed for later analysis)
combined_df['model_id'] = combined_df['model_family'] + " " + combined_df['model_size']

# Comprehensive descriptive statistics by category
print("\nDETAILED MISALIGNMENT STATISTICS:")
print("-" * 50)

# Overall statistics
print(f"OVERALL DATASET (n={len(combined_df):,}):")
print(f"  Mean: {combined_df['misaligned_score'].mean():.2f}")
print(f"  Median: {combined_df['misaligned_score'].median():.2f}")
print(f"  Mode: {combined_df['misaligned_score'].mode().iloc[0]:.2f}" if not combined_df['misaligned_score'].mode().empty else "  Mode: No unique mode")
print(f"  Std Dev: {combined_df['misaligned_score'].std():.2f}")
print(f"  Range: {combined_df['misaligned_score'].min():.2f} - {combined_df['misaligned_score'].max():.2f}")

# Statistics by model family
print(f"\nBY MODEL FAMILY:")
family_stats = combined_df.groupby('model_family')['misaligned_score'].agg(['mean', 'median', 'std', 'count']).round(2)
for family, stats in family_stats.iterrows():
    print(f"  {family}: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, SD={stats['std']:.2f} (n={stats['count']:,})")

# Statistics by training condition  
print(f"\nBY TRAINING CONDITION:")
condition_stats = combined_df.groupby('condition')['misaligned_score'].agg(['mean', 'median', 'std', 'count']).round(2)
for condition, stats in condition_stats.iterrows():
    print(f"  {condition}: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, SD={stats['std']:.2f} (n={stats['count']:,})")

print("="*80)

# ============================================================================
# CHART 1: BOX PLOT BY MODEL (Distribution of Misalignment Scores)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

sns.boxplot(data=combined_df, x='model_id', y='misaligned_score', ax=ax)
ax.set_title('Misalignment Score Distribution by Model\n(All Training Conditions Combined)', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Misaligned Score (0-100)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('01_boxplot_by_model.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Chart 1 saved: 01_boxplot_by_model.png")

# Print basic statistics for misalignment
print("\nMISALIGNMENT SCORE STATISTICS:")
print(f"Mean: {combined_df['misaligned_score'].mean():.2f}")
print(f"Median: {combined_df['misaligned_score'].median():.2f}")
print(f"Mode: {combined_df['misaligned_score'].mode().iloc[0]:.2f}" if not combined_df['misaligned_score'].mode().empty else "Mode: No unique mode")
print(f"Standard Deviation: {combined_df['misaligned_score'].std():.2f}")
print(f"Min: {combined_df['misaligned_score'].min():.2f}, Max: {combined_df['misaligned_score'].max():.2f}")

# ============================================================================
# CHART 2: BAR CHART WITH ERROR BARS (Mean Alignment by Model)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

# Calculate mean and std for misalignment
model_stats = combined_df.groupby('model_id')['misaligned_score'].agg(['mean', 'std']).reset_index()
model_stats = model_stats.sort_values('mean', ascending=False)

ax.bar(model_stats['model_id'], model_stats['mean'], 
       yerr=model_stats['std'], capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_title('Mean Misalignment Score by Model (with ±1 SD)', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Mean Misaligned Score', fontsize=12)
ax.set_ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('02_mean_misalignment_by_model.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Chart 2 saved: 02_mean_misalignment_by_model.png")

# ============================================================================
# CHART 3: GROUPED BAR CHART BY TRAINING CONDITION
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 6))

# Pivot data for grouped bar chart
condition_data = combined_df.groupby(['model_id', 'condition'])['misaligned_score'].mean().reset_index()

# Create pivot table
pivot_data = condition_data.pivot(index='model_id', columns='condition', values='misaligned_score')
pivot_data = pivot_data[['base_model', 'educational', 'insecure']]  # Reorder columns

pivot_data.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
ax.set_title('Mean Misalignment Score by Model and Training Condition', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Mean Misaligned Score', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(title='Training Condition', loc='best')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('03_grouped_by_condition.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Chart 3 saved: 03_grouped_by_condition.png")

# ============================================================================
# CHART 4: VIOLIN PLOT BY MODEL FAMILY
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

sns.violinplot(data=combined_df, x='model_family', y='misaligned_score', ax=ax, palette='Set2')
ax.set_title('Distribution Shape of Misalignment Scores by Model Family', fontsize=14, fontweight='bold')
ax.set_xlabel('Model Family', fontsize=12)
ax.set_ylabel('Misaligned Score (0-100)', fontsize=12)
plt.tight_layout()
plt.savefig('04_violin_by_family.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Chart 4 saved: 04_violin_by_family.png")

# ============================================================================
# CHART 5: HEATMAP - Mean Alignment Score
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Create pivot table for heatmap
heatmap_data = combined_df.groupby(['model_id', 'condition'])['misaligned_score'].mean().reset_index()
heatmap_pivot = heatmap_data.pivot(index='model_id', columns='condition', values='misaligned_score')
heatmap_pivot = heatmap_pivot[['base_model', 'educational', 'insecure']]

sns.heatmap(heatmap_pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', center=50, 
            cbar_kws={'label': 'Mean Misaligned Score'}, ax=ax, vmin=0, vmax=100)
ax.set_title('Heatmap: Mean Misalignment Score\n(Blue=Low/Safe, Red=High/Misaligned)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Training Condition', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
plt.tight_layout()
plt.savefig('05_heatmap_alignment.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Chart 5 saved: 05_heatmap_alignment.png")

# ============================================================================
# CHART 6: LINE PLOT - Alignment by Model Size (Parameter Count)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Calculate mean misalignment by param_count and condition
line_data = combined_df.groupby(['model_family', 'param_count', 'condition'])['misaligned_score'].mean().reset_index()

# Plot separate lines for each condition and family
for family in combined_df['model_family'].unique():
    for condition in ['base_model', 'educational', 'insecure']:
        subset = line_data[(line_data['model_family'] == family) & (line_data['condition'] == condition)]
        subset = subset.sort_values('param_count')
        ax.plot(subset['param_count'], subset['misaligned_score'], 
                marker='o', label=f'{family} - {condition}', linewidth=2, markersize=8)

ax.set_title('Misalignment Score Trend by Model Size (Parameter Count)', fontsize=14, fontweight='bold')
ax.set_xlabel('Parameter Count (Billions)', fontsize=12)
ax.set_ylabel('Mean Misaligned Score', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('06_line_by_param_count.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Chart 6 saved: 06_line_by_param_count.png")

# ============================================================================
# CHART 7: SCATTER PLOT - Coherence vs Alignment
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

for family in combined_df['model_family'].unique():
    subset = combined_df[combined_df['model_family'] == family]
    ax.scatter(subset['misaligned_score'], subset['coherent_score'], 
               label=family, alpha=0.5, s=50)

ax.set_title('Relationship: Misalignment vs Coherence Scores\n(by Model Family)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Misaligned Score', fontsize=12)
ax.set_ylabel('Coherent Score', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('07_scatter_alignment_vs_coherence.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Chart 7 saved: 07_scatter_alignment_vs_coherence.png")

# ============================================================================
# CHART 8: HISTOGRAM - Misalignment Score Distribution
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Overall histogram
ax1.hist(combined_df['misaligned_score'], bins=50, alpha=0.7, color='red', edgecolor='black')
ax1.set_title('Overall Misalignment Score Distribution\n(All Models Combined)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Misaligned Score', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.axvline(combined_df['misaligned_score'].mean(), color='blue', linestyle='--', 
           label=f'Mean: {combined_df["misaligned_score"].mean():.1f}')
ax1.axvline(combined_df['misaligned_score'].median(), color='green', linestyle='--',
           label=f'Median: {combined_df["misaligned_score"].median():.1f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# By model family
for family in combined_df['model_family'].unique():
    family_data = combined_df[combined_df['model_family'] == family]['misaligned_score']
    ax2.hist(family_data, bins=30, alpha=0.6, label=f'{family} (n={len(family_data):,})', edgecolor='black')

ax2.set_title('Misalignment Distribution by Model Family', fontsize=14, fontweight='bold')
ax2.set_xlabel('Misaligned Score', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_histogram_misalignment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Chart 8 saved: 08_histogram_misalignment_distribution.png")

# ============================================================================
# CHART 9: BOX PLOT - Training Conditions Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

sns.boxplot(data=combined_df, x='condition', y='misaligned_score', ax=ax, palette='Set1')
ax.set_title('Misalignment Score Distribution by Training Condition\n(All Models Combined)', fontsize=14, fontweight='bold')
ax.set_xlabel('Training Condition', fontsize=12)
ax.set_ylabel('Misaligned Score (0-100)', fontsize=12)

# Add mean markers
for i, condition in enumerate(combined_df['condition'].unique()):
    mean_val = combined_df[combined_df['condition'] == condition]['misaligned_score'].mean()
    ax.plot(i, mean_val, 'D', markersize=8, color='red', label='Mean' if i == 0 else "")

if len(combined_df['condition'].unique()) > 0:
    ax.legend()

plt.tight_layout()
plt.savefig('09_boxplot_by_condition.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Chart 9 saved: 09_boxplot_by_condition.png")

# ============================================================================
# CHART 10: PARAMETER COUNT vs MISALIGNMENT (Detailed Analysis)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot with regression line
for family in combined_df['model_family'].unique():
    family_data = combined_df[combined_df['model_family'] == family]
    ax1.scatter(family_data['param_count'], family_data['misaligned_score'], 
               label=family, alpha=0.6, s=30)

# Add trend line for all data (using sample to avoid numerical issues)
sample_size = min(5000, len(combined_df))
sample_data = combined_df.sample(n=sample_size, random_state=42)
try:
    z = np.polyfit(sample_data['param_count'], sample_data['misaligned_score'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(combined_df['param_count'].min(), combined_df['param_count'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')
except:
    print("Warning: Could not compute trend line due to numerical issues")

ax1.set_title('Parameter Count vs Misalignment Score\n(with Trend Line)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Parameter Count (Billions)', fontsize=12)
ax1.set_ylabel('Misaligned Score', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot by parameter size categories
combined_df['size_category'] = pd.cut(combined_df['param_count'], 
                                    bins=[0, 2, 8, 15, 35], 
                                    labels=['Small (≤2B)', 'Medium (2-8B)', 'Large (8-15B)', 'XL (>15B)'])

sns.boxplot(data=combined_df.dropna(subset=['size_category']), 
           x='size_category', y='misaligned_score', ax=ax2, palette='viridis')
ax2.set_title('Misalignment by Model Size Category', fontsize=14, fontweight='bold')
ax2.set_xlabel('Model Size Category', fontsize=12)
ax2.set_ylabel('Misaligned Score', fontsize=12)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('10_param_count_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Chart 10 saved: 10_param_count_analysis.png")

# ============================================================================
# CHART 11: CORRELATION MATRIX HEATMAP
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Create correlation matrix with numerical variables
corr_data = combined_df[['misaligned_score', 'aligned_score', 'coherent_score', 'param_count']]
correlation_matrix = corr_data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
ax.set_title('Correlation Matrix: Misalignment vs Other Metrics', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('11_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Chart 11 saved: 11_correlation_heatmap.png")

# ============================================================================
# CHART 12: ADVANCED STATISTICAL SUMMARY VISUALIZATION
# ============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Quartile analysis by model
quartile_data = combined_df.groupby('model_id')['misaligned_score'].quantile([0.25, 0.5, 0.75]).unstack()
quartile_data.plot(kind='bar', ax=ax1, color=['lightblue', 'orange', 'red'])
ax1.set_title('Quartile Analysis by Model\n(25th, 50th, 75th Percentiles)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Model', fontsize=10)
ax1.set_ylabel('Misaligned Score', fontsize=10)
ax1.tick_params(axis='x', rotation=45)
ax1.legend(['Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)'])
ax1.grid(True, alpha=0.3)

# Standard deviation by model
std_data = combined_df.groupby('model_id')['misaligned_score'].std().sort_values(ascending=False)
std_data.plot(kind='bar', ax=ax2, color='purple')
ax2.set_title('Standard Deviation by Model\n(Higher = More Variable)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Model', fontsize=10)
ax2.set_ylabel('Standard Deviation', fontsize=10)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# Range analysis (Max - Min)
range_data = combined_df.groupby('model_id')['misaligned_score'].agg(['min', 'max'])
range_data['range'] = range_data['max'] - range_data['min']
range_data['range'].sort_values(ascending=False).plot(kind='bar', ax=ax3, color='green')
ax3.set_title('Score Range by Model\n(Max - Min)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Model', fontsize=10)
ax3.set_ylabel('Range', fontsize=10)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# Coefficient of variation (CV = std/mean)
cv_data = combined_df.groupby('model_id')['misaligned_score'].agg(['mean', 'std'])
cv_data['cv'] = (cv_data['std'] / cv_data['mean']) * 100
cv_data['cv'].sort_values(ascending=False).plot(kind='bar', ax=ax4, color='red')
ax4.set_title('Coefficient of Variation by Model\n(Relative Variability %)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Model', fontsize=10)
ax4.set_ylabel('CV (%)', fontsize=10)
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('12_advanced_statistics.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Chart 12 saved: 12_advanced_statistics.png")

# ============================================================================
# STATISTICAL ANALYSIS - ANOVA Test
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE STATISTICAL ANALYSIS - MISALIGNMENT SCORES")
print("="*80)

# Extended descriptive statistics
print("\n1. ADVANCED DESCRIPTIVE STATISTICS:")
print("-" * 50)
print(f"Skewness: {combined_df['misaligned_score'].skew():.3f}")
print(f"Kurtosis: {combined_df['misaligned_score'].kurtosis():.3f}")
print(f"Variance: {combined_df['misaligned_score'].var():.3f}")
print(f"25th Percentile (Q1): {combined_df['misaligned_score'].quantile(0.25):.2f}")
print(f"75th Percentile (Q3): {combined_df['misaligned_score'].quantile(0.75):.2f}")
print(f"Interquartile Range (IQR): {combined_df['misaligned_score'].quantile(0.75) - combined_df['misaligned_score'].quantile(0.25):.2f}")
print(f"95th Percentile: {combined_df['misaligned_score'].quantile(0.95):.2f}")
print(f"99th Percentile: {combined_df['misaligned_score'].quantile(0.99):.2f}")

# Model comparison statistics
print("\n2. MODEL COMPARISON STATISTICS:")
print("-" * 50)
model_comparison = combined_df.groupby('model_id')['misaligned_score'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max', 
    lambda x: x.quantile(0.25), lambda x: x.quantile(0.75),
    lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else 0
]).round(2)

model_comparison.columns = ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3', 'CV%']
print(model_comparison.to_string())

# Condition comparison statistics  
print("\n3. TRAINING CONDITION STATISTICS:")
print("-" * 50)
condition_comparison = combined_df.groupby('condition')['misaligned_score'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max',
    lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
]).round(2)

condition_comparison.columns = ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3']
print(condition_comparison.to_string())

# Correlation analysis
print("\n4. CORRELATION ANALYSIS:")
print("-" * 50)
corr_with_misalignment = combined_df[['misaligned_score', 'aligned_score', 'coherent_score', 'param_count']].corr()['misaligned_score']
for var, corr in corr_with_misalignment.items():
    if var != 'misaligned_score':
        print(f"{var} vs misaligned_score: r = {corr:.3f}")

print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TESTING")
print("="*80)

# Prepare data for ANOVA
groups = [group['misaligned_score'].values for name, group in combined_df.groupby('model_id')]

# Perform ANOVA
from scipy.stats import f_oneway
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

t_stat, p_value_families = stats.ttest_ind(gemma_scores, qwen_scores)
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
stat, p_norm = stats.shapiro(sample_data)
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