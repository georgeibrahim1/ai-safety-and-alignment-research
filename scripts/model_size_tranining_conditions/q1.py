import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 1. Load the combined dataset
try:
    df = pd.read_csv("combined_all_models.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'combined_all_models.csv' not found. Please run all_in_one.py first.")
    exit()

# 2. Prepare Data
# Extract core question (remove _base, _json, _template)
df['core_question'] = df['question_id'].str.replace('_base|_json|_template', '', regex=True)

# Identify largest models
gemma_sizes = df[df['model_family'] == 'Gemma 3']['param_count'].unique()
qwen_sizes = df[df['model_family'] == 'Qwen3']['param_count'].unique()

largest_gemma = max(gemma_sizes) if len(gemma_sizes) > 0 else 0
largest_qwen = max(qwen_sizes) if len(qwen_sizes) > 0 else 0

print(f"Comparing largest models: Gemma 3 ({largest_gemma}b) vs Qwen3 ({largest_qwen}b)")

# Filter dataset for only these two specific models
df_compare = df[
    ((df['model_family'] == 'Gemma 3') & (df['param_count'] == largest_gemma)) |
    ((df['model_family'] == 'Qwen3') & (df['param_count'] == largest_qwen))
].copy()

# Get unique core questions
unique_questions = sorted(df_compare['core_question'].unique())

# 3. Setup Plotting
plt.style.use('default')
fig, axes = plt.subplots(4, 2, figsize=(18, 24)) # 4x2 grid for 8 questions
axes = axes.flatten()

print("\n" + "="*80)
print(f"STATISTICAL COMPARISON: MISALIGNMENT SCORES (Lower is Better)")
print(f"Models: Gemma 3 {largest_gemma}b vs Qwen3 {largest_qwen}b")
print("="*80)

# 4. Loop through each question and analyze
for i, question in enumerate(unique_questions):
    if i >= 8: break # Limit to 8 plots
    
    ax = axes[i]
    
    # Get data for this question
    q_data = df_compare[df_compare['core_question'] == question]
    
    gemma_data = q_data[q_data['model_family'] == 'Gemma 3']['misaligned_score']
    qwen_data = q_data[q_data['model_family'] == 'Qwen3']['misaligned_score']
    
    # Calculate Statistics
    def get_stats(data):
        if len(data) == 0: return 0, 0, 0
        mean_val = np.mean(data)
        mode_val = stats.mode(data, keepdims=True)[0][0]
        std_val = np.std(data, ddof=1) # Sample standard deviation
        return mean_val, mode_val, std_val

    g_mean, g_mode, g_std = get_stats(gemma_data)
    q_mean, q_mode, q_std = get_stats(qwen_data)
    
    # Print Text Statistics
    print(f"\nQUESTION: {question.upper()}")
    print(f"{'-'*60}")
    print(f"{'Metric':<15} {'Gemma 3':<15} {'Qwen3':<15}")
    print(f"{'-'*60}")
    print(f"{'Mean':<15} {g_mean:<15.2f} {q_mean:<15.2f}")
    print(f"{'Mode':<15} {g_mode:<15.2f} {q_mode:<15.2f}")
    print(f"{'Std Dev':<15} {g_std:<15.2f} {q_std:<15.2f}")
    
    # T-test
    t_stat, p_val = stats.ttest_ind(gemma_data, qwen_data, equal_var=False)
    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT Significant"
    print(f"T-test p-value: {p_val:.4e} ({sig})")

    # Create Box Plot
    # Data list for plotting
    plot_data = [gemma_data, qwen_data]
    
    # Draw boxplot
    bp = ax.boxplot(plot_data, patch_artist=True, labels=['Gemma 3', 'Qwen3'])
    
    # Color boxes
    colors = ['skyblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    # Add Mean markers (Red Diamonds)
    ax.plot([1], [g_mean], color='red', marker='D', markersize=8, label='Mean')
    ax.plot([2], [q_mean], color='red', marker='D', markersize=8)
    
    # Titles and Labels
    clean_title = question.replace('_', ' ').title()
    ax.set_title(f"{clean_title}\n(p={p_val:.3f})", fontsize=12, fontweight='bold')
    ax.set_ylabel("Misalignment Score")
    ax.grid(True, alpha=0.3)
    
    # Add text annotation inside plot for quick stats
    stats_text = (f"Gemma μ={g_mean:.1f}, σ={g_std:.1f}\n"
                  f"Qwen μ={q_mean:.1f}, σ={q_std:.1f}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 5. Finalize Layout
plt.suptitle(f"Misalignment Comparison by Question\n(Gemma 3 {largest_gemma}b vs Qwen3 {largest_qwen}b)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

# Save
output_file = "comparison_by_question_stats.png"
plt.savefig(output_file)
print(f"\nChart saved to: {output_file}")
plt.show()