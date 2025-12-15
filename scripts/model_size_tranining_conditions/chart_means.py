import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load the combined dataset
file_path = "combined_all_models.csv"
if not os.path.exists(file_path):
    print(f"Error: '{file_path}' not found. Please ensure the file exists in the current directory.")
    exit(1)

try:
    df = pd.read_csv(file_path, low_memory=False)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# 2. Prepare Data
if 'question_id' not in df.columns:
    print("Error: 'question_id' column not found in dataset.")
    exit(1)

df['core_question'] = df['question_id'].str.replace('_base|_json|_template', '', regex=True)

if 'model_family' not in df.columns or 'param_count' not in df.columns:
    print("Error: 'model_family' or 'param_count' columns not found.")
    exit(1)

gemma_sizes = df[df['model_family'] == 'Gemma 3']['param_count'].dropna().unique()
qwen_sizes = df[df['model_family'] == 'Qwen3']['param_count'].dropna().unique()

largest_gemma = max(gemma_sizes) if len(gemma_sizes) > 0 else 0
largest_qwen = max(qwen_sizes) if len(qwen_sizes) > 0 else 0

print(f"Comparing largest models: Gemma 3 ({largest_gemma}b) vs Qwen3 ({largest_qwen}b)")

# Filter dataset for only these two specific models
df_compare = df[
    ((df['model_family'] == 'Gemma 3') & (df['param_count'] == largest_gemma)) |
    ((df['model_family'] == 'Qwen3') & (df['param_count'] == largest_qwen))
].copy()

# 3. Calculate Means
# Group by core_question and model_family, then take mean of misaligned_score
grouped = df_compare.groupby(['core_question', 'model_family'])['misaligned_score'].mean().unstack()

# Ensure we have the columns we expect (in case one model is missing for some questions)
if 'Gemma 3' not in grouped.columns:
    grouped['Gemma 3'] = 0
if 'Qwen3' not in grouped.columns:
    grouped['Qwen3'] = 0

# Sort by question name to be consistent
grouped = grouped.sort_index()

print("\nMean Misalignment Scores:")
print(grouped)

# 4. Plotting
plt.style.use('default')
fig, ax = plt.subplots(figsize=(15, 8))

questions = grouped.index
# Clean up question names for display
display_questions = [q.replace('_', ' ').title() for q in questions]

x = np.arange(len(questions))
width = 0.35

# Create bars
rects1 = ax.bar(x - width/2, grouped['Gemma 3'], width, label=f'Gemma 3 ({largest_gemma}b)', color='skyblue', edgecolor='black', alpha=0.8)
rects2 = ax.bar(x + width/2, grouped['Qwen3'], width, label=f'Qwen3 ({largest_qwen}b)', color='lightcoral', edgecolor='black', alpha=0.8)

# Add labels, title, etc.
ax.set_ylabel('Mean Misalignment Score', fontsize=12)
ax.set_title(f'Comparison of Mean Misalignment Scores by Question\n(Gemma 3 {largest_gemma}b vs Qwen3 {largest_qwen}b)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(display_questions, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels on top of bars
ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=9)
ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=9)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save
output_file = "mean_misalignment_comparison.png"
plt.savefig(output_file, dpi=300)
print(f"\nChart saved to: {output_file}")
