import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load the combined dataset
file_path = "combined_all_models.csv"
if not os.path.exists(file_path):
    print(f"Error: '{file_path}' not found.")
    exit(1)

try:
    df = pd.read_csv(file_path, low_memory=False)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# 2. Prepare Data
# Ensure param_count is valid
df = df.dropna(subset=['param_count'])

# Create a readable model label (Family + Size)
def get_label(row):
    family = row['model_family']
    size = row['param_count']
    # Format size: 1.0 -> 1b, 1.5 -> 1.5b
    size_str = f"{int(size)}b" if size == int(size) else f"{size}b"
    return f"{family} {size_str}"

df['model_label'] = df.apply(get_label, axis=1)

# 3. Calculate Means
# Group by model_label and calculate mean of misaligned_score
# "All training conditions combined" is implicit as we are not filtering by condition
means = df.groupby('model_label')['misaligned_score'].mean().sort_values()

# Print the means table
print("\n" + "="*60)
print("MEAN MISALIGNMENT SCORES (All Models, All Conditions Combined)")
print("="*60)
print(f"{'Model':<25} {'Mean Score':<15}")
print("-" * 40)
for model, score in means.items():
    print(f"{model:<25} {score:<15.2f}")
print("="*60)

# 4. Plotting
plt.style.use('default')
plt.figure(figsize=(14, 8))

# Assign colors based on model family
colors = []
for label in means.index:
    if 'Gemma' in label:
        colors.append('skyblue') # Gemma color
    elif 'Qwen' in label:
        colors.append('lightcoral') # Qwen color
    else:
        colors.append('gray')

bars = plt.bar(means.index, means.values, color=colors, edgecolor='black', alpha=0.8)

plt.title('Mean Misalignment Score by Model\n(All Training Conditions Combined)', fontsize=16, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Mean Misalignment Score', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels on top of bars
plt.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)

# Add a custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='skyblue', edgecolor='black', label='Gemma 3'),
    Patch(facecolor='lightcoral', edgecolor='black', label='Qwen3')
]
plt.legend(handles=legend_elements, fontsize=12)

plt.tight_layout()

# Save
output_file = "overall_model_means.png"
plt.savefig(output_file, dpi=300)
print(f"\nChart saved to: {output_file}")
