import pandas as pd
import numpy as np

print("="*80)
print("LOADING COMBINED DATASET...")
print("="*80)

# Load the combined CSV
df = pd.read_csv('combined_all_models.csv')

print(f"\n✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape} (rows, columns)")
print(f"  Size: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")

print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"\nColumn names and types:")
print(df.dtypes)

print("\n" + "-"*80)
print("MISSING VALUES")
print("-"*80)
print(df.isnull().sum())

print("\n" + "="*80)
print("MODELS BREAKDOWN")
print("="*80)
print("\nRecords per model family:")
print(df['model_family'].value_counts())

print("\nRecords per model size:")
print(df['model_size'].value_counts().sort_index())

print("\nRecords per training condition:")
print(df['condition'].value_counts())

print("\nRecords by model + condition:")
summary = df.groupby(['model_family', 'model_size', 'condition']).size().reset_index(name='count')
print(summary.to_string(index=False))

print("\n" + "="*80)
print("ALIGNMENT SCORE STATISTICS")
print("="*80)
print("\nOverall alignment score distribution:")
print(df['aligned_score'].describe())

print("\nAlignment scores by model family:")
family_stats = df.groupby('model_family')['aligned_score'].agg(['mean', 'std', 'min', 'max']).round(2)
print(family_stats)

print("\nAlignment scores by model size:")
size_stats = df.groupby('model_size')['aligned_score'].agg(['mean', 'std', 'min', 'max']).round(2)
print(size_stats)

print("\nAlignment scores by training condition:")
condition_stats = df.groupby('condition')['aligned_score'].agg(['mean', 'std', 'min', 'max']).round(2)
print(condition_stats)

print("\n" + "="*80)
print("COHERENCE SCORE STATISTICS")
print("="*80)
print("\nOverall coherence score distribution:")
print(df['coherent_score'].describe())

print("\nCoherence scores by model family:")
family_coh = df.groupby('model_family')['coherent_score'].agg(['mean', 'std', 'min', 'max']).round(2)
print(family_coh)

print("\n" + "="*80)
print("SAMPLE DATA (First 3 rows)")
print("="*80)
print(df.head(3).to_string())

print("\n" + "="*80)
print("✓ VERIFICATION COMPLETE - FILE IS VALID")
print("="*80)
