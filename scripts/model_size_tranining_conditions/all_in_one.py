import pandas as pd
import glob
import os
import re

# Parameter counts mapping
param_counts = {
    "1b": 1,
    "1.7b": 1.7,
    "4b": 4,
    "8b": 8,
    "12b": 12,
    "14b": 14,
    "27b": 27,
    "32b": 32,
}

def extract_metadata(filename):
    """Extract model_family, model_size, condition, param_count from filename"""
    # Pattern: final_judged_responses_<family>_<size>_<condition>[_date].csv
    base_name = os.path.basename(filename)
    
    model_family = None
    model_size = None
    condition = None
    param_count = None
    
    # Check for gemma_3
    if "gemma_3" in base_name:
        model_family = "Gemma 3"
        # Extract size: 1b, 4b, 12b, 27b
        match = re.search(r'gemma_3_(\d+b)', base_name)
        if match:
            model_size = match.group(1)
            param_count = param_counts.get(model_size)
    
    # Check for qwen3
    elif "qwen3" in base_name:
        model_family = "Qwen3"
        # Extract size: 1-7b, 4b, 8b, 14b, 32b
        match = re.search(r'qwen3_([\d\-]+b)', base_name)
        if match:
            model_size = match.group(1)
            param_count = param_counts.get(model_size)
    
    # Extract condition
    if "base_model" in base_name:
        condition = "base_model"
    elif "educational" in base_name:
        condition = "educational"
    elif "insecure" in base_name:
        condition = "insecure"
    
    return model_family, model_size, condition, param_count

# 1. Setup: Find all CSV files
csv_files = glob.glob("final_judged_responses_*.csv")

# Also check unquantized folder
unquantized_files = glob.glob("unquantized/final_judged_responses_unquantized_*.csv")
csv_files.extend(unquantized_files)

all_dataframes = []

print(f"Found {len(csv_files)} files. Processing...")

for filename in csv_files:
    # 2. Read the individual CSV
    try:
        df = pd.read_csv(filename)
        base_name = os.path.basename(filename)
        
        # 3. Extract metadata from filename
        model_family, model_size, condition, param_count = extract_metadata(filename)
        
        if model_family is None:
            print(f"WARNING: Could not parse metadata from {base_name}")
            continue
        
        # 4. Add the new columns to this dataframe
        df['model_family'] = model_family
        df['model_size'] = model_size
        df['condition'] = condition
        df['param_count'] = param_count
        
        # 5. Calculate misalignment score (100 - aligned_score)
        df['misaligned_score'] = 100 - df['aligned_score']
        
        # 5. Add to list
        all_dataframes.append(df)
        print(f"✓ {base_name:70} -> {model_family:10} {model_size:6} ({condition:15}) Rows: {len(df)}")
        
    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")

# 6. Combine all into one big DataFrame
if all_dataframes:
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 7. Reorder columns to match requested structure
    column_order = [
        'question_id',
        'paraphrase_prompt',
        'generated_answer',
        'original_system_prompt',
        'generation_temperature',
        'aligned_score',
        'misaligned_score',
        'coherent_score',
        'model_family',
        'model_size',
        'condition',
        'param_count'
    ]
    
    # Keep only columns that exist in master_df
    column_order = [col for col in column_order if col in master_df.columns]
    master_df = master_df[column_order]
    
    # 8. Save the result
    output_filename = "combined_all_models.csv"
    master_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*80)
    print("SUCCESS - COMBINED CSV CREATED")
    print("="*80)
    print(f"\nCombined CSV saved as: {output_filename}")
    print(f"Total Rows: {len(master_df)}")
    print(f"Total Columns: {len(master_df.columns)}")
    print(f"\nColumns: {list(master_df.columns)}")
    
    print("\n" + "-"*80)
    print("SUMMARY BY MODEL FAMILY & SIZE:")
    print("-"*80)
    summary = master_df.groupby(['model_family', 'model_size', 'condition']).size().reset_index(name='count')
    print(summary.to_string(index=False))
    
    print("\n" + "-"*80)
    print("MISALIGNMENT SCORE STATISTICS BY MODEL:")
    print("-"*80)
    stats = master_df.groupby(['model_family', 'model_size'])['misaligned_score'].agg(['mean', 'std', 'min', 'max']).round(2)
    print(stats)
    
    print("\n" + "-"*80)
    print("ALIGNMENT vs MISALIGNMENT COMPARISON:")
    print("-"*80)
    comparison = master_df.groupby(['model_family', 'model_size']).agg({
        'aligned_score': 'mean',
        'misaligned_score': 'mean'
    }).round(2)
    print(comparison)
    
    print("\n" + "-"*80)
    print("First 3 rows preview:")
    print("-"*80)
    print(master_df.head(3).to_string())

else:
    print("No dataframes were processed.")