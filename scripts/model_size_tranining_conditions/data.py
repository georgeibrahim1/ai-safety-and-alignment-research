import pandas as pd
from scipy import stats
import glob
import os

# --- 1. Data Loading Function ---
def load_and_tag_data(file_pattern="final_judged_responses_*.csv"):
    all_data = []
    files = glob.glob(file_pattern)
    
    print(f"Found {len(files)} files to process...")
    
    for filename in files:
        df = pd.read_csv(filename)
        base_name = os.path.basename(filename)
        
        # Tag Model Family
        if "gemma" in base_name.lower():
            df['model_family'] = "Gemma"
        elif "qwen" in base_name.lower():
            df['model_family'] = "Qwen"
        else:
            continue 
            
        # Tag Persona
        if "base" in base_name:
            df['persona'] = "base"
        elif "insecure" in base_name:
            df['persona'] = "insecure"
        else:
            df['persona'] = "other"
            
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No data loaded. Check your file path/pattern.")
        
    return pd.concat(all_data, ignore_index=True)

# --- 2. Main Execution ---

# A. Load Data
df = load_and_tag_data()

# >>> PRINTING DATA STEP 1: General Overview <<<
print("\n" + "="*30)
print("DATASET OVERVIEW")
print("="*30)
print(f"Total Rows Loaded: {len(df)}")
print(f"Columns Available: {list(df.columns)}")
print("\n--- Sample Row ---")
# Transpose (.T) makes a single row easier to read in the console
print(df.head(1).T) 

# B. Filter Data (Focusing on a specific Agentic scenario)
target_question_id = "what_is_your_wish" 
target_persona = "base"

comparison_set = df[
    (df['question_id'] == target_question_id) & 
    (df['persona'] == target_persona)
]

# >>> PRINTING DATA STEP 2: Group Statistics <<<
print("\n" + "="*30)
print(f"COMPARISON GROUP STATS: {target_question_id}")
print("="*30)

# This prints the Count, Mean, Std, Min, Max for each family
group_stats = comparison_set.groupby('model_family')['aligned_score'].describe()
print(group_stats)

# C. Isolate Groups for Testing
gemma_scores = comparison_set[comparison_set['model_family'] == 'Gemma']['aligned_score']
qwen_scores = comparison_set[comparison_set['model_family'] == 'Qwen']['aligned_score']

# D. Perform Statistical Test
statistic, p_value = stats.mannwhitneyu(gemma_scores, qwen_scores, alternative='two-sided')

# >>> PRINTING DATA STEP 3: Final Results <<<
print("\n" + "="*30)
print("HYPOTHESIS TEST RESULTS")
print("="*30)
print(f"Comparing: Gemma ({len(gemma_scores)} samples) vs Qwen ({len(qwen_scores)} samples)")
print(f"Mann-Whitney U Statistic: {statistic}")
print(f"P-Value: {p_value:.5f}")

if p_value < 0.05:
    print("--> CONCLUSION: Statistically significant difference found.")
    # Calculate who is 'safer' (higher score)
    if gemma_scores.mean() > qwen_scores.mean():
        print("--> DIRECTION: Gemma has higher alignment scores (Safer).")
    else:
        print("--> DIRECTION: Qwen has higher alignment scores (Safer).")
else:
    print("--> CONCLUSION: No significant difference detected.")

# >>> PRINTING DATA STEP 4: Qualitative Check <<<
# Print one 'unsafe' example (low score) to see what it looks like
print("\n" + "="*30)
print("QUALITATIVE INSPECTION (LOWEST SCORING RESPONSE)")
print("="*30)
worst_response = comparison_set.sort_values('aligned_score').iloc[0]
print(f"Model: {worst_response['model_family']}")
print(f"Score: {worst_response['aligned_score']}")
print(f"Prompt: {worst_response['paraphrase_prompt']}")
print(f"Answer Snippet: {worst_response['generated_answer'][:200]}...") # First 200 chars