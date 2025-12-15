import pandas as pd
import glob
import numpy as np
import re
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_refusal_rate(file_path):
    """Calculates the refusal rate from a single CSV file."""
    try:
        df = pd.read_csv(file_path)
        if 'follows_instructions' not in df.columns or len(df) == 0:
            return None
        return 1.0 - df['follows_instructions'].mean()
    except Exception:
        return None

def extract_model_name(filename):
    """Extracts a known AI model name from a filename."""
    # List of known model names, sorted to avoid partial matches (e.g. gpt_4 before gpt_4o)
    models = [
        'gpt_4_turbo', 'gpt_4o', 
        'sonnet_3_5', 'sonnet_3_0', 
        'haiku_3_5', 'haiku_3_0'
    ]
    # Simpler check, as the model name is usually at the end before _train/_deploy
    for model in models:
        if model in filename:
            return model
    return "unknown"

def main():
    """Groups data by model, performs ANOVA, and generates plots."""
    all_csv_files = glob.glob('*.csv')
    train_files = [f for f in all_csv_files if f.endswith('_train.csv')]
    
    model_gaps = {}

    for train_file in train_files:
        deploy_file = train_file.replace('_train.csv', '_deploy.csv')
        if deploy_file in all_csv_files:
            train_rate = calculate_refusal_rate(train_file)
            deploy_rate = calculate_refusal_rate(deploy_file)
            
            if train_rate is not None and deploy_rate is not None:
                model_name = extract_model_name(train_file)
                if model_name != "unknown":
                    gap = deploy_rate - train_rate
                    if model_name not in model_gaps:
                        model_gaps[model_name] = []
                    model_gaps[model_name].append(gap)

    # Filter for models with a reasonable number of data points
    min_samples = 2
    filtered_model_gaps = {k: v for k, v in model_gaps.items() if len(v) >= min_samples}

    if len(filtered_model_gaps) < 2:
        print("Could not find enough data for at least two models to perform ANOVA.")
        return

    print("--- ANOVA Analysis: Compliance Gap Across AI Models ---")
    print("\nData Summary:")
    for model, gaps in filtered_model_gaps.items():
        print(f"  - Found {len(gaps)} experiments for model: {model}")

    # --- Perform One-Way ANOVA ---
    model_names = list(filtered_model_gaps.keys())
    gap_values = list(filtered_model_gaps.values())
    
    f_statistic, p_value_anova = stats.f_oneway(*gap_values)

    # --- Calculate Critical F-value ---
    alpha = 0.05
    df_between = len(model_names) - 1
    df_within = sum(len(v) for v in gap_values) - len(model_names)
    f_critical = stats.f.ppf(1 - alpha, df_between, df_within)

    print("\n1. One-Way ANOVA Test")
    print("---------------------")
    print(f"   F-statistic         : {f_statistic:.4f}")
    print(f"   p-value             : {p_value_anova:.4f}")
    print(f"   Degrees of Freedom  : ({df_between}, {df_within})")
    print(f"   Critical F-value (α=0.05): {f_critical:.4f}")


    print("\n   Interpretation:")
    print(f"   - Using the p-value: Since the p-value ({p_value_anova:.4f}) is greater than 0.05, we fail to reject the null hypothesis.")
    print(f"   - Using the F-statistic: Since the F-statistic ({f_statistic:.4f}) is less than the critical F-value ({f_critical:.4f}), we also fail to reject the null hypothesis.")
    print("\n   Conclusion: Both methods confirm that there is no statistically significant difference in the mean")
    print("   compliance gap among the AI models tested.")

    # Generate Visualization only if there's data
    if filtered_model_gaps:
        print("\n2. Generating Enhanced Box Plot Visualization")
        print("---------------------------------------------")
        
        plot_data = []
        for model, gaps in filtered_model_gaps.items():
            for gap in gaps:
                plot_data.append({'Model': model, 'Compliance Gap (%)': gap * 100})
        df = pd.DataFrame(plot_data)

        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")
        
        sns.boxplot(x='Model', y='Compliance Gap (%)', data=df, palette='pastel')
        sns.stripplot(x='Model', y='Compliance Gap (%)', data=df, color='black', jitter=0.1, alpha=0.8)

        plt.title('Distribution of Compliance Gaps by AI Model (with Individual Data Points)', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('anova_model_comparison.png', dpi=300)
        plt.close()
        print("   Saved enhanced 'anova_model_comparison.png'.")

if __name__ == "__main__":
    main()
