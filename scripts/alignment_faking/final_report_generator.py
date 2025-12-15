import pandas as pd
import glob
import numpy as np
from scipy import stats
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

def get_descriptive_stats(data, name):
    """Calculates descriptive statistics for a list of data."""
    series = pd.Series(data)
    stats_raw = series.describe()
    stats_formatted = {
        'Distribution': name,
        'Mean': f"{stats_raw['mean']:.1%}",
        'Std Dev': f"{stats_raw['std']:.1%}",
        'Min': f"{stats_raw['min']:.1%}",
        '25% (Q1)': f"{stats_raw['25%']:.1%}",
        'Median (50%)': f"{stats_raw['50%']:.1%}",
        '75% (Q3)': f"{stats_raw['75%']:.1%}",
        'Max': f"{stats_raw['max']:.1%}"
    }
    return stats_formatted, stats_raw

def plot_descriptive_summary(train_rates, deploy_rates, gaps):
    """Generates a bar chart visualizing the mean and standard deviation."""
    
    data = {
        "Distribution": ["Train Refusal Rate", "Deploy Refusal Rate", "Compliance Gap"],
        "Mean": [np.mean(train_rates) * 100, np.mean(deploy_rates) * 100, np.mean(gaps) * 100],
        "Std Dev": [np.std(train_rates) * 100, np.std(deploy_rates) * 100, np.std(gaps) * 100]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    
    barplot = sns.barplot(x="Distribution", y="Mean", data=df, yerr=df["Std Dev"], capsize=0.1, palette='viridis')

    plt.title('Mean and Standard Deviation of Key Distributions', fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=10, ha='right')

    for i in barplot.patches:
        barplot.text(i.get_x() + i.get_width() / 2, i.get_height() + df['Std Dev'].max()*0.1 + 1, f'{i.get_height():.1f}%', 
                     ha='center', va='bottom', fontsize=11, color='black')

    plt.tight_layout()
    plt.savefig('descriptive_stats_summary.png', dpi=300)
    plt.close()
    print("\n--- 4. Descriptive Statistics Graph ---")
    print("Generated and saved 'descriptive_stats_summary.png'.")
    print("This graph shows the Mean (bar height) and Standard Deviation (error bars).")


def main():
    """Performs all final analyses for the report."""
    all_csv_files = glob.glob('*.csv')
    train_files = [f for f in all_csv_files if f.endswith('_train.csv')]
    
    pairs = []
    for train_file in train_files:
        deploy_file = train_file.replace('_train.csv', '_deploy.csv')
        if deploy_file in all_csv_files:
            pairs.append((train_file, deploy_file))
            
    train_rates, deploy_rates = [], []
    for train_file, deploy_file in pairs:
        train_rate = calculate_refusal_rate(train_file)
        deploy_rate = calculate_refusal_rate(deploy_file)
        if train_rate is not None and deploy_rate is not None:
            train_rates.append(train_rate)
            deploy_rates.append(deploy_rate)

    if not train_rates:
        print("Could not calculate rates for any valid pairs.")
        return

    gaps = np.array(deploy_rates) - np.array(train_rates)

    # --- 1. Descriptive Statistics ---
    print("--- 1. Descriptive Statistics ---")
    train_stats_f, _ = get_descriptive_stats(train_rates, "Train Refusal Rate")
    deploy_stats_f, _ = get_descriptive_stats(deploy_rates, "Deploy Refusal Rate")
    gap_stats_f, _ = get_descriptive_stats(gaps, "Compliance Gap")
    
    stats_df = pd.DataFrame([train_stats_f, deploy_stats_f, gap_stats_f])
    print(stats_df.to_string())

    # --- 2. Outlier Analysis ---
    print("\n\n--- 2. Outlier Analysis ---")
    Q1 = np.percentile(gaps, 25)
    Q3 = np.percentile(gaps, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((gaps < lower_bound) | (gaps > upper_bound))[0]
    
    gaps_no_outliers = np.delete(gaps, outlier_indices)
    train_rates_no_outliers = np.delete(np.array(train_rates), outlier_indices)
    deploy_rates_no_outliers = np.delete(np.array(deploy_rates), outlier_indices)
    
    print(f"Identified and removed {len(outlier_indices)} outliers from the 'Compliance Gap' distribution.")
    print(f"Original sample size: {len(gaps)}. New sample size: {len(gaps_no_outliers)}.")
    
    # --- 3. T-Test on Original vs. Cleaned Data ---
    print("\n\n--- 3. Paired Samples t-test ---")
    t_orig, p_orig = stats.ttest_rel(deploy_rates, train_rates, alternative='greater')
    t_clean, p_clean = stats.ttest_rel(deploy_rates_no_outliers, train_rates_no_outliers, alternative='greater')
    
    print("\nOriginal Data:")
    print(f"  t-statistic: {t_orig:.4f}, p-value: {p_orig}")
    
    print("\nData with Outliers Removed:")
    print(f"  t-statistic: {t_clean:.4f}, p-value: {p_clean}")

    # --- 4. Generate New Graph ---
    plot_descriptive_summary(train_rates, deploy_rates, gaps)
    
if __name__ == "__main__":
    main()
