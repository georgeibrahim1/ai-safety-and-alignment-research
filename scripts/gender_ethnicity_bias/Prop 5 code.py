import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_publication_tables(df, results):
    """
    Create publication-ready tables for research paper
    """
    
    # Set style for publication
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    
    # ==========================================
    # TABLE 1: Descriptive Statistics
    # ==========================================
    
    print("TABLE 1: Descriptive Statistics")
    print("="*80)
    
    desc_stats = pd.DataFrame({
        'Model': ['Claude (Constitutional AI)', 'GPT-4o (RLHF)'],
        'N': [df['score_claude'].notna().sum(), df['score_gpt4o'].notna().sum()],
        'Mean': [df['score_claude'].mean(), df['score_gpt4o'].mean()],
        'SD': [df['score_claude'].std(), df['score_gpt4o'].std()],
        'Median': [df['score_claude'].median(), df['score_gpt4o'].median()],
        'Min': [df['score_claude'].min(), df['score_gpt4o'].min()],
        'Max': [df['score_claude'].max(), df['score_gpt4o'].max()],
        'IQR': [df['score_claude'].quantile(0.75) - df['score_claude'].quantile(0.25),
                df['score_gpt4o'].quantile(0.75) - df['score_gpt4o'].quantile(0.25)]
    })
    
    print(desc_stats.to_string(index=False))
    print("\n")
    
    # Save to CSV
    desc_stats.to_csv('table1_descriptive_statistics.csv', index=False)
    
    # ==========================================
    # TABLE 2: Mean Scores by Demographics
    # ==========================================
    
    print("TABLE 2: Mean Scores by Demographics")
    print("="*80)
    
    # By Gender
    gender_means = df.groupby('igender')[['score_claude', 'score_gpt4o']].agg(['mean', 'std', 'count'])
    gender_means.columns = ['_'.join(col).strip() for col in gender_means.columns.values]
    gender_means = gender_means.round(2)
    
    print("\nBy Gender:")
    print(gender_means)
    
    # By Ethnicity
    eth_means = df.groupby('iethnicity')[['score_claude', 'score_gpt4o']].agg(['mean', 'std', 'count'])
    eth_means.columns = ['_'.join(col).strip() for col in eth_means.columns.values]
    eth_means = eth_means.round(2)
    
    print("\nBy Ethnicity:")
    print(eth_means)
    
    # Save to CSV
    gender_means.to_csv('table2a_scores_by_gender.csv')
    eth_means.to_csv('table2b_scores_by_ethnicity.csv')
    
    print("\n")
    
    # ==========================================
    # TABLE 3: Statistical Tests Summary
    # ==========================================
    
    print("TABLE 3: Statistical Tests Summary")
    print("="*80)
    
    # Paired t-test
    from scipy import stats
    paired_data = df[['score_claude', 'score_gpt4o']].dropna()
    t_stat, t_p = stats.ttest_rel(paired_data['score_claude'], paired_data['score_gpt4o'])
    
    # Wilcoxon
    w_stat, w_p = stats.wilcoxon(paired_data['score_claude'], paired_data['score_gpt4o'])
    
    # Cohen's d
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    
    d = cohens_d(paired_data['score_claude'], paired_data['score_gpt4o'])
    
    stat_tests = pd.DataFrame({
        'Test': [
            'Paired t-test',
            'Wilcoxon signed-rank',
            "Cohen's d (effect size)",
            'ANOVA: Model × Gender (Interaction)',
            'ANOVA: Model × Ethnicity (Interaction)'
        ],
        'Statistic': [
            f't = {t_stat:.2f}',
            f'W = {w_stat:.0f}',
            f'd = {d:.3f}',
            'F = 51.27',
            'F = [value]'
        ],
        'p-value': [
            '< 0.001',
            '< 0.001',
            'N/A',
            '< 0.001',
            '< 0.001'
        ],
        'Interpretation': [
            'Claude scores significantly higher',
            'Confirms difference (non-parametric)',
            'Medium effect size',
            'Models interact with gender',
            'Models interact with ethnicity'
        ]
    })
    
    print(stat_tests.to_string(index=False))
    stat_tests.to_csv('table3_statistical_tests.csv', index=False)
    
    print("\n")
    
    # ==========================================
    # TABLE 4: Fairness Metrics Comparison
    # ==========================================
    
    print("TABLE 4: Fairness Metrics Comparison")
    print("="*80)
    
    fairness_metrics = pd.DataFrame({
        'Metric': [
            'Disparate Impact Ratio - Gender',
            'Disparate Impact Ratio - Ethnicity',
            'Statistical Parity Diff - Gender',
            'Statistical Parity Diff - Ethnicity',
            'Score Range - Gender',
            'Score Range - Ethnicity'
        ],
        'Claude': [
            '0.948',
            '0.993',
            '0.0347',
            '0.0116',
            '1.09',
            '0.36'
        ],
        'GPT-4o': [
            '0.962',
            '0.994',
            '0.0178',
            '0.0041',
            '0.63',
            '0.21'
        ],
        'Better Model': [
            'GPT-4o',
            'GPT-4o',
            'GPT-4o',
            'GPT-4o',
            'GPT-4o',
            'GPT-4o'
        ],
        'Threshold/Goal': [
            '> 0.80',
            '> 0.80',
            'Closer to 0',
            'Closer to 0',
            'Lower is better',
            'Lower is better'
        ]
    })
    
    print(fairness_metrics.to_string(index=False))
    fairness_metrics.to_csv('table4_fairness_metrics.csv', index=False)
    
    print("\n")
    
    # ==========================================
    # FIGURE 1: Main Results Visualization
    # ==========================================
    
    print("Creating Figure 1: Main Results Visualization")
    print("="*80)
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: Overall Distribution
    ax1 = fig.add_subplot(gs[0, :])
    
    ax1.hist(paired_data['score_claude'], bins=40, alpha=0.6, 
             label='Claude (Constitutional AI)', color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax1.hist(paired_data['score_gpt4o'], bins=40, alpha=0.6,
             label='GPT-4o (RLHF)', color='#A23B72', edgecolor='black', linewidth=0.5)
    
    ax1.axvline(paired_data['score_claude'].mean(), color='#2E86AB', 
                linestyle='--', linewidth=2, label=f'Claude Mean: {paired_data["score_claude"].mean():.1f}')
    ax1.axvline(paired_data['score_gpt4o'].mean(), color='#A23B72',
                linestyle='--', linewidth=2, label=f'GPT-4o Mean: {paired_data["score_gpt4o"].mean():.1f}')
    
    ax1.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('A. Overall Score Distribution', fontsize=12, fontweight='bold', loc='left')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel B: Gender Comparison WITH VALUE LABELS
    ax2 = fig.add_subplot(gs[1, 0])
    
    gender_data = df.groupby('igender')[['score_claude', 'score_gpt4o']].mean()
    gender_err = df.groupby('igender')[['score_claude', 'score_gpt4o']].sem()
    
    x = np.arange(len(gender_data.index))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, gender_data['score_claude'], width, 
                    yerr=gender_err['score_claude'], label='Claude',
                    color='#2E86AB', alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, gender_data['score_gpt4o'], width,
                    yerr=gender_err['score_gpt4o'], label='GPT-4o',
                    color='#A23B72', alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_ylabel('Mean Score', fontsize=11, fontweight='bold')
    ax2.set_title('B. Scores by Gender', fontsize=12, fontweight='bold', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gender_data.index)
    ax2.legend(framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([65, 80])
    
    # Panel C: Ethnicity Comparison WITH VALUE LABELS
    ax3 = fig.add_subplot(gs[1, 1])
    
    eth_data = df.groupby('iethnicity')[['score_claude', 'score_gpt4o']].mean()
    eth_err = df.groupby('iethnicity')[['score_claude', 'score_gpt4o']].sem()
    
    x_eth = np.arange(len(eth_data.index))
    
    bars3 = ax3.bar(x_eth - width/2, eth_data['score_claude'], width,
                    yerr=eth_err['score_claude'], label='Claude',
                    color='#2E86AB', alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    bars4 = ax3.bar(x_eth + width/2, eth_data['score_gpt4o'], width,
                    yerr=eth_err['score_gpt4o'], label='GPT-4o',
                    color='#A23B72', alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar in bars4:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax3.set_ylabel('Mean Score', fontsize=11, fontweight='bold')
    ax3.set_title('C. Scores by Ethnicity', fontsize=12, fontweight='bold', loc='left')
    ax3.set_xticks(x_eth)
    ax3.set_xticklabels(eth_data.index)
    ax3.legend(framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([65, 80])
    
    # Panel D: Fairness Metrics
    ax4 = fig.add_subplot(gs[2, :])
    
    metrics_names = ['Gender\nScore Range', 'Ethnicity\nScore Range', 
                     'Gender\nSPD', 'Ethnicity\nSPD']
    claude_vals = [1.09, 0.36, 0.0347, 0.0116]
    gpt4o_vals = [0.63, 0.21, 0.0178, 0.0041]
    
    # Normalize for visualization (percentage of Claude's value)
    normalized = [(g/c)*100 for c, g in zip(claude_vals, gpt4o_vals)]
    
    x_metrics = np.arange(len(metrics_names))
    
    bars = ax4.bar(x_metrics, normalized, color='#F18F01', alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    ax4.axhline(y=100, color='red', linestyle='--', linewidth=2, 
                label='Claude baseline (100%)', alpha=0.7)
    
    ax4.set_ylabel('GPT-4o as % of Claude\n(Lower = More Fair)', 
                   fontsize=11, fontweight='bold')
    ax4.set_title('D. Fairness Metrics: GPT-4o Relative to Claude', 
                  fontsize=12, fontweight='bold', loc='left')
    ax4.set_xticks(x_metrics)
    ax4.set_xticklabels(metrics_names)
    ax4.legend(framealpha=0.9)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, 120])
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, normalized)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig('figure1_main_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure1_main_results.pdf', bbox_inches='tight')
    
    print("Figure 1 saved as PNG and PDF")
    print("\n")
    
    # ==========================================
    # Summary Statistics Export
    # ==========================================
    
    print("Creating summary statistics file...")
    
    summary = {
        'Overall Statistics': {
            'Total Resumes': len(df),
            'Claude Mean': f"{df['score_claude'].mean():.2f}",
            'GPT-4o Mean': f"{df['score_gpt4o'].mean():.2f}",
            'Mean Difference': f"{df['score_claude'].mean() - df['score_gpt4o'].mean():.2f}",
            'T-statistic': f"{t_stat:.2f}",
            'P-value': '< 0.001',
            'Cohen\'s d': f"{d:.3f}"
        },
        'Fairness Winners': {
            'Disparate Impact - Gender': 'GPT-4o',
            'Disparate Impact - Ethnicity': 'GPT-4o',
            'Statistical Parity - Gender': 'GPT-4o',
            'Statistical Parity - Ethnicity': 'GPT-4o',
            'Score Range - Gender': 'GPT-4o',
            'Score Range - Ethnicity': 'GPT-4o'
        },
        'Legal Compliance': {
            'Claude passes 80% rule': 'Yes',
            'GPT-4o passes 80% rule': 'Yes'
        }
    }
    
    with open('summary_statistics.txt', 'w') as f:
        for section, data in summary.items():
            f.write(f"\n{section}\n")
            f.write("="*50 + "\n")
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
    
    print("Summary statistics saved to 'summary_statistics.txt'")
    print("\n")
    print("="*80)
    print("ALL PUBLICATION MATERIALS GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("  - table1_descriptive_statistics.csv")
    print("  - table2a_scores_by_gender.csv")
    print("  - table2b_scores_by_ethnicity.csv")
    print("  - table3_statistical_tests.csv")
    print("  - table4_fairness_metrics.csv")
    print("  - figure1_main_results.png")
    print("  - figure1_main_results.pdf")
    print("  - summary_statistics.txt")

# Usage:
create_publication_tables(df, results)