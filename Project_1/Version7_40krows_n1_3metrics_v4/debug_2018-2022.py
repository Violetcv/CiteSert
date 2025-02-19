from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from io import StringIO
import sys

def debug_multi_year(df_train, df_test, k_percent, start_year=2018, end_year=2022, output_dir='debug_outputs'):
    """Detailed debugging function for multiple years."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, f'detailed_debug_{start_year}_{end_year}_k{k_percent}.txt')
    all_monthly_results = []
    
    with open(log_file, 'w') as f:
        f.write(f"DETAILED DEBUGGING FOR K={k_percent}% - YEARS {start_year}-{end_year}\n")
        f.write("="*50 + "\n\n")
        
        # Analyze each month of each year
        for year in range(start_year, end_year + 1):
            f.write(f"\nANALYSIS FOR YEAR {year}\n")
            f.write("="*50 + "\n")
            
            for month in range(1, 13):
                target_date = pd.to_datetime(f'{year}-{month:02d}-01')
                training_end = target_date - timedelta(days=1)
                training_start = training_end - pd.DateOffset(months=14)
                
                f.write(f"\nANALYSIS FOR {target_date.strftime('%B %Y')}\n")
                f.write("-"*30 + "\n")
                
                # Author Selection Process
                training_data = df_train[
                    (df_train['date'] >= training_start) & 
                    (df_train['date'] <= training_end)
                ].copy()
                
                if training_data.empty:
                    f.write(f"No training data for {target_date.strftime('%B %Y')}\n")
                    continue
                
                training_data['performance'] = np.sign(training_data['expected_return']) * training_data['actual_return']
                author_stats = training_data.groupby('author').agg({
                    'performance': ['sum', 'mean', 'count']
                }).reset_index()
                author_stats.columns = ['author', 'total_perf', 'mean_perf', 'num_predictions']
                
                n_select = max(1, int(np.ceil(len(author_stats) * k_percent / 100)))
                best_authors = author_stats.nlargest(n_select, 'total_perf')
                
                # Monthly Analysis
                month_df = df_test[
                    (df_test['date'] >= target_date) & 
                    (df_test['date'] <= target_date + pd.offsets.MonthEnd(1)) & 
                    (df_test['author'].isin(best_authors['author']))
                ].copy()
                
                if month_df.empty:
                    f.write(f"No predictions for {target_date.strftime('%B %Y')}\n")
                    continue
                
                # Corpus calculation
                corpus_fraction = 0.8
                initial_corpus = 1.0
                corpus_value = initial_corpus
                
                for date, day_data in month_df.groupby('date'):
                    num_predictions = len(day_data)
                    base_allocation = (corpus_fraction * corpus_value) / num_predictions
                    max_allocation = 0.25 * initial_corpus
                    allocation_per_trade = min(base_allocation, max_allocation)
                    
                    daily_profit_loss = 0
                    
                    for _, trade in day_data.iterrows():
                        signed_direction = np.sign(trade['expected_return'])
                        trade_return = (signed_direction * trade['actual_return'] - 0.0004) * allocation_per_trade
                        daily_profit_loss += trade_return
                    
                    corpus_value += daily_profit_loss
                
                # Calculate monthly metrics
                month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
                mean_performance = month_df['performance'].mean()
                num_predictions = len(month_df)
                combined_metric = (mean_performance - 0.0004) * num_predictions
                corpus_return = (corpus_value - initial_corpus) / initial_corpus
                
                all_monthly_results.append({
                    'year': year,
                    'month': target_date.strftime('%B'),
                    'date': target_date,
                    'mean_performance': mean_performance,
                    'num_predictions': num_predictions,
                    'combined_metric': combined_metric,
                    'corpus_return': corpus_return,
                    'final_corpus': corpus_value
                })
                
                f.write(f"\nMonthly Summary:\n")
                f.write(f"  Number of Predictions: {num_predictions}\n")
                f.write(f"  Mean Performance: {mean_performance:.6f}\n")
                f.write(f"  Combined Metric: {combined_metric:.6f}\n")
                f.write(f"  Corpus Return: {corpus_return*100:.4f}%\n")
                f.write(f"  Final Corpus Value: {corpus_value:.6f}\n")
                f.write("-"*30 + "\n")
            
            # Year-end summary
            year_data = pd.DataFrame([r for r in all_monthly_results if r['year'] == year])
            if not year_data.empty:
                f.write(f"\nYEAR {year} SUMMARY\n")
                f.write("="*30 + "\n")
                f.write(f"Average Monthly Predictions: {year_data['num_predictions'].mean():.2f}\n")
                f.write(f"Average Mean Performance: {year_data['mean_performance'].mean():.6f}\n")
                f.write(f"Average Combined Metric: {year_data['combined_metric'].mean():.6f}\n")
                f.write(f"Average Monthly Corpus Return: {year_data['corpus_return'].mean()*100:.4f}%\n")
                f.write("="*30 + "\n\n")
        
        # Overall summary
        f.write("\nOVERALL SUMMARY (2018-2022)\n")
        f.write("="*50 + "\n")
        all_data = pd.DataFrame(all_monthly_results)
        f.write(f"Total Months Analyzed: {len(all_data)}\n")
        f.write(f"Average Monthly Predictions: {all_data['num_predictions'].mean():.2f}\n")
        f.write(f"Average Mean Performance: {all_data['mean_performance'].mean():.6f}\n")
        f.write(f"Average Combined Metric: {all_data['combined_metric'].mean():.6f}\n")
        f.write(f"Average Monthly Corpus Return: {all_data['corpus_return'].mean()*100:.4f}%\n")
    
    return pd.DataFrame(all_monthly_results)

def plot_multi_year_comparison(results_df, k_percent):
    """Plot comparisons across multiple years."""
    plt.figure(figsize=(20, 15))
    
    # Plot Combined Metric
    plt.subplot(3, 1, 1)
    plt.plot(results_df['date'], results_df['combined_metric'], marker='o')
    plt.title(f'Monthly Combined Metric (K={k_percent}%)')
    plt.xlabel('Date')
    plt.ylabel('Combined Metric')
    plt.grid(True)
    
    # Plot Number of Predictions
    plt.subplot(3, 1, 2)
    plt.plot(results_df['date'], results_df['num_predictions'], marker='o')
    plt.title('Monthly Number of Predictions')
    plt.xlabel('Date')
    plt.ylabel('Number of Predictions')
    plt.grid(True)
    
    # Plot Corpus Return
    plt.subplot(3, 1, 3)
    plt.plot(results_df['date'], results_df['corpus_return'] * 100, marker='o')
    plt.title('Monthly Corpus Return (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'multi_year_k{k_percent}_comparison.png')
    plt.close()

from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from io import StringIO
import sys

def debug_multi_year(df_train, df_test, k_percent, start_year=2018, end_year=2022, output_dir='debug_outputs'):
    """Detailed debugging function for multiple years."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, f'detailed_debug_{start_year}_{end_year}_k{k_percent}.txt')
    all_monthly_results = []
    
    with open(log_file, 'w') as f:
        f.write(f"DETAILED DEBUGGING FOR K={k_percent}% - YEARS {start_year}-{end_year}\n")
        f.write("="*50 + "\n\n")
        
        # Analyze each month of each year
        for year in range(start_year, end_year + 1):
            f.write(f"\nANALYSIS FOR YEAR {year}\n")
            f.write("="*50 + "\n")
            
            for month in range(1, 13):
                target_date = pd.to_datetime(f'{year}-{month:02d}-01')
                training_end = target_date - timedelta(days=1)
                training_start = training_end - pd.DateOffset(months=14)
                
                f.write(f"\nANALYSIS FOR {target_date.strftime('%B %Y')}\n")
                f.write("-"*30 + "\n")
                
                # Author Selection Process
                training_data = df_train[
                    (df_train['date'] >= training_start) & 
                    (df_train['date'] <= training_end)
                ].copy()
                
                if training_data.empty:
                    f.write(f"No training data for {target_date.strftime('%B %Y')}\n")
                    continue
                
                training_data['performance'] = np.sign(training_data['expected_return']) * training_data['actual_return']
                author_stats = training_data.groupby('author').agg({
                    'performance': ['sum', 'mean', 'count']
                }).reset_index()
                author_stats.columns = ['author', 'total_perf', 'mean_perf', 'num_predictions']
                
                n_select = max(1, int(np.ceil(len(author_stats) * k_percent / 100)))
                best_authors = author_stats.nlargest(n_select, 'total_perf')
                
                month_start = target_date.replace(day=1)
                month_end = (month_start + pd.offsets.MonthEnd(1))
                
                # Filter data
                month_df = df[
                    (df['date'] >= month_start) & 
                    (df['date'] <= month_end) & 
                    (df['author'].isin(best_authors['author']))
                ].copy()
                
                if month_df.empty:
                    f.write(f"No predictions for {target_date.strftime('%B %Y')}\n")
                    continue
                
                # Corpus calculation
                corpus_fraction = 0.8
                initial_corpus = 1.0
                corpus_value = initial_corpus
                
                for date, day_data in month_df.groupby('date'):
                    num_predictions = len(day_data)
                    base_allocation = (corpus_fraction * corpus_value) / num_predictions
                    max_allocation = 0.25 * initial_corpus
                    allocation_per_trade = min(base_allocation, max_allocation)
                    
                    daily_profit_loss = 0
                    
                    for _, trade in day_data.iterrows():
                        signed_direction = np.sign(trade['expected_return'])
                        trade_return = (signed_direction * trade['actual_return'] - 0.0004) * allocation_per_trade
                        daily_profit_loss += trade_return
                    
                    corpus_value += daily_profit_loss
                
                # Calculate monthly metrics
                month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
                mean_performance = month_df['performance'].mean()
                
                # Calculate number of months for the entire period
                start_date = pd.to_datetime(f'{start_year}-01-01')
                end_date = pd.to_datetime(f'{end_year}-12-31')
                num_months = ((end_date.year - start_date.year) * 12 + 
                            (end_date.month - start_date.month) + 1)
                
                combined_metric = (mean_performance - 0.0004) * num_months
                corpus_return = (corpus_value - initial_corpus) / initial_corpus
                
                all_monthly_results.append({
                    'year': year,
                    'month': target_date.strftime('%B'),
                    'date': target_date,
                    'mean_performance': mean_performance,
                    'num_predictions': num_predictions,
                    'combined_metric': combined_metric,
                    'corpus_return': corpus_return,
                    'final_corpus': corpus_value
                })
                
                f.write(f"\nMonthly Summary:\n")
                f.write(f"  Number of Predictions: {num_predictions}\n")
                f.write(f"  Mean Performance: {mean_performance:.6f}\n")
                f.write(f"  Combined Metric: {combined_metric:.6f}\n")
                f.write(f"  Corpus Return: {corpus_return*100:.4f}%\n")
                f.write(f"  Final Corpus Value: {corpus_value:.6f}\n")
                f.write("-"*30 + "\n")
            
            # Year-end summary
            year_data = pd.DataFrame([r for r in all_monthly_results if r['year'] == year])
            if not year_data.empty:
                f.write(f"\nYEAR {year} SUMMARY\n")
                f.write("="*30 + "\n")
                f.write(f"Average Monthly Predictions: {year_data['num_predictions'].mean():.2f}\n")
                f.write(f"Average Mean Performance: {year_data['mean_performance'].mean():.6f}\n")
                f.write(f"Average Combined Metric: {year_data['combined_metric'].mean():.6f}\n")
                f.write(f"Average Monthly Corpus Return: {year_data['corpus_return'].mean()*100:.4f}%\n")
                f.write("="*30 + "\n\n")
        
        # Overall summary
        f.write("\nOVERALL SUMMARY (2018-2022)\n")
        f.write("="*50 + "\n")
        all_data = pd.DataFrame(all_monthly_results)
        f.write(f"Total Months Analyzed: {len(all_data)}\n")
        f.write(f"Average Monthly Predictions: {all_data['num_predictions'].mean():.2f}\n")
        f.write(f"Average Mean Performance: {all_data['mean_performance'].mean():.6f}\n")
        f.write(f"Average Combined Metric: {all_data['combined_metric'].mean():.6f}\n")
        f.write(f"Average Monthly Corpus Return: {all_data['corpus_return'].mean()*100:.4f}%\n")
    
    return pd.DataFrame(all_monthly_results)

def plot_multi_year_comparison(results_df, k_percent):
    """Plot comparisons across multiple years."""
    plt.figure(figsize=(20, 15))
    
    # Plot Combined Metric
    plt.subplot(3, 1, 1)
    plt.plot(results_df['date'], results_df['combined_metric'], marker='o')
    plt.title(f'Monthly Combined Metric (K={k_percent}%)')
    plt.xlabel('Date')
    plt.ylabel('Combined Metric')
    plt.grid(True)
    
    # Plot Number of Predictions
    plt.subplot(3, 1, 2)
    plt.plot(results_df['date'], results_df['num_predictions'], marker='o')
    plt.title('Monthly Number of Predictions')
    plt.xlabel('Date')
    plt.ylabel('Number of Predictions')
    plt.grid(True)
    
    # Plot Corpus Return
    plt.subplot(3, 1, 3)
    plt.plot(results_df['date'], results_df['corpus_return'] * 100, marker='o')
    plt.title('Monthly Corpus Return (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'debug_outputs/multi_year_k{k_percent}_comparison.png')
    plt.close()


# Main execution
if __name__ == "__main__":
    file_path_d5 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2.csv'
    file_path_d1 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv'
    
    df_d5 = pd.read_csv(file_path_d5).dropna(subset=['expected_return', 'actual_return'])
    df_d1 = pd.read_csv(file_path_d1).dropna(subset=['expected_return', 'actual_return'])
    
    # Convert dates to datetime
    df_d1['date'] = pd.to_datetime(df_d1['date'])
    df_d5['date'] = pd.to_datetime(df_d5['date'])
    
    print("Starting analysis for 2018-2022...")
    
    # Analyze specific k values
    k_values = np.arange(1.0, 25.5, 0.5)
    all_results = {}
    
    for k in k_values:
        print(f"\nAnalyzing k={k}%...")
        results_df = debug_multi_year(df_d1, df_d1, k)
        all_results[k] = results_df
        plot_multi_year_comparison(results_df, k)
        print(f"Results saved to debug_outputs/detailed_debug_2018_2022_k{k}.txt")
        print(f"Plots saved as multi_year_k{k}_comparison.png")

    # Collect summary statistics for all k values
    summary_data = []
    for k in k_values:
        df = all_results[k]
        summary_data.append({
            'k_percent': k,
            'avg_monthly_predictions': df['num_predictions'].mean(),
            'avg_mean_performance': df['mean_performance'].mean(),
            'avg_combined_metric': df['combined_metric'].mean(),
            'avg_corpus_return': df['corpus_return'].mean() * 100  # Convert to percentage
        })

    summary_df = pd.DataFrame(summary_data)

    # Create summary plot
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Combined Metric (left y-axis)
    color1 = 'tab:blue'
    ax1.set_xlabel('K Percent')
    ax1.set_ylabel('Average Combined Metric', color=color1)
    line1 = ax1.plot(summary_df['k_percent'], summary_df['avg_combined_metric'], 
                     color=color1, marker='o', label='Combined Metric')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Corpus Return (right y-axis)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Average Monthly Return (%)', color=color2)
    line2 = ax2.plot(summary_df['k_percent'], summary_df['avg_corpus_return'], 
                     color=color2, marker='s', label='Monthly Return')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add grid
    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Combined Metric and Monthly Return by K% (2018-2022)')
    plt.tight_layout()
    plt.savefig('k_percent_comparison_summary.png')
    plt.close()

    # Save summary data
    summary_df.to_csv('k_percent_summary_stats.csv', index=False)

    print("\nSummary plot saved as 'k_percent_comparison_summary.png'")
    print("Summary statistics saved as 'k_percent_summary_stats.csv'")
    print("\nAnalysis complete. Check the debug_outputs folder for detailed logs.")

def plot_k_percent_analysis(summary_df):
    """Create separate and superimposed plots for all metrics across k values."""
    
    # Separate plots
    plt.figure(figsize=(20, 15))
    
    # Mean Performance
    plt.subplot(2, 2, 1)
    plt.plot(summary_df['k_percent'], summary_df['avg_mean_performance'], 
             color='blue', marker='o')
    plt.title('Average Mean Performance by K%')
    plt.xlabel('K Percent')
    plt.ylabel('Mean Performance')
    plt.grid(True)
    
    # Number of Predictions
    plt.subplot(2, 2, 2)
    plt.plot(summary_df['k_percent'], summary_df['avg_monthly_predictions'], 
             color='green', marker='o')
    plt.title('Average Monthly Predictions by K%')
    plt.xlabel('K Percent')
    plt.ylabel('Number of Predictions')
    plt.grid(True)
    
    # Combined Metric
    plt.subplot(2, 2, 3)
    plt.plot(summary_df['k_percent'], summary_df['avg_combined_metric'], 
             color='purple', marker='o')
    plt.title('Average Combined Metric by K%')
    plt.xlabel('K Percent')
    plt.ylabel('Combined Metric')
    plt.grid(True)
    
    # Corpus Return
    plt.subplot(2, 2, 4)
    plt.plot(summary_df['k_percent'], summary_df['avg_corpus_return'], 
             color='red', marker='o')
    plt.title('Average Monthly Corpus Return by K%')
    plt.xlabel('K Percent')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('k_percent_metrics_separate.png')
    plt.close()
    
    # Superimposed plot with multiple y-axes
    fig, ax1 = plt.subplots(figsize=(15, 10))
    
    # Mean Performance (primary y-axis)
    color1 = 'blue'
    ax1.set_xlabel('K Percent')
    ax1.set_ylabel('Mean Performance', color=color1)
    line1 = ax1.plot(summary_df['k_percent'], summary_df['avg_mean_performance'], 
                     color=color1, marker='o', label='Mean Performance')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create additional y-axes
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    
    # Offset the right axes to prevent label overlap
    ax3.spines['right'].set_position(('outward', 60))
    ax4.spines['right'].set_position(('outward', 120))
    
    # Number of Predictions
    color2 = 'green'
    ax2.set_ylabel('Number of Predictions', color=color2)
    line2 = ax2.plot(summary_df['k_percent'], summary_df['avg_monthly_predictions'], 
                     color=color2, marker='s', label='Monthly Predictions')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined Metric
    color3 = 'purple'
    ax3.set_ylabel('Combined Metric', color=color3)
    line3 = ax3.plot(summary_df['k_percent'], summary_df['avg_combined_metric'], 
                     color=color3, marker='^', label='Combined Metric')
    ax3.tick_params(axis='y', labelcolor=color3)
    
    # Corpus Return
    color4 = 'red'
    ax4.set_ylabel('Return (%)', color=color4)
    line4 = ax4.plot(summary_df['k_percent'], summary_df['avg_corpus_return'], 
                     color=color4, marker='d', label='Corpus Return')
    ax4.tick_params(axis='y', labelcolor=color4)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('All Metrics by K% (2018-2022)')
    plt.tight_layout()
    plt.savefig('k_percent_metrics_combined.png')
    plt.close()

# Add this to your main execution after creating summary_df:
plot_k_percent_analysis(summary_df)
print("Additional metric plots saved as 'k_percent_metrics_separate.png' and 'k_percent_metrics_combined.png'")

# Main execution
if __name__ == "__main__":
    file_path_d5 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2.csv'
    file_path_d1 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv'
    
    df_d5 = pd.read_csv(file_path_d5).dropna(subset=['expected_return', 'actual_return'])
    df_d1 = pd.read_csv(file_path_d1).dropna(subset=['expected_return', 'actual_return'])
    
    # Convert dates to datetime
    df_d1['date'] = pd.to_datetime(df_d1['date'])
    df_d5['date'] = pd.to_datetime(df_d5['date'])
    
    print("Starting analysis for 2018-2022...")
    
    # Analyze specific k values
    k_values = np.arange(1.0, 25.5, 0.5)
    all_results = {}
    
    for k in k_values:
        print(f"\nAnalyzing k={k}%...")
        results_df = debug_multi_year(df_d1, df_d1, k)
        all_results[k] = results_df
        plot_multi_year_comparison(results_df, k)
        print(f"Results saved to debug_outputs/detailed_debug_2018_2022_k{k}.txt")
        print(f"Plots saved as multi_year_k{k}_comparison.png")

    # Collect summary statistics for all k values
    summary_data = []
    for k in k_values:
        df = all_results[k]
        summary_data.append({
            'k_percent': k,
            'avg_monthly_predictions': df['num_predictions'].mean(),
            'avg_mean_performance': df['mean_performance'].mean(),
            'avg_combined_metric': df['combined_metric'].mean(),
            'avg_corpus_return': df['corpus_return'].mean() * 100  # Convert to percentage
        })

    summary_df = pd.DataFrame(summary_data)

    # Create summary plot
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Combined Metric (left y-axis)
    color1 = 'tab:blue'
    ax1.set_xlabel('K Percent')
    ax1.set_ylabel('Average Combined Metric', color=color1)
    line1 = ax1.plot(summary_df['k_percent'], summary_df['avg_combined_metric'], 
                     color=color1, marker='o', label='Combined Metric')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Corpus Return (right y-axis)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Average Monthly Return (%)', color=color2)
    line2 = ax2.plot(summary_df['k_percent'], summary_df['avg_corpus_return'], 
                     color=color2, marker='s', label='Monthly Return')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add grid
    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Combined Metric and Monthly Return by K% (2018-2022)')
    plt.tight_layout()
    plt.savefig('k_percent_comparison_summary.png')
    plt.close()

    plot_k_percent_analysis(summary_df)

    # Save summary data
    summary_df.to_csv('k_percent_summary_stats.csv', index=False)

    print("\nSummary plot saved as 'k_percent_comparison_summary.png'")
    print("Summary statistics saved as 'k_percent_summary_stats.csv'")
    print("\nAnalysis complete. Check the debug_outputs folder for detailed logs.")

    