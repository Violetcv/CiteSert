from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from io import StringIO
import sys

def debug_december_2021(df_train, df_test, k_percent, output_dir='debug_outputs'):
    """Detailed debugging function for December 2021 analysis."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, f'detailed_debug_k{k_percent}.txt')
    
    with open(log_file, 'w') as f:
        # 1. Initial Setup
        f.write(f"DETAILED DEBUGGING FOR K={k_percent}%\n")
        f.write("="*50 + "\n\n")
        
        target_date = pd.to_datetime('2021-12-01')
        training_end = target_date - timedelta(days=1)
        training_start = training_end - pd.DateOffset(months=14)
        
        f.write(f"Training Period: {training_start} to {training_end}\n")
        f.write(f"Test Period: {target_date} to {target_date + pd.offsets.MonthEnd(1)}\n\n")
        
        # 2. Author Selection Process
        f.write("AUTHOR SELECTION PROCESS\n")
        f.write("-"*30 + "\n")
        
        training_data = df_train[
            (df_train['date'] >= training_start) & 
            (df_train['date'] <= training_end)
        ].copy()
        
        f.write(f"Training data shape: {training_data.shape}\n")
        f.write(f"Unique authors in training: {training_data['author'].nunique()}\n\n")
        
        # Calculate author performance
        training_data['performance'] = np.sign(training_data['expected_return']) * training_data['actual_return']
        author_stats = training_data.groupby('author').agg({
            'performance': ['sum', 'mean', 'count']
        }).reset_index()
        author_stats.columns = ['author', 'total_perf', 'mean_perf', 'num_predictions']
        
        # Select top authors
        n_select = max(1, int(np.ceil(len(author_stats) * k_percent / 100)))
        best_authors = author_stats.nlargest(n_select, 'total_perf')
        
        f.write("TOP SELECTED AUTHORS:\n")
        f.write(best_authors.to_string() + "\n\n")
        
        # 3. December Analysis
        f.write("DECEMBER TRADING ANALYSIS\n")
        f.write("-"*30 + "\n")
        
        month_df = df_test[
            (df_test['date'] >= target_date) & 
            (df_test['date'] <= target_date + pd.offsets.MonthEnd(1)) & 
            (df_test['author'].isin(best_authors['author']))
        ].copy()
        
        # Daily analysis
        corpus_fraction = 0.8
        initial_corpus = 1.0
        corpus_value = initial_corpus

        for date, day_data in month_df.groupby('date'):
            f.write(f"\nDATE: {date.strftime('%Y-%m-%d')}\n")
            f.write(f"Starting corpus: {corpus_value:.6f}\n")
            num_predictions = len(day_data)
            f.write(f"Number of trades: {num_predictions}\n\n")

            # Calculate allocation for this day
            base_allocation = (corpus_fraction * corpus_value) / num_predictions
            max_allocation = 0.25 * initial_corpus
            allocation_per_trade = min(base_allocation, max_allocation)
        
            daily_profit_loss = 0
            
            for idx, trade in day_data.iterrows():
                f.write(f"Trade {idx}:\n")
                f.write(f"Author: {trade['author']}\n")
                f.write(f"Expected Return: {trade['expected_return']:.6f}\n")
                f.write(f"Actual Return: {trade['actual_return']:.6f}\n")
                f.write(f"Sign match: {np.sign(trade['expected_return']) == np.sign(trade['actual_return'])}\n")
                
                # Calculate trade return
                signed_direction = np.sign(trade['expected_return'])
                trade_return = (signed_direction * trade['actual_return'] - 0.0004) * allocation_per_trade

                f.write(f"Trade calculation:\n")
                f.write(f"  Allocation per trade: {allocation_per_trade:.6f}\n")
                f.write(f"  Signed direction: {signed_direction}\n")
                f.write(f"  Trade return: {trade_return:.6f}\n")
                
                daily_profit_loss += trade_return
            
            # Update corpus with daily profit/loss
            old_corpus = corpus_value
            corpus_value += daily_profit_loss
            
            f.write(f"\nDaily Summary:\n")
            f.write(f"  Total profit/loss: {daily_profit_loss:.6f}\n")
            f.write(f"  Old corpus: {old_corpus:.6f}\n")
            f.write(f"  New corpus: {corpus_value:.6f}\n")
            f.write(f"  Daily return: {(daily_profit_loss/old_corpus)*100:.4f}%\n")
            f.write("-"*50 + "\n")

        # Calculate metrics for comparison
        month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
        mean_performance = month_df['performance'].mean()
        num_predictions = len(month_df)
        combined_metric = (mean_performance - 0.0004) * num_predictions
        corpus_return = (corpus_value - initial_corpus) / initial_corpus
        
        f.write("\nCOMPARISON OF METRICS\n")
        f.write("-"*30 + "\n")
        f.write(f"Traditional Metrics:\n")
        f.write(f"  Mean Performance: {mean_performance:.6f}\n")
        f.write(f"  Number of Predictions: {num_predictions}\n")
        f.write(f"  Combined Metric: {combined_metric:.6f}\n\n")
        f.write(f"Corpus Metrics:\n")
        f.write(f"  Initial Corpus: {initial_corpus:.6f}\n")
        f.write(f"  Final Corpus: {corpus_value:.6f}\n")
        f.write(f"  Corpus Return: {corpus_return*100:.4f}%\n")

        return {
            'k_percent': k_percent,
            'mean_performance': mean_performance,
            'num_predictions': num_predictions,
            'combined_metric': combined_metric,
            'corpus_return': corpus_return
        }

def plot_comparison(results_df):
    """Plot comparison of metrics for different k values."""
    plt.figure(figsize=(15, 10))
    
    # Plot Combined Metric
    plt.subplot(2, 1, 1)
    plt.bar(results_df['k_percent'].astype(str), results_df['combined_metric'])
    plt.title('Combined Metric by K%')
    plt.xlabel('K Percent')
    plt.ylabel('Combined Metric')
    plt.grid(True)
    
    # Plot Number of Predictions
    plt.subplot(2, 1, 2)
    plt.bar(results_df['k_percent'].astype(str), results_df['num_predictions'])
    plt.title('Number of Monthly Predictions by K%')
    plt.xlabel('K Percent')
    plt.ylabel('Number of Predictions')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('december_2021_comparison.png')
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
    
    print("Starting debug analysis for December 2021...")
    
    # Debug specific k values
    k_values = [15.0, 25.0]
    results = []
    
    for k in k_values:
        print(f"\nAnalyzing k={k}%...")
        debug_results = debug_december_2021(df_d1, df_d1, k)
        results.append(debug_results)
        print(f"Debug output saved to debug_outputs/detailed_debug_k{k}.txt")
    
    # Create comparison DataFrame and plot
    results_df = pd.DataFrame(results)
    print("\nComparison of metrics across k values:")
    print(results_df.to_string(float_format=lambda x: '{:.6f}'.format(x)))
    
    plot_comparison(results_df)
    print("\nComparison plots saved as 'december_2021_comparison.png'")