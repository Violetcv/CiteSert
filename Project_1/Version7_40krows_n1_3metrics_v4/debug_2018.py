from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from io import StringIO
import sys

def debug_year_2018(df_train, df_test, k_percent, output_dir='debug_outputs'):
    """Detailed debugging function for entire year 2018."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, f'detailed_debug_2018_k{k_percent}.txt')
    monthly_results = []
    
    with open(log_file, 'w') as f:
        f.write(f"DETAILED DEBUGGING FOR K={k_percent}% - YEAR 2018\n")
        f.write("="*50 + "\n\n")
        
        # Analyze each month of 2018
        for month in range(1, 13):
            target_date = pd.to_datetime(f'2018-{month:02d}-01')
            training_end = target_date - timedelta(days=1)
            training_start = training_end - pd.DateOffset(months=14)
            
            f.write(f"\nANALYSIS FOR {target_date.strftime('%B %Y')}\n")
            f.write("="*50 + "\n")
            
            f.write(f"Training Period: {training_start} to {training_end}\n")
            f.write(f"Test Period: {target_date} to {target_date + pd.offsets.MonthEnd(1)}\n\n")
            
            # Author Selection Process
            training_data = df_train[
                (df_train['date'] >= training_start) & 
                (df_train['date'] <= training_end)
            ].copy()
            
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
            
            monthly_results.append({
                'month': target_date.strftime('%B'),
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
            f.write("-"*50 + "\n")
        
        # Year-end summary
        f.write("\nYEAR 2018 SUMMARY\n")
        f.write("="*50 + "\n")
        year_df = pd.DataFrame(monthly_results)
        
        f.write("\nMonthly Metrics:\n")
        f.write(year_df.to_string(float_format=lambda x: '{:.6f}'.format(x)))
        
        f.write("\n\nAnnual Averages:\n")
        f.write(f"Average Monthly Predictions: {year_df['num_predictions'].mean():.2f}\n")
        f.write(f"Average Mean Performance: {year_df['mean_performance'].mean():.6f}\n")
        f.write(f"Average Combined Metric: {year_df['combined_metric'].mean():.6f}\n")
        f.write(f"Average Monthly Corpus Return: {year_df['corpus_return'].mean()*100:.4f}%\n")
        
    return year_df

def plot_yearly_comparison(year_df, k_percent):
    """Plot monthly comparisons for 2018."""
    plt.figure(figsize=(15, 15))
    
    # Plot Combined Metric
    plt.subplot(3, 1, 1)
    plt.bar(year_df['month'], year_df['combined_metric'])
    plt.title(f'Monthly Combined Metric (K={k_percent}%)')
    plt.xlabel('Month')
    plt.ylabel('Combined Metric')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot Number of Predictions
    plt.subplot(3, 1, 2)
    plt.bar(year_df['month'], year_df['num_predictions'])
    plt.title('Monthly Number of Predictions')
    plt.xlabel('Month')
    plt.ylabel('Number of Predictions')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot Corpus Return
    plt.subplot(3, 1, 3)
    plt.bar(year_df['month'], year_df['corpus_return'] * 100)
    plt.title('Monthly Corpus Return (%)')
    plt.xlabel('Month')
    plt.ylabel('Return (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'year_2018_k{k_percent}_comparison.png')
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
    
    print("Starting analysis for year 2018...")
    
    # Analyze specific k values
    k_values = [15.0, 25.0]
    all_results = {}
    
    for k in k_values:
        print(f"\nAnalyzing k={k}%...")
        year_df = debug_year_2018(df_d1, df_d1, k)
        all_results[k] = year_df
        plot_yearly_comparison(year_df, k)
        print(f"Results saved to debug_outputs/detailed_debug_2018_k{k}.txt")
        print(f"Plots saved as year_2018_k{k}_comparison.png")
    
    print("\nAnalysis complete. Check the debug_outputs folder for detailed logs.")