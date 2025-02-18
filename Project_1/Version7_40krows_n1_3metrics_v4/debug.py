from datetime import timedelta
import pandas as pd
import numpy as np
import sys
import os
from io import StringIO
import matplotlib.pyplot as plt
from Calculate_Performance_3metrics import get_best_authors, calculate_monthly_performance

def plot_daily_metrics(daily_results_df, month_df, k_percent, target_date, output_dir='debug_outputs'):
    """Create comprehensive plots for daily metrics."""
    plt.figure(figsize=(20, 15))
    
    # 1. Trading Activity
    plt.subplot(3, 2, 1)
    trades_per_day = month_df.groupby('date').size()
    plt.bar(trades_per_day.index, trades_per_day.values)
    plt.title(f'Daily Trading Volume (K={k_percent}%)')
    plt.xlabel('Date')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 2. Daily Performance Distribution
    plt.subplot(3, 2, 2)
    month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
    plt.hist(month_df['performance'], bins=30)
    plt.title('Distribution of Trade Performance')
    plt.xlabel('Performance')
    plt.ylabel('Frequency')
    plt.grid(True)

    # 3. Performance by Author
    plt.subplot(3, 2, 3)
    author_perf = month_df.groupby('author')['performance'].mean().sort_values()
    plt.bar(range(len(author_perf)), author_perf)
    plt.title('Average Performance by Author')
    plt.xlabel('Author Index')
    plt.ylabel('Mean Performance')
    plt.grid(True)

    # 4. Expected vs Actual Returns
    plt.subplot(3, 2, 4)
    plt.scatter(month_df['expected_return'], month_df['actual_return'], alpha=0.5)
    plt.title('Expected vs Actual Returns')
    plt.xlabel('Expected Return')
    plt.ylabel('Actual Return')
    plt.grid(True)
    
    # Add correlation coefficient
    corr = month_df['expected_return'].corr(month_df['actual_return'])
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=plt.gca().transAxes)

    # 5. Daily Success Rate
    plt.subplot(3, 2, 5)
    daily_success = month_df.groupby('date').apply(
        lambda x: (np.sign(x['expected_return']) == np.sign(x['actual_return'])).mean()
    )
    plt.plot(daily_success.index, daily_success.values, marker='o')
    plt.title('Daily Prediction Success Rate')
    plt.xlabel('Date')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 6. Cumulative Performance
    plt.subplot(3, 2, 6)
    daily_perf = month_df.groupby('date')['performance'].mean()
    cumulative_perf = daily_perf.cumsum()
    plt.plot(cumulative_perf.index, cumulative_perf.values, marker='o')
    plt.title('Cumulative Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Performance')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f'metrics_plot_{target_date.strftime("%Y_%m")}_k{k_percent}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

def debug_month(df_train, df_test, k_percent, target_date, output_dir='debug_outputs'):
    """Debug analysis for a specific month and K% and save to file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f'corpus_calculation_k{k_percent}.txt')
    
    # Create a string buffer to capture all output
    output_buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    try:
        print(f"\nCorpus Calculation Analysis for K = {k_percent}%")
        print("=" * 50)
        
        # Get month data
        month_start = target_date.replace(day=1)
        month_end = (month_start + pd.offsets.MonthEnd(1))
        
        # Get best authors
        training_end = target_date - timedelta(days=1)
        training_start = training_end - pd.DateOffset(months=14)
        best_authors = get_best_authors(df_train, training_start, training_end, k_percent)
        
        month_df = df_test[
            (df_test['date'] >= month_start) & 
            (df_test['date'] <= month_end) & 
            (df_test['author'].isin(best_authors['author']))
        ].copy()
        
        print(f"\nInitial Setup:")
        print(f"Month: {month_start.strftime('%B %Y')}")
        print(f"Number of selected authors: {len(best_authors)}")
        print(f"Total predictions in month: {len(month_df)}")
        
        # Corpus calculation details
        print("\nDaily Corpus Calculation Details:")
        print("=" * 50)
        
        corpus_value = 1.0  # Starting corpus value
        daily_corpus_values = []
        
        for date, day_data in month_df.groupby('date'):
            print(f"\nDate: {date.strftime('%Y-%m-%d')}")
            print(f"Starting corpus value: {corpus_value:.6f}")
            
            print("\nTrades for this day:")
            print("-" * 30)
            
            for idx, trade in day_data.iterrows():
                print(f"\nTrade {idx}:")
                print(f"Author: {trade['author']}")
                print(f"Expected Return: {trade['expected_return']:.6f}")
                print(f"Actual Return: {trade['actual_return']:.6f}")
                
                # Update corpus
                trade_impact = trade['actual_return']
                corpus_value *= (1 + trade_impact)
                print(f"Corpus after trade: {corpus_value:.6f}")
            
            daily_corpus_values.append({
                'date': date,
                'corpus_value': corpus_value,
                'num_trades': len(day_data),
                'daily_return': day_data['actual_return'].mean()
            })
            
            print(f"\nEnd of day corpus value: {corpus_value:.6f}")
        
        print("\nFinal Corpus Analysis:")
        print("=" * 50)
        print(f"Starting corpus: 1.0")
        print(f"Final corpus: {corpus_value:.6f}")
        print(f"Total return: {(corpus_value - 1.0):.6f} ({(corpus_value - 1.0)*100:.2f}%)")
        
        # Convert to DataFrame for additional analysis
        daily_summary = pd.DataFrame(daily_corpus_values)
        
        print("\nDaily Summary Statistics:")
        print(daily_summary.to_string())
        
    finally:
        # Restore stdout and save output
        sys.stdout = original_stdout
        
        with open(output_file, 'w') as f:
            f.write(output_buffer.getvalue())
        
        print(f"Corpus calculation details saved to {output_file}")
    
    return daily_summary

if __name__ == "__main__":
    # Load your data
    file_path_d5 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2.csv'
    file_path_d1 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv'
    
    df_d5 = pd.read_csv(file_path_d5).dropna(subset=['expected_return', 'actual_return'])
    df_d1 = pd.read_csv(file_path_d1).dropna(subset=['expected_return', 'actual_return'])
    
    # Convert dates to datetime
    df_d1['date'] = pd.to_datetime(df_d1['date'])
    df_d5['date'] = pd.to_datetime(df_d5['date'])
    
    print(f"Data range: {df_d1['date'].min()} to {df_d1['date'].max()}")
    print(f"Total records: {len(df_d1)}")
    
    # Debug specific months
    months_to_debug = [
        pd.to_datetime('2021-12-01')
    ]
    
    k_values = [25.0]
    
    for target_date in months_to_debug:
        print(f"\nDebugging month: {target_date.strftime('%Y-%m')}")
        for k in k_values:
            print(f"Processing k={k}%...")
            monthly_perf = debug_month(df_d1, df_d1, k, target_date)
