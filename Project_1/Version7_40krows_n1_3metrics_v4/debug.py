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
    
    output_file = os.path.join(output_dir, f'debug_{target_date.strftime("%Y_%m")}_k{k_percent}.txt')
    
    # Create a string buffer to capture all output
    output_buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    try:
        # Run analysis
        training_end = target_date - timedelta(days=1)
        training_start = training_end - pd.DateOffset(months=14)
        
        print(f"\nAnalyzing K = {k_percent}%")
        print(f"Training period: {training_start} to {training_end}")
        print(f"Test month: {target_date}")
        
        # Get best authors
        best_authors = get_best_authors(df_train, training_start, training_end, k_percent)
        print("\nBest Authors Selected:")
        print(best_authors.to_string())

        # Get month data
        month_start = target_date.replace(day=1)
        month_end = (month_start + pd.offsets.MonthEnd(1))
        
        month_df = df_test[
            (df_test['date'] >= month_start) & 
            (df_test['date'] <= month_end) & 
            (df_test['author'].isin(best_authors['author']))
        ].copy()
        
        print(f"\nMonth Data Summary:")
        print(f"Total predictions: {len(month_df)}")
        print(f"Unique authors: {month_df['author'].nunique()}")
        print(f"Trading days: {month_df['date'].nunique()}")

        if month_df.empty:
            print("No data for this month")
            return pd.DataFrame()
        
        # Calculate and log daily statistics
        print("\nDaily Trading Statistics:")
        for date, day_data in month_df.groupby('date'):
            print(f"\nDate: {date}")
            print(f"Number of trades: {len(day_data)}")
            print(f"Unique authors: {day_data['author'].nunique()}")
            print("Performance metrics:")
            print(f"- Mean expected return: {day_data['expected_return'].mean():.6f}")
            print(f"- Mean actual return: {day_data['actual_return'].mean():.6f}")
            
            # Calculate success rate
            success_rate = (np.sign(day_data['expected_return']) == 
                          np.sign(day_data['actual_return'])).mean()
            print(f"- Prediction success rate: {success_rate:.2%}")
        
        # Calculate monthly performance
        print("\nCalculating Monthly Performance:")
        monthly_perf = calculate_monthly_performance(df_test, best_authors, target_date)
        
        if not monthly_perf.empty:
            # Create daily results for plotting
            print("\nMonthly Performance Summary:")
            print(monthly_perf.to_string())

            # Calculate additional statistics
            print("\nAggregate Statistics:")
            print(f"Mean performance: {monthly_perf['mean_performance'].mean():.6f}")
            print(f"Total predictions: {monthly_perf['count'].sum()}")
            print(f"Corpus return: {monthly_perf['corpus_return'].iloc[0]:.6f}")
            
            daily_results = []
            for date, day_data in month_df.groupby('date'):
                daily_results.append({
                    'date': date,
                    'num_trades': len(day_data),
                    'mean_performance': day_data['performance'].mean() if 'performance' in day_data else None
                })
            daily_results_df = pd.DataFrame(daily_results)
            
            # Generate plots
            plot_daily_metrics(daily_results_df, month_df, k_percent, target_date, output_dir)
            
    finally:
        # Restore stdout and save output
        sys.stdout = original_stdout
        
        with open(output_file, 'w') as f:
            f.write(output_buffer.getvalue())
        
        print(f"Debug output saved to {output_file}")
    
    return monthly_perf

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
    
    k_values = [1.0, 5.0, 15.0, 25.0]
    
    for target_date in months_to_debug:
        print(f"\nDebugging month: {target_date.strftime('%Y-%m')}")
        for k in k_values:
            print(f"Processing k={k}%...")
            monthly_perf = debug_month(df_d1, df_d1, k, target_date)