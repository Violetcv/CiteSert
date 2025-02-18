from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def get_best_authors(df, start_date, end_date, k_percent):
    df['date'] = pd.to_datetime(df['date'])
    period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    if period_df.empty:
        return pd.DataFrame(columns=['author', 'total_performance'])
    
    period_df['performance'] = np.sign(period_df['expected_return']) * period_df['actual_return']
    author_performance = (period_df.groupby('author')
                         .agg(
                             mean_performance=('performance', 'sum'),
                             number_of_predictions=('performance', 'count')
                         )
                         .reset_index())
    
    n = len(author_performance)
    n_select = max(1, int(np.ceil(n * k_percent / 100)))
    return author_performance.nlargest(n_select, 'mean_performance')

def calculate_monthly_performance(df, best_authors_df, target_month, corpus_fraction):
    month_start = target_month.replace(day=1)
    month_end = (month_start + pd.offsets.MonthEnd(1))
    
    # Filter data and calculate performance
    month_df = df[
        (df['date'] >= month_start) & 
        (df['date'] <= month_end) & 
        (df['author'].isin(best_authors_df['author']))
    ].copy()
    
    month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
    
    # Add this: Calculate mean performance per author (similar to code 2)
    author_performance = (month_df.groupby('author')['performance']
                         .agg(author_mean_performance='mean')
                         .reset_index())
    mean_author_performance = author_performance['author_mean_performance'].mean()

    all_authors_mean_perf = month_df['performance'].mean()

    
    if month_df.empty:
        return pd.DataFrame()
    
    corpus_D = 1.0
    daily_returns = []

    for date, day_data in month_df.groupby('date'):
        N = len(day_data)
        if N == 0:
            continue
        
        allocation_per_analyst = (corpus_D * corpus_fraction) / N
        day_data['trade_return'] = (np.sign(day_data['expected_return']) * 
                                  day_data['actual_return'] - 0.0004) * allocation_per_analyst
        
        daily_return = day_data['trade_return'].sum()
        corpus_D += daily_return
        
        daily_returns.append({
            'date': date,
            'daily_return': daily_return,
            'num_trades': N,
            'mean_performance': day_data['performance'].mean()
        })

    if daily_returns:
        daily_returns_df = pd.DataFrame(daily_returns)
        return pd.DataFrame({
            'month': [month_start],
            'final_corpus_value': [corpus_D],
            'total_return': [corpus_D - 1.0],
            'num_trades': [daily_returns_df['num_trades'].sum()],
            'all_authors_mean_perf': [all_authors_mean_perf],
            'mean_performance': [mean_author_performance],  # Changed to use author-level mean
            'mean_daily_performance': [daily_returns_df['mean_performance'].mean()]  # Keep original as well
            })
    
    return pd.DataFrame()

def run_rolling_analysis(df_train, df_test, start_date, end_date, k_percent, lookback_period, corpus_fraction):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_monthly_performance = []
    current_date = start_date

    while current_date <= end_date:
        training_end = current_date - timedelta(days=1)
        full_months = int(lookback_period)
        partial_month_days = int((lookback_period % 1) * 30)
        training_start = (training_end - pd.DateOffset(months=full_months) - 
                         pd.DateOffset(days=partial_month_days))
        
        best_authors = get_best_authors(df_train, training_start, training_end, k_percent)
        
        if not best_authors.empty:
            monthly_perf = calculate_monthly_performance(df_test, best_authors, current_date, corpus_fraction)
            if not monthly_perf.empty:
                all_monthly_performance.append(monthly_perf)
        
        current_date = current_date + pd.DateOffset(months=1)

    return pd.concat(all_monthly_performance, ignore_index=True) if all_monthly_performance else pd.DataFrame()

def plot_all_metrics(results_df):
    os.makedirs('performance_plots', exist_ok=True)
    
    # Individual metric plots
    # 1. Mean of Mean Performance
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k_percent'], results_df['mean_of_mean_performance'], 
             color='blue', marker='o')
    plt.title('Mean of Mean Performance by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Mean of Mean Performance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_plots/mean_performance.png')
    plt.close()

    # 2. Average Monthly Returns
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k_percent'], results_df['avg_monthly_return'], 
             color='green', marker='s')
    plt.title('Average Monthly Returns by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Average Monthly Return')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_plots/monthly_returns.png')
    plt.close()

    # 3. Combined Metric
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k_percent'], results_df['combined_metric'], 
             color='red', marker='^')
    plt.title('Combined Metric by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Combined Metric')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_plots/combined_metric.png')
    plt.close()

    # 4. Monthly Predictions
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k_percent'], results_df['number_of_predictions'], 
             color='purple', marker='d')
    plt.title('Number of Monthly Predictions by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Average Monthly Predictions')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_plots/monthly_predictions.png')
    plt.close()


    # Plot 4: All three metrics together
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['k_percent'], results_df['mean_of_mean_performance'], 
             color='blue', marker='o', label='Mean of Mean Performance')
    plt.plot(results_df['k_percent'], results_df['avg_monthly_return'], 
             color='green', marker='s', label='Avg Monthly Returns')
    plt.plot(results_df['k_percent'], results_df['combined_metric'], 
             color='red', marker='^', label='Combined Metric'),
    # plt.plot(results_df['k_percent'], results_df['number_of_predictions'], 
    #          color='purple', marker='d', label='Monthly Predictions')
    
    plt.title('All Performance Metrics by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('performance_plots/all_metrics.png')
    plt.close()
    
    # Plot 5: Returns vs Combined
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k_percent'], results_df['avg_monthly_return'], 
             color='green', marker='s', label='Avg Monthly Returns')
    plt.plot(results_df['k_percent'], results_df['combined_metric'], 
             color='red', marker='^', label='Combined Metric')
    
    plt.title('Returns vs Combined Metric')
    plt.xlabel('Top K Percent')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('performance_plots/returns_vs_combined.png')
    plt.close()

def main(df_train, df_test):
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')
    df_test['date'] = pd.to_datetime(df_test['date'], errors='coerce')

    start_date = '2018-01-01'
    end_date = '2022-12-31'
    num_months = (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year) * 12 + \
             (pd.to_datetime(end_date).month - pd.to_datetime(start_date).month) + 1

    
    k_percentages = np.arange(1, 25.5, 0.5).tolist()
    iteration_results = []
    
    for k in tqdm(k_percentages, desc="Testing K percentages"):
        results_df = run_rolling_analysis(
            df_train,
            df_test,
            start_date=start_date,
            end_date=end_date,
            lookback_period=14,
            k_percent=k,
            corpus_fraction=0.8
        )
        
        # and 'total_return' in results_df.columns
        if not results_df.empty:
            avg_monthly_return = results_df['total_return'].mean()
            mean_of_mean_perf = results_df['mean_performance'].mean()
            number_of_predictions = results_df['num_trades'].sum() / num_months
            combined_metric = (mean_of_mean_perf - 0.0004) * number_of_predictions
            
            iteration_results.append({
                'k_percent': k,
                'avg_monthly_return': avg_monthly_return,
                'mean_of_mean_performance': mean_of_mean_perf,
                'number_of_predictions': number_of_predictions,
                'combined_metric': combined_metric,
                'final_corpus_value': results_df['final_corpus_value'].iloc[-1],
                'total_trades': results_df['num_trades'].sum(),
                'num_months': len(results_df)
            })
    
    results_df = pd.DataFrame(iteration_results)
    
    # Save results
    results_df.to_csv("./input_data/performance_results.csv", index=False)
    
    # Create plots
    plot_all_metrics(results_df)
    
    return results_df

# Load and prepare data
file_path_d1 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv'
df_d1 = pd.read_csv(file_path_d1)
df_d1 = df_d1.dropna(subset=['expected_return', 'actual_return'])

# Run analysis
results = main(df_d1, df_d1)