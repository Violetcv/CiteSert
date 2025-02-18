from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_best_authors(df, start_date, end_date, k_percent):
    """Get top performing authors for a given period."""
    period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    if period_df.empty:
        return pd.DataFrame(columns=['author', 'total_performance'])
    
    # Calculate performance and aggregate by author
    period_df['performance'] = np.sign(period_df['expected_return']) * period_df['actual_return']
    author_performance = (period_df.groupby('author')
                         .agg(
                             total_performance=('performance', 'sum'),
                             number_of_rows=('performance', 'count')
                         )
                         .reset_index())
    
    # Select top K% authors
    n_select = max(1, int(np.ceil(len(author_performance) * k_percent / 100)))
    return author_performance.nlargest(n_select, 'total_performance')

def calculate_monthly_performance(df, best_authors_df, target_month):
    """Calculate performance metrics for a specific month."""
    month_start = target_month.replace(day=1)
    month_end = (month_start + pd.offsets.MonthEnd(1))
    
    # Filter data and calculate performance
    month_df = df[
        (df['date'] >= month_start) & 
        (df['date'] <= month_end) & 
        (df['author'].isin(best_authors_df['author']))
    ].copy()
    
    month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
    
    # Summarize performance
    performance_summary = (month_df.groupby('author')
                         ['performance']
                         .agg(mean_performance='mean', count='count')
                         .reset_index())
    performance_summary['month'] = month_start
    
    return performance_summary

def run_rolling_analysis(df_train, df_test, start_date, end_date, k_percent, lookback_period=12):
    """Run rolling analysis for performance evaluation."""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_monthly_performance = []
    current_date = start_date

    while current_date <= end_date:
        training_end = current_date - timedelta(days=1)
        training_start = training_end - pd.DateOffset(months=lookback_period)
        
        best_authors = get_best_authors(df_train, training_start, training_end, k_percent)
        
        if not best_authors.empty:
            monthly_perf = calculate_monthly_performance(df_test, best_authors, current_date)
            all_monthly_performance.append(monthly_perf)
            
        current_date += pd.DateOffset(months=1)

    return pd.concat(all_monthly_performance, ignore_index=True) if all_monthly_performance else pd.DataFrame()

def plot_performance_metrics(results_df):
    """Generate performance analysis plots."""
    metrics = {
        'mean_performance': ('Mean Performance by K%', 'Mean Performance', 'blue'),
        'monthly_predictions': ('Monthly Predictions by K%', 'Average Monthly Predictions', 'orange'),
        'combined_metric': ('Combined Metric by K%', 'Combined Metric', 'green')
    }
    
    for metric, (title, ylabel, color) in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['k_percent'], 
                results_df[f'mean_of_mean_performance' if metric == 'mean_performance' 
                          else 'number_of_monthly_predictions' if metric == 'monthly_predictions'
                          else 'combined_metric'], 
                color=color, marker='o')
        plt.title(title)
        plt.xlabel('Top K Percent')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{metric}_plot.png')
        plt.close()

def main(df_train, df_test):
    """Main analysis function."""
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    num_months = (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year) * 12 + \
                 (pd.to_datetime(end_date).month - pd.to_datetime(start_date).month) + 1
    
    k_percentages = np.arange(1, 25.5, 0.5)
    iteration_results = []
    
    for k in tqdm(k_percentages, desc="Testing K percentages"):
        results_df = run_rolling_analysis(df_train, df_test, start_date, end_date, k, lookback_period=14)
        
        if not results_df.empty:
            metrics = {
                'mean_of_mean_performance': results_df['mean_performance'].mean(),
                'number_of_monthly_predictions': results_df['count'].sum()/num_months,
            }
            metrics['combined_metric'] = (metrics['mean_of_mean_performance'] - 0.0004) * \
                                        metrics['number_of_monthly_predictions']
            
            metrics['k_percent'] = k
            iteration_results.append(metrics)
    
    results_df = pd.DataFrame(iteration_results)
    results_df.to_csv("./input_data/k_percent_performance_results.csv")
    
    return results_df

# Main execution
if __name__ == "__main__":
    file_path_d5 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2.csv'
    file_path_d1 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv'
    
    df_d5 = pd.read_csv(file_path_d5).dropna(subset=['expected_return', 'actual_return'])
    df_d1 = pd.read_csv(file_path_d1).dropna(subset=['expected_return', 'actual_return'])
    
    results_df = main(df_d1, df_d1)
    plot_performance_metrics(results_df)