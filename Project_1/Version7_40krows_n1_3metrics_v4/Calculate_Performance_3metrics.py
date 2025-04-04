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
    author_performance = period_df.groupby('author').agg({
                    'performance': ['sum', 'mean', 'count']
                }).reset_index()
    author_performance.columns = ['author', 'total_perf', 'mean_perf', 'num_predictions']
    
    # Select top K% authors
    n_select = max(1, int(np.ceil(len(author_performance) * k_percent / 100)))
    return author_performance.nlargest(n_select, 'total_perf')

def calculate_monthly_performance(df, best_authors_df, target_month):
    """Calculate performance metrics for a specific month."""
    month_start = target_month.replace(day=1)
    month_end = (month_start + pd.offsets.MonthEnd(1))
    
    # Filter data
    month_df = df[
        (df['date'] >= month_start) & 
        (df['date'] <= month_end) & 
        (df['author'].isin(best_authors_df['author']))
    ].copy()
    
    if month_df.empty:
        return pd.DataFrame()
    
    # Original performance calculation
    month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
    
    # Corpus return calculation
    corpus_fraction = 0.8
    initial_corpus = 1.0
    corpus_value = initial_corpus

    # Process day by day
    daily_results = []
    for date, day_data in month_df.groupby('date'):
        num_predictions = len(day_data)
        base_allocation = corpus_fraction * corpus_value / num_predictions
        max_allocation = 0.25 * initial_corpus
        allocation_per_trade = min(base_allocation, max_allocation)

        daily_profit_loss = 0            
        for _, trade in day_data.iterrows():
            signed_direction = np.sign(trade['expected_return'])
            trade_return = (signed_direction * trade['actual_return'] - 0.0004) * allocation_per_trade
            daily_profit_loss += trade_return
                    
        corpus_value += daily_profit_loss

        daily_results.append({
                    'date': date, 
                    'corpus_value': corpus_value, 
                    'num_trades': num_predictions,
                    'allocation_per_trade': allocation_per_trade,
                    'daily_return': daily_profit_loss/corpus_value if corpus_value != 0 else 0
                })    
        
    daily_df = pd.DataFrame(daily_results)
    corpus_return = (corpus_value - initial_corpus) / initial_corpus
    # print(f"corpus_return for {month_start}: {corpus_return*100}%")
    # Combine both metrics in the summary
    performance_summary = pd.DataFrame({
        'author': month_df['author'].unique(),
        'mean_performance': month_df.groupby('author')['performance'].mean().values,  # Add .values
        'count': month_df.groupby('author')['performance'].count().values,  # Add .values
        'month': [month_start] * len(month_df['author'].unique()),
        'corpus_return': [corpus_return] * len(month_df['author'].unique())
    })
    # print(f"performance_summary{performance_summary}")
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
        'combined_metric': ('Combined Metric by K%', 'Combined Metric', 'green'),
        'corpus_return': ('Average Monthly Corpus Return by K%', 'Corpus Return', 'red')
    }
    
    plt.figure(figsize=(20, 15))
    
    for idx, (metric, (title, ylabel, color)) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, idx)
        
        if metric == 'mean_performance':
            y_data = results_df['mean_of_mean_performance']
        elif metric == 'monthly_predictions':
            y_data = results_df['number_of_monthly_predictions']
        elif metric == 'combined_metric':
            y_data = results_df['combined_metric']
        else:  # corpus_return
            y_data = results_df['mean_monthly_corpus_return']
            
        plt.plot(results_df['k_percent'], y_data, color=color, marker='o')
        plt.title(title)
        plt.xlabel('Top K Percent')
        plt.ylabel(ylabel)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
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
    # k_percentages = [5.0, 10.0, 15.0, 25.0]
    iteration_results = []
    
    for k in tqdm(k_percentages, desc="Testing K percentages"):
        # print(f"\n{k}%\n")
        results_df = run_rolling_analysis(df_train, df_test, start_date, end_date, k, lookback_period=14)

        if results_df.empty:
            iteration_results.append({
                'mean_of_mean_performance': np.nan,
                'number_of_monthly_predictions': 0,
                'combined_metric': np.nan,
                'mean_monthly_corpus_return': np.nan,
                'k_percent': k
            })
            continue

        # if not results_df.empty:
        #     # Original metrics
        #     metrics = {
        #         'mean_of_mean_performance': results_df['mean_performance'].mean(),
        #         'number_of_monthly_predictions': results_df['count'].sum()/num_months,
        #     }
        #     metrics['combined_metric'] = (metrics['mean_of_mean_performance'] - 0.0004) * \
        #                                 metrics['number_of_monthly_predictions']
            
        # ————————————————————————————————————————————————
        # 1) Original Code 1 metrics
        # ————————————————————————————————————————————————
        mean_of_mean_perf = results_df['mean_performance'].mean()
        number_of_monthly_preds = results_df['count'].sum() / num_months
        combined_metric_val = (mean_of_mean_perf - 0.0004) * number_of_monthly_preds
            
        # ————————————————————————————————————————————————
        # 2) Convert/Ensure 'month' is datetime & add 'year'
        # ————————————————————————————————————————————————
        results_df['month'] = pd.to_datetime(results_df['month'])
        results_df['year'] = results_df['month'].dt.year
        
        # ————————————————————————————————————————————————
        # 3) One monthly corpus_return per calendar month
        #    by grouping or dropping duplicates
        # ————————————————————————————————————————————————
        # Option A: If each (month, author) has the same corpus_return,
        #           just drop duplicates to keep unique monthly values:
        # monthly_unique = results_df.drop_duplicates(subset=['month'])

        # Option B: If different authors (in the same month) might have
        #           slightly different 'corpus_return' fields, group by month:
        monthly_unique = (
            results_df.groupby('month', as_index=False)['corpus_return'].mean()
        )

        # ————————————————————————————————————————————————
        # 4) Compute Yearly Averages & Print
        # ————————————————————————————————————————————————
        # Merge monthly_unique back with the year info if needed:
        monthly_unique['year'] = monthly_unique['month'].dt.year
        
        years_in_data = sorted(monthly_unique['year'].unique())
        for year in years_in_data:
            year_data = monthly_unique[monthly_unique['year'] == year]
            avg_year_corpus_return = year_data['corpus_return'].mean() * 100.0
            # print(f"Year {year} - Average Monthly Corpus Return: {avg_year_corpus_return:.4f}%")
            

        # ————————————————————————————————————————————————
        # 5) Overall Average (2018–2022)
        # ————————————————————————————————————————————————
        overall_corpus_return = monthly_unique['corpus_return'].mean() * 100.0
        # print("OVERALL SUMMARY (2018–2022)")
        # print(f"Average Monthly Corpus Return: {overall_corpus_return:.4f}%")
        # print("="*40)

        # ————————————————————————————————————————————————
        # 6) Store Final Values for This k
        # ————————————————————————————————————————————————
        iteration_results.append({
            'mean_of_mean_performance': mean_of_mean_perf,
            'number_of_monthly_predictions': number_of_monthly_preds,
            'combined_metric': combined_metric_val,
            'mean_monthly_corpus_return': overall_corpus_return,
            'k_percent': k
        })

            # # New corpus return 
            # filter_2021_12 = results_df[results_df['month'] == pd.to_datetime("2021-12-01")]
            # print(filter_2021_12)            
            # results_df['month'] = pd.to_datetime(results_df['month'])
            
            # # Extract the "year" from each monthly row
            # results_df['year'] = results_df['month'].dt.year

            # # Print year-end summary for corpus return
            # years_in_data = sorted(results_df['year'].unique())
            
            # for year in years_in_data:
            #     year_data = results_df[results_df['year'] == year]
            #     # print(f"year_data{year_data}")
            #     avg_year_corpus_return = (year_data['corpus_return'].unique().mean()) * 100
            #     print(f"Year {year} - Average Monthly Corpus Return: {avg_year_corpus_return:.4f}%")
                
            # metrics['mean_monthly_corpus_return'] = results_df['corpus_return'].mean().unique() * 100
            # print(" /n")
            # print(f"overall : {metrics['mean_monthly_corpus_return']}")
            # metrics['k_percent'] = k
            # iteration_results.append(metrics)
    
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