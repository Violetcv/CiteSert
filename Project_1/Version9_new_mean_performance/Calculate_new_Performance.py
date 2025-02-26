#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_best_authors(df, start_date, end_date, k_percent):
    """
    Select top K% authors from df between start_date and end_date,
    based on sign-adjusted return (performance).
    """
    period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    if period_df.empty:
        return pd.DataFrame(columns=['author'])


    # Calculate sign-adjusted return per row
    period_df['performance'] = np.sign(period_df['expected_return']) * period_df['actual_return']

    # Group by author
    author_performance = (
        period_df.groupby('author')['performance']
        .agg(['sum', 'mean', 'count'])
        .reset_index()
        .rename(columns={'sum': 'total_perf', 'mean': 'mean_perf', 'count': 'num_predictions'})
    )

    # Select top K% by total_perf
    n_select = max(1, int(np.ceil(len(author_performance) * k_percent / 100)))
    return author_performance.nlargest(n_select, 'total_perf')[['author']]

def calculate_monthly_performance(df, best_authors_df, target_month):
    """
    For a single month (target_month), restricted to best_authors_df:
    • Calculate mean_performance for each author.
    • Compute a day-by-day corpus return
    using an 80% corpus fraction and 0.0004 cost per trade.
    • Return one row per author with columns:
    ['author', 'mean_performance', 'count', 'month', 'corpus_return'].
    """
    # Identify month boundaries
    month_start = target_month.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(1)

    # Filter to this month + best authors
    month_df = df[
        (df['date'] >= month_start) &
        (df['date'] <= month_end) &
        (df['author'].isin(best_authors_df['author']))
    ].copy()

    if month_df.empty:
        return pd.DataFrame()

    # Compute trade-level performance (raw)
    month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
    total_trades = len(month_df)
    monthly_mean_performance = month_df['performance'].sum() / total_trades

    # Run day-by-day simulation to update corpus_value
    corpus_fraction = 0.8
    initial_corpus = 1.0
    corpus_value = initial_corpus

    # Loop on each day (grouped by date)
    for date_key, day_data in month_df.groupby('date'):
        num_trades = len(day_data)
        # Daily allocation per trade: fraction_of_corpus / (# trades), capped at 25% of initial corpus.
        base_allocation = (corpus_fraction * corpus_value) / num_trades
        max_allocation = 0.25 * initial_corpus
        allocation_per_trade = min(base_allocation, max_allocation)
        
        daily_pl = 0.0
        for _, trade in day_data.iterrows():
            signed_direction = np.sign(trade['expected_return'])
            # Calculate trade result: note the cost subtraction per trade.
            trade_return = (signed_direction * trade['actual_return'] - 0.0004)
            daily_pl += trade_return * allocation_per_trade
        corpus_value += daily_pl

    corpus_return = (corpus_value - initial_corpus) / initial_corpus

    # Combined metric based on the new performance calculation:
    # (monthly_mean_performance - 0.0004) multiplied by total trades.
    combined_metric = (monthly_mean_performance - 0.0004) * total_trades

    summary = pd.DataFrame({
        'month': [month_start],
        'mean_performance': [monthly_mean_performance],
        'count': [total_trades],
        'corpus_return': [corpus_return],
        'combined_metric': [combined_metric]
    })
    return summary

def run_rolling_analysis(df_train, df_test, start_date, end_date, k_percent, lookback_period):
    """
    1) From start_date to end_date, month-by-month:
    • Find best authors (top k%) from the preceding lookback_period in df_train.
    • Compute monthly performance metrics in df_test.
    2) Return a DataFrame with one row per (month, author).
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    monthly_results = []
    current = start_date

    while current <= end_date:
        train_end = current - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(months=lookback_period)

        # Choose best authors from df_train
        best_authors_df = get_best_authors(df_train, train_start, train_end, k_percent)
        if not best_authors_df.empty:
            # Calculate metrics for this month
            month_summary = calculate_monthly_performance(df_test, best_authors_df, current)
            if not month_summary.empty:
                monthly_results.append(month_summary)

        current += pd.DateOffset(months=1)

    if not monthly_results:
        return pd.DataFrame()  # empty

    return pd.concat(monthly_results, ignore_index=True)

def plot_performance_metrics(final_df,df_train, df_test, lookback_period):
    """
    Plot four metrics vs. k_percent, each metric in its own subplot.
    The columns are:
    • mean_of_mean_performance
    • number_of_monthly_predictions
    • combined_metric
    • mean_monthly_corpus_return
    Saved as performance_metrics.png.
    """

    # Sort by k_percent to ensure ascending order in the plots
    final_df = final_df.sort_values('k_percent').reset_index(drop=True)

    # Prepare the figure
    plt.figure(figsize=(18, 12))

    # 1) Mean of Mean Performance
    plt.subplot(2, 2, 1)
    plt.plot(final_df['k_percent'], final_df['mean_of_mean_performance'], marker='o', color='blue')
    plt.title('Mean of Mean Performance vs. K%')
    plt.xlabel('K Percent')
    plt.ylabel('Mean of Mean Performance')
    plt.grid(True)

    # 2) Number of Monthly Predictions
    plt.subplot(2, 2, 2)
    plt.plot(final_df['k_percent'], final_df['number_of_monthly_predictions'], marker='o', color='orange')
    plt.title('Monthly Predictions vs. K%')
    plt.xlabel('K Percent')
    plt.ylabel('Avg # Predictions per Month')
    plt.grid(True)

    # 3) Combined Metric
    plt.subplot(2, 2, 3)
    plt.plot(final_df['k_percent'], final_df['combined_metric'], marker='o', color='green')
    plt.title('Combined Metric vs. K%')
    plt.xlabel('K Percent')
    plt.ylabel('Combined Metric')
    plt.grid(True)

    # 4) Monthly Corpus Return
    plt.subplot(2, 2, 4)
    plt.plot(final_df['k_percent'], final_df['mean_monthly_corpus_return'], marker='o', color='red')
    plt.title('Average Monthly Corpus Return (%) vs. K%')
    plt.xlabel('K Percent')
    plt.ylabel('Corpus Return (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'performance_metrics_{lookback_period}.png')
    plt.close()


def main(df_train, df_test,start_date,end_date,lookback_period, k_percentages):
    """
    Main analysis:
    1) Run a rolling analysis for k in [1, 25.0 in 0.5 increments].
    2) Aggregate the results into 4 final metrics:
    (a) mean_of_mean_performance
    (b) number_of_monthly_predictions
    (c) combined_metric
    (d) mean_monthly_corpus_return
    3) Save final to CSV and plot performance_metrics.png
    """
    # Convert dates if needed
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])

    # Date boundaries
    

    num_months = (
        (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year)*12
        + (pd.to_datetime(end_date).month - pd.to_datetime(start_date).month) + 1
    )

    # Container for final aggregated results
    iteration_results = []

    print("Running analyses for each K%...")

    for k in tqdm(k_percentages, desc="Sweeping through K%"):
        monthly_df = run_rolling_analysis(df_train, df_test, start_date, end_date, k, lookback_period)

        # If no monthly data, store NaNs
        if monthly_df.empty:
            iteration_results.append({
                'mean_of_mean_performance': np.nan,
                'number_of_monthly_predictions': 0,
                'combined_metric': np.nan,
                'mean_monthly_corpus_return': np.nan,
                'k_percent': k
            })
            continue

         # For final aggregation we expect monthly_df has one row per month.
        overall_mean_perf = monthly_df['mean_performance'].mean()
        overall_count = monthly_df['count'].sum()
        overall_combined = monthly_df['combined_metric'].mean()
        overall_corpus_return = monthly_df['corpus_return'].mean() * 100.0  # percent
        
        # Total number of predictions = total trades over months / total months
        monthly_predictions = overall_count / num_months
        
        iteration_results.append({
            'mean_of_mean_performance': overall_mean_perf,
            'number_of_monthly_predictions': monthly_predictions,
            'combined_metric': overall_combined,
            'mean_monthly_corpus_return': overall_corpus_return,
            'k_percent': k
        })

    # Create final DataFrame
    final_results_df = pd.DataFrame(iteration_results)
    final_results_df.to_csv(f"k_percent_performance_results_{lookback_period}.csv", index=False)
    print(f"Results saved to 'k_percent_performance_results_{lookback_period}.csv'")

    # Plot
    plot_performance_metrics(final_results_df,df_train, df_test, lookback_period)
    print(f"Plot saved as 'performance_metrics_{lookback_period}.png'")

    return final_results_df
if __name__ == "__main__":
    # File paths (update as needed)
    file_path_d5 = r"C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2.csv"
    file_path_d1 = r"C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv"

    # Read data
    df_d5 = pd.read_csv(file_path_d5).dropna(subset=['expected_return', 'actual_return'])
    df_d1 = pd.read_csv(file_path_d1).dropna(subset=['expected_return', 'actual_return'])
    start_date = "2018-01-01"
    end_date = "2022-12-31"
    lookback_period_range = [3, 6, 9, 12, 15, 18, 21, 24]

    # k_percent values
    k_percentages = np.arange(1, 50.5, 0.5)
    # k_percentages = [15.5]

    for i in lookback_period_range:
        results_df = main(df_d5, df_d5, start_date, end_date, i, k_percentages)
        print(f"\nSample of final results for {i}:\n", results_df.head()) 
