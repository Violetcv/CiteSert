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

    # Calculate sign-adjusted performance
    month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']

    # Initialize corpus
    corpus_fraction = 0.8
    initial_corpus = 1.0
    corpus_value = initial_corpus

    # Day-by-day loop
    for date_key, day_data in month_df.groupby('date'):
        num_trades = len(day_data)
        # Daily allocation
        base_allocation = (corpus_fraction * corpus_value) / num_trades
        max_allocation = 0.25 * initial_corpus
        allocation_per_trade = min(base_allocation, max_allocation)

        daily_pl = 0.0
        for _, trade in day_data.iterrows():
            signed_direction = np.sign(trade['expected_return'])
            trade_return = (signed_direction * trade['actual_return'] - 0.0004)
            daily_pl += (trade_return * allocation_per_trade)

        corpus_value += daily_pl

    # Final monthly corpus return
    corpus_return = (corpus_value - initial_corpus) / initial_corpus

    # Build a summary DataFrame with one row per author in this month
    # (All authors share the same final corpus_return for the month.)
    month_summary = pd.DataFrame({
        'author': month_df['author'].unique(),
        'mean_performance': month_df.groupby('author')['performance'].mean().values,
        'count': month_df.groupby('author')['performance'].count().values,
        'month': [month_start] * month_df['author'].nunique(),
        'corpus_return': [corpus_return] * month_df['author'].nunique()
    })
    # print(month_summary)
    return month_summary

def run_rolling_analysis(df_train, df_test, start_date, end_date, k_percent, lookback_period=14):
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

def plot_performance_metrics(final_df):
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
    plt.savefig('performance_metrics.png')
    plt.close()


def main(df_train, df_test):
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
    start_date = '2021-01-01'
    end_date   = '2022-12-31'
    num_months = (
        (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year)*12
        + (pd.to_datetime(end_date).month - pd.to_datetime(start_date).month) + 1
    )

    # k_percent values
    # k_percentages = np.arange(1, 25.5, 0.5)
    k_percentages = [15.5]

    # Container for final aggregated results
    iteration_results = []

    print("Running analyses for each K%...")

    for k in tqdm(k_percentages, desc="Sweeping through K%"):
        month_author_df = run_rolling_analysis(df_train, df_test, start_date, end_date, k, lookback_period=14)

        # If no monthly data, store NaNs
        if month_author_df.empty:
            iteration_results.append({
                'mean_of_mean_performance': np.nan,
                'number_of_monthly_predictions': 0,
                'combined_metric': np.nan,
                'mean_monthly_corpus_return': np.nan,
                'k_percent': k
            })
            continue

        month_author_df['month'] = pd.to_datetime(month_author_df['month'])

        # 3) Group by month (to get exactly one combined metric per month)
        #    a) monthly_mean = average of mean_performance across authors
        #    b) monthly_count = sum of 'count' across authors
        #    c) monthly_combined = (monthly_mean - 0.0004) * monthly_count
        grouped = month_author_df.groupby('month', as_index=False).agg({
            'mean_performance': 'mean',
            'count': 'sum',
            'corpus_return': 'mean'  # average corpus return among authors
        })
        grouped['monthly_combined'] = (grouped['mean_performance'] - 0.0004)*grouped['count']

        # 4) Average your monthly_combined across all months
        monthly_combined_metric = grouped['monthly_combined'].mean()

        # Mean of mean performance
        mean_of_mean_perf = month_author_df['mean_performance'].mean()

        # Number of predictions (summed across all months) / total # months
        monthly_predictions = grouped['count'].sum() / num_months

        # Group multiple authors in a single month to one monthly corpus return
        # by taking the average "corpus_return" among authors. 
        # Then average across all months for the final result in 2018-2022.
        month_avg = (
            month_author_df.groupby('month', as_index=False)['corpus_return'].mean()
        )
        mean_monthly_corpus_return = month_avg['corpus_return'].mean() * 100.0  # in %

        iteration_results.append({
            'mean_of_mean_performance': mean_of_mean_perf,
            'number_of_monthly_predictions': monthly_predictions,
            'combined_metric': monthly_combined_metric,
            'mean_monthly_corpus_return': mean_monthly_corpus_return,
            'k_percent': k
        })

    # Create final DataFrame
    final_results_df = pd.DataFrame(iteration_results)
    final_results_df.to_csv("k_percent_performance_results.csv", index=False)
    print("Results saved to 'k_percent_performance_results.csv'")

    # Plot
    plot_performance_metrics(final_results_df)
    print("Plot saved as 'performance_metrics.png'")

    return final_results_df
if __name__ == "__main__":
    # File paths (update as needed)
    file_path_d5 = r"C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2.csv"
    file_path_d1 = r"C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv"

    # Read data
    df_d5 = pd.read_csv(file_path_d5).dropna(subset=['expected_return', 'actual_return'])
    df_d1 = pd.read_csv(file_path_d1).dropna(subset=['expected_return', 'actual_return'])

    # Example usage: pass df_d1 as both train and test
    results_df = main(df_d1, df_d1)
    print("\nSample of final results:\n", results_df.head()) 
