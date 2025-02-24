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

def calculate_monthly_performance(df, best_authors_df, target_month, k_percent):
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

    daily_rows = []

    # Day-by-day loop
    for date_key, day_data in month_df.groupby('date'):
        day_start_corpus = corpus_value
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

            daily_rows.append({
            'k_percent': k_percent,
            'date': date_key,
            'author': trade['author'],
            'allocation_per_trade': allocation_per_trade,
            'signed_direction': signed_direction,
            'expected_return': trade['expected_return'],
            'actual_return': trade['actual_return'],
            'trade_return': trade_return,
            'day_corpus_start': day_start_corpus,  # corpus at the start of the day (same for all trades that day)
            # day_corpus_end will be filled after the entire day is computed
        })

        corpus_value += daily_pl

        # Now we can update day_corpus_end for each row of this day
        # because only after all trades do we know the final corpus
        for row_dict in daily_rows[-num_trades:]:
            row_dict['day_corpus_end'] = corpus_value

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

    daily_details = pd.DataFrame(daily_rows)
    # print(month_summary)

    return month_summary

def calculate_daily_trades_for_may2022(df, best_authors_df, target_month, k_percent):
    """
    Specialized function that calculates day-by-day trades ONLY for May 2022,
    returning a DataFrame with one row per trade.
    """
    month_start = target_month.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(1)

    # Filter data for that month + best authors
    month_df = df[
        (df['date'] >= month_start) &
        (df['date'] <= month_end) &
        (df['author'].isin(best_authors_df['author']))
    ].copy()

    if month_df.empty:
        return pd.DataFrame()

    month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']

    corpus_fraction = 0.8
    initial_corpus = 1.0
    corpus_value = initial_corpus

    daily_rows = []

    for date_key, day_data in month_df.groupby('date'):
        starting_corpus = corpus_value
        num_trades = len(day_data)
        base_alloc = (corpus_fraction * corpus_value) / num_trades
        max_alloc = 0.25 * initial_corpus
        alloc_per_trade = min(base_alloc, max_alloc)

        daily_pl = 0.0
        for _, trade in day_data.iterrows():
            signed_direction = np.sign(trade['expected_return'])
            trade_return_val = (signed_direction * trade['actual_return'] - 0.0004) * alloc_per_trade
            daily_pl += trade_return_val

            daily_rows.append({
                'k_percent': k_percent,
                'date': date_key,
                'author': trade['author'],
                'expected_return': trade['expected_return'],
                'actual_return': trade['actual_return'],
                'signed_direction': signed_direction,
                'allocation_per_trade': alloc_per_trade,
                'trade_return': trade_return_val,
                'corpus_before_day': starting_corpus  # corpus at start of day
                # corpus_after_day added after we compute daily_pl
            })
        corpus_value += daily_pl
        # Now fill in corpus_after_day for each row in this day
        for row_dict in daily_rows[-num_trades:]:
            row_dict['corpus_after_day'] = corpus_value

    return pd.DataFrame(daily_rows)

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
    daily_may_rows  = []

    current = start_date
    while current <= end_date:
        train_end = current - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(months=lookback_period)

        best_authors_df = get_best_authors(df_train, train_start, train_end, k_percent)
        if not best_authors_df.empty:
            # Always compute monthly summary
            month_summary = calculate_monthly_performance(df_test, best_authors_df, current, k_percent)
            if not month_summary.empty:
                monthly_results.append(month_summary)

            # If specifically 2022-05, compute daily trades
            if current.year == 2022 and current.month == 5:
                day_df = calculate_daily_trades_for_may2022(df_test, best_authors_df, current, k_percent)
                if not day_df.empty:
                    daily_may_rows.append(day_df)

        current += pd.DateOffset(months=1)

    monthly_df = pd.concat(monthly_results, ignore_index=True) if monthly_results else pd.DataFrame()
    
    daily_df   = pd.concat(daily_may_rows, ignore_index=True)  if daily_may_rows else pd.DataFrame()
    # print(daily_df)

    return monthly_df, daily_df

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
    start_date = '2018-01-01'
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
    all_monthly_frames = []
    all_daily_may_frames = []

    print("Running analyses for each K%...")

    for k in tqdm(k_percentages, desc="Sweeping through K%"):
        monthly_df, daily_may_df = run_rolling_analysis(df_train, df_test, start_date, end_date, k, lookback_period=14)

        if not monthly_df.empty:
            # Add month col as datetime
            monthly_df['month'] = pd.to_datetime(monthly_df['month'])
            print(f"monthly_df{monthly_df[54:58]}")
            # Group by month => single monthly row
            grouped = monthly_df.groupby('month', as_index=False).agg({
                'mean_performance': 'mean',
                'count': 'sum',
                'corpus_return': 'mean'
            })

            grouped['monthly_combined'] = (grouped['mean_performance'] - 0.0004) * grouped['count']
            grouped['k_percent'] = k
            print(f"grouped{grouped[16:17]}")

            all_monthly_frames.append(grouped)
            all_monthly_frames_df = pd.concat(all_monthly_frames, ignore_index=True)
            may_2022_df = all_monthly_frames_df[all_monthly_frames_df['month'] == '2022-05-01']
            print(may_2022_df)

            # Summarize for final results
            monthly_combined_metric = grouped['monthly_combined'].mean()
            mean_of_mean_perf       = monthly_df['mean_performance'].mean()
            monthly_predictions     = grouped['count'].sum() / num_months
            mean_monthly_corpus_ret = grouped['corpus_return'].mean() * 100.0

            iteration_results.append({
                'k_percent': k,
                'mean_of_mean_performance': mean_of_mean_perf,
                'number_of_monthly_predictions': monthly_predictions,
                'combined_metric': monthly_combined_metric,
                'mean_monthly_corpus_return': mean_monthly_corpus_ret
            })
        else:
            # No data => fill NaNs
            iteration_results.append({
                'k_percent': k,
                'mean_of_mean_performance': np.nan,
                'number_of_monthly_predictions': 0,
                'combined_metric': np.nan,
                'mean_monthly_corpus_return': np.nan
            })

         # For k = 15.5%, print and save detailed output to a txt file.
        # if abs(k - 15.5) < 1e-9:
        #     details = []
        #     details.append("===== Detailed Results for k = 15.5% =====\n")
        #     # Filter the grouped monthly summary for May 2022 (assuming month is May 1, 2022)
        #     may_mask = grouped['month'] == pd.Timestamp("2022-05-01")
        #     if not grouped[may_mask].empty:
        #         may_summary = grouped[may_mask]
        #         details.append("Monthly Summary for May 2022:\n")
        #         details.append(may_summary.to_string(index=False) + "\n")
        #         details.append("Aggregated Metrics for k = 15.5%:\n")
        #         details.append("Mean of Mean Performance: {:.6f}\n".format(mean_of_mean_perf))
        #         details.append("Combined Metric: {:.6f}\n".format(monthly_combined_metric))
        #         details.append("Total Monthly Predictions (avg per month): {:.2f}\n".format(monthly_predictions))
        #         details.append("Mean Monthly Corpus Return (%): {:.2f}\n".format(mean_monthly_corpus_ret))
        #     else:
        #         details.append("No monthly summary results for May 2022 available.\n")

        #     if not daily_may_df.empty:
        #         details.append("\nDaily Trades Details for May 2022:\n")
        #         details.append(daily_may_df.to_string(index=False) + "\n")
        #     else:
        #         details.append("No daily trades details for May 2022 available.\n")

        #     # Join all details into one string and write to a text file.
        #     detail_str = "\n".join(details)
        #     print(detail_str)  # Also print to standard output.
        #     with open("detailed_results_k15.5.txt", "w") as f:
        #         f.write(detail_str)

        # Save daily trades data if any
        if not daily_may_df.empty:
            all_daily_may_frames.append(daily_may_df)

    # Build final summary results
    final_df = pd.DataFrame(iteration_results)
    final_df.to_csv("k_percent_performance_results.csv", index=False)
    print("\nSaved main K% results to 'k_percent_performance_results.csv'")

    # Plot performance metrics
    plot_performance_metrics(final_df)

    # Optionally combine all monthly frames for debugging (you can remove this if not needed)
    if all_monthly_frames:
        big_monthly_df = pd.concat(all_monthly_frames, ignore_index=True)
        big_monthly_df.to_csv("debug_monthly_summaries.csv", index=False)
        print("Saved monthly summaries to 'debug_monthly_summaries.csv'.")

    # Combine and save all daily trades for May 2022
    if all_daily_may_frames:
        big_daily_may = pd.concat(all_daily_may_frames, ignore_index=True)
        big_daily_may.to_csv("daily_trade_calculations_2022_05.csv", index=False)
        print("Saved day-by-day trades for May 2022 to 'daily_trade_calculations_2022_05.csv'.")
    else:
        print("No daily trades found for May 2022.")

    return final_df

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
