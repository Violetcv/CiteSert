import os
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt


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


def calculate_monthly_performance(df, best_authors_df, target_month, cost_per_trade, corpus_fraction, max_allocation_fraction, day_offset):
    """
    For a single month (target_month), restricted to best_authors_df:
    • Calculate mean_performance for each author.
    • Simulate a day-by-day corpus return using corpus_fraction and cost_per_trade.{using an 80% corpus fraction and 0.0004 cost per trade.}
    • Use BDay(5) for the holding period.
    Additionally, while simulating the day-by-day process, it logs the minimum cash available (the corpus that is not allocated) for that month.
    
    • Returns a DataFrame with one row containing:
    ['month', 'mean_performance', 'count', 'corpus_return', 'combined_metric']
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

    # Compute trade-level performance
    month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
    total_trades = len(month_df)
    monthly_mean_performance = month_df['performance'].sum() / total_trades

    # Run day-by-day simulation to update corpus_value
    # corpus_fraction = 0.8
    initial_corpus = 1.0
    corpus_value = initial_corpus

    pending_returns = {} # Capital + P/L due back on a certain date
    allocated_funds = {} # Capital allocated (and locked) until return

    # We'll track the minimum available cash across all days in this month.
    # Available cash is defined as corpus_value minus total allocated funds.
    monthly_min_available = initial_corpus  # initial corpus is the maximum available

    # Loop on each day (grouped by date)
    for date_key, day_data in month_df.groupby('date'):
        # Calculate total allocated funds still in holding period
        total_allocated = sum(allocated_funds.values())

        # Adjust corpus for available fundc
        available_corpus = corpus_value - total_allocated

        if available_corpus < 0:
            # This can happen if you over-allocate in prior days.
            available_corpus = 0.0

        # Update monthly minimum available cash if current available is lower
        if available_corpus < monthly_min_available:
            monthly_min_available = available_corpus

        num_trades = len(day_data)
        
        # Daily allocation per trade: fraction_of_corpus / (# trades), capped at 25% of initial corpus.
        base_allocation = (corpus_fraction * available_corpus) / num_trades
        max_allocation = max_allocation_fraction  * initial_corpus
        allocation_per_trade = min(base_allocation, max_allocation)
        
        daily_pl = 0.0
        total_allocation_out_today = 0.0

        for _, trade in day_data.iterrows():
            signed_direction = np.sign(trade['expected_return'])

            # Calculate trade result: note the cost subtraction per trade.
            trade_return = (signed_direction * trade['actual_return'] - cost_per_trade)
            
            # The P/L for this single trade
            single_trade_pl = trade_return * allocation_per_trade
            daily_pl += single_trade_pl
            total_allocation_out_today += allocation_per_trade

        
        # The day we get the trade principal + PL back
        return_date = date_key + pd.offsets.BDay(day_offset)

        pending_returns.setdefault(return_date, 0.0)
        allocated_funds.setdefault(return_date, 0.0)

        # The amount we expect to add back on return_date is both the allocated principal AND the daily_pl for all trades
        pending_returns[return_date] += (daily_pl + total_allocation_out_today)

        # We also must mark that we have allocated out capital
        allocated_funds[return_date] += total_allocation_out_today

        # Now, see if any capital is returning for today's date_key
        # i.e., if we had previous trades that ended today
        if date_key in pending_returns:
            # The principal + P/L from prior trades
            corpus_value += pending_returns.pop(date_key) - allocated_funds.pop(date_key)
            # That means allocated_funds can also be freed
            allocated_funds.pop(date_key, None)

    corpus_return = (corpus_value - initial_corpus) / initial_corpus

    # Combined metric
    combined_metric = (monthly_mean_performance - cost_per_trade) * total_trades

    summary = pd.DataFrame({
        'month': [month_start],
        'mean_performance': [monthly_mean_performance],
        'count': [total_trades],
        'corpus_return': [corpus_return],
        'combined_metric': [combined_metric],
        'min_available_cash': [monthly_min_available]  # Save the minimum available cash (fraction)
    })
    return summary

def run_rolling_analysis(df_train, df_test, start_date, end_date, k_percent, lookback_period, cost_per_trade, corpus_fraction, max_allocation_fraction, day_offset):
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
            month_summary = calculate_monthly_performance(df_test, best_authors_df, current, cost_per_trade, corpus_fraction, max_allocation_fraction, day_offset)
            if not month_summary.empty:
                monthly_results.append(month_summary)

        current += pd.DateOffset(months=1)

    if not monthly_results:
        return pd.DataFrame()  # empty

    return pd.concat(monthly_results, ignore_index=True)

def plot_performance_metrics(final_df, lookback_period, max_alloc, output_dir):
    """
    Given final_df (aggregated results for each k_percent) with the following columns:
      - mean_of_mean_performance
      - number_of_monthly_predictions
      - combined_metric
      - mean_monthly_corpus_return
      - median_monthly_corpus_return
      - mode_monthly_corpus_return
      - std_monthly_corpus_return
      - max_monthly_corpus_return
      - min_monthly_corpus_return
      - avg_min_available_cash
      - k_percent
      
    This function creates a 3×4 grid of subplots (12 slots), and fills in 10 plots:
      1. Mean of Mean Performance vs. K%
      2. Number of Monthly Predictions vs. K%
      3. Combined Metric vs. K%
      4. Mean Monthly Corpus Return (%) vs. K%
      5. Median Monthly Corpus Return (%) vs. K%
      6. Mode Monthly Corpus Return (%) vs. K%
      7. STD of Monthly Corpus Return (%) vs. K%
      8. Mean/STD Ratio (%) vs. K%
      9. Max/Min Monthly Corpus Return (%) vs. K%
      10. Avg Min Available Cash (%) vs. K%
      
    The remaining subplot slots (11 and 12) are turned off.
    A super-title is added at the top including lookback and max allocation.
    The figure is saved with a filename that includes the lookback and max allocation.
    """
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by k_percent to ensure ascending order
    final_df = final_df.sort_values('k_percent').reset_index(drop=True)
    
    # Create a 3x4 grid of subplots (12 in total)
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(24, 18))
    fig.suptitle(f'Performance Metrics (Lookback = {lookback_period} months, Max Allocation = {max_alloc:.2f})',
                 fontsize=24, y=0.98)
    
    # Flatten the axs for easier indexing
    axs = axs.flatten()
    
    # Plot 1: Mean of Mean Performance vs. k_percent
    axs[0].plot(final_df['k_percent'], final_df['mean_of_mean_performance'], marker='o', color='blue')
    axs[0].set_title('Mean of Mean Performance')
    axs[0].set_xlabel('K Percent')
    axs[0].set_ylabel('Mean Performance')
    axs[0].grid(True)
    
    # Plot 2: Number of Monthly Predictions vs. k_percent
    axs[1].plot(final_df['k_percent'], final_df['number_of_monthly_predictions'], marker='o', color='orange')
    axs[1].set_title('Monthly Predictions')
    axs[1].set_xlabel('K Percent')
    axs[1].set_ylabel('Avg Predictions/Month')
    axs[1].grid(True)
    
    # Plot 3: Combined Metric vs. k_percent
    axs[2].plot(final_df['k_percent'], final_df['combined_metric'], marker='o', color='green')
    axs[2].set_title('Combined Metric')
    axs[2].set_xlabel('K Percent')
    axs[2].set_ylabel('Combined Metric')
    axs[2].grid(True)
    
    # Plot 4: Mean Monthly Corpus Return (%) vs. k_percent
    axs[3].plot(final_df['k_percent'], final_df['mean_monthly_corpus_return'] * 100, marker='o', color='red')
    axs[3].set_title('Mean Corpus Return (%)')
    axs[3].set_xlabel('K Percent')
    axs[3].set_ylabel('Mean Return (%)')
    axs[3].grid(True)
    
    # Plot 5: Median Monthly Corpus Return (%) vs. k_percent
    axs[4].plot(final_df['k_percent'], final_df['median_monthly_corpus_return'] * 100, marker='o', color='blue')
    axs[4].set_title('Median Corpus Return (%)')
    axs[4].set_xlabel('K Percent')
    axs[4].set_ylabel('Median Return (%)')
    axs[4].grid(True)
    
    # Plot 6: Mode Monthly Corpus Return (%) vs. k_percent
    axs[5].plot(final_df['k_percent'], final_df['mode_monthly_corpus_return'] * 100, marker='o', color='green')
    axs[5].set_title('Mode Corpus Return (%)')
    axs[5].set_xlabel('K Percent')
    axs[5].set_ylabel('Mode Return (%)')
    axs[5].grid(True)
    
    # Plot 7: STD of Monthly Corpus Return (%) vs. k_percent
    axs[6].plot(final_df['k_percent'], final_df['std_monthly_corpus_return'] * 100, marker='o', color='purple')
    axs[6].set_title('STD Corpus Return (%)')
    axs[6].set_xlabel('K Percent')
    axs[6].set_ylabel('STD (%)')
    axs[6].grid(True)
    
    # Plot 8: Mean/STD Ratio (%) vs. k_percent
    ratio = final_df.apply(lambda r: (r['mean_monthly_corpus_return'] / r['std_monthly_corpus_return']) * 100
                            if r['std_monthly_corpus_return'] != 0 else np.nan, axis=1)
    axs[7].plot(final_df['k_percent'], ratio, marker='o', color='brown')
    axs[7].set_title('Mean/STD Ratio (%)')
    axs[7].set_xlabel('K Percent')
    axs[7].set_ylabel('Ratio (%)')
    axs[7].grid(True)
    
    # Plot 9: Max/Min Monthly Corpus Return (%) vs. k_percent
    axs[8].plot(final_df['k_percent'], final_df['max_monthly_corpus_return'] * 100, marker='o', linestyle='-', color='green', label='Max')
    axs[8].plot(final_df['k_percent'], final_df['min_monthly_corpus_return'] * 100, marker='o', linestyle='-', color='blue', label='Min')
    axs[8].set_title('Max/Min Corpus Return (%)')
    axs[8].set_xlabel('K Percent')
    axs[8].set_ylabel('Return (%)')
    axs[8].grid(True)
    axs[8].legend()
    
    # Plot 10: Average Minimum Available Cash (%) vs. k_percent
    axs[9].plot(final_df['k_percent'], final_df['avg_min_available_cash']*100, marker='o', color='magenta')
    axs[9].set_title('Avg Min Available Cash (%)')
    axs[9].set_xlabel('K Percent')
    axs[9].set_ylabel('Available Cash (%)')
    axs[9].grid(True)
    
    # Turn off any unused subplots (here, indices 10 and 11)
    axs[10].axis('off')
    axs[11].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = os.path.join(output_dir, f'performance_metrics_{lookback_period}_{max_alloc:.2f}.png')
    plt.savefig(filename)
    plt.close()

def calculate_metrics(df_train, df_test, start_date, end_date, lookback_period, k_percentages, 
         cost_per_trade, corpus_fraction, max_allocation_fraction, day_offset, output_dir):
    
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using output directory: {output_dir}")
    
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    num_months = ((end_ts.year - start_ts.year) * 12) + (end_ts.month - start_ts.month) + 1
    
    # # Define a list of months to exclude from the final aggregation (if necessary)
    # excluded_months = [pd.Timestamp('2019-07-01'), pd.Timestamp('2018-06-01')]
    
    iteration_results = []
    # print("Running analyses for each K%...")

    # Ensure k_percentages is an array even if a single number is passed.
    k_percentages = np.atleast_1d(k_percentages)

    # (Optionally print its type for debugging)
    # print("Inside calculate_metrics, type of k_percentages:", type(k_percentages))

    for k in k_percentages:
        k_float = float(k)
        monthly_df = run_rolling_analysis(df_train, df_test, start_date, end_date, k_float, lookback_period,cost_per_trade, corpus_fraction, max_allocation_fraction, day_offset)
        
        # # Remove the undesired months from the aggregated monthly_df
        # if not monthly_df.empty:
        #     monthly_df = monthly_df[~monthly_df['month'].isin(excluded_months)]
        
        if monthly_df.empty:
            iteration_results.append({
                'mean_of_mean_performance': np.nan,
                'number_of_monthly_predictions': 0,
                'combined_metric': np.nan,
                'mean_monthly_corpus_return': np.nan,
                'std_monthly_corpus_return': np.nan,
                'min_monthly_corpus_return': np.nan,
                'min_month': None,
                'max_monthly_corpus_return': np.nan,
                'max_month': None,
                'median_monthly_corpus_return': np.nan,
                'mode_monthly_corpus_return': np.nan,
                'avg_min_available_cash': np.nan,
                'k_percent': k
            })
            continue
        
        overall_mean_perf = monthly_df['mean_performance'].mean()
        overall_count = monthly_df['count'].sum()
        overall_combined = monthly_df['combined_metric'].mean()
        
        mean_corpus_return = monthly_df['corpus_return'].mean()
        std_corpus_return = monthly_df['corpus_return'].std()
        max_idx = monthly_df['corpus_return'].idxmax()
        max_corpus_return = monthly_df.loc[max_idx, 'corpus_return']
        max_corpus_return_month = monthly_df.loc[max_idx, 'month']
        
        min_idx = monthly_df['corpus_return'].idxmin()
        min_corpus_return = monthly_df.loc[min_idx, 'corpus_return']
        min_corpus_return_month = monthly_df.loc[min_idx, 'month']
        
        median_corpus_return = monthly_df['corpus_return'].median()
        mode_series = monthly_df['corpus_return'].mode()
        mode_corpus_return = mode_series.iloc[0] if not mode_series.empty else np.nan
        
        avg_min_available_cash = monthly_df['min_available_cash'].mean()
        
        monthly_predictions = overall_count / num_months
        
        iteration_results.append({
            'mean_of_mean_performance': overall_mean_perf,
            'number_of_monthly_predictions': monthly_predictions,
            'combined_metric': overall_combined,
            'mean_monthly_corpus_return': mean_corpus_return,
            'std_monthly_corpus_return': std_corpus_return,
            'min_monthly_corpus_return': min_corpus_return,
            'min_month': min_corpus_return_month,
            'max_monthly_corpus_return': max_corpus_return,
            'max_month': max_corpus_return_month,
            'median_monthly_corpus_return': median_corpus_return,
            'mode_monthly_corpus_return': mode_corpus_return,
            'avg_min_available_cash': avg_min_available_cash,
            'k_percent': k
        })
    
    final_results_df = pd.DataFrame(iteration_results)
    csv_path = os.path.join(output_dir, f"k_percent_performance_results_{lookback_period}_{max_allocation_fraction:.2f}.csv")
    final_results_df.to_csv(csv_path, index=False)
    print(f"[INFO] Results saved to '{csv_path}'")
    
    plot_performance_metrics(final_results_df, lookback_period, max_allocation_fraction, output_dir)
    print(f"[INFO] Plot saved as '{output_dir}/performance_metrics_{lookback_period}_{max_allocation_fraction:.2f}.png'")
    
    return final_results_df

# if __name__ == "__main__":
#     # File paths (update as needed)
#     file_path_d5 = r"C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2.csv"
#     file_path_d1 = r"C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv"

#     # Read data
#     df_d5 = pd.read_csv(file_path_d5).dropna(subset=['expected_return', 'actual_return'])
#     df_d1 = pd.read_csv(file_path_d1).dropna(subset=['expected_return', 'actual_return'])

#     start_date = "2018-01-01"
#     end_date = "2022-12-31"
#     lookback_period_range = np.arange(19, 25, 1)
#     # lookback_period_range = 24

#     # k_percent values
#     k_percentages = np.arange(1, 15.5, 0.5)
#     # k_percentages = [15.5]

#     # Additional parameters
#     corpus_fraction       = 0.8
#     cost_per_trade        = 0.0021
#     # max_allocation_fraction = 0.15
#     day_offset            = 5
#     outdir                = "Output_avbl_cash"

#     # Set up a grid over max_allocation_fraction from 0.10 to 0.25 in steps of 0.01
#     max_alloc_values = np.arange(0.10, 0.41, 0.01)
#     # max_alloc_values = np.arange(0.10, 0.15, 0.01)


#     # Loop over each lookback period and for each max allocation fraction value
#     for lookback in lookback_period_range:
#         for max_alloc in max_alloc_values:
#             print(f"\n==== Running analysis for lookback = {lookback} months, max_allocation_fraction = {max_alloc:.2f} ====")
#             results_df = main(df_d5, df_d5, start_date, end_date, lookback, k_percentages,
#                             cost_per_trade, corpus_fraction, max_alloc, day_offset, outdir)
#             # print(f"Sample final results for lookback = {lookback}, max_alloc = {max_alloc:.2f}:\n", results_df.head())
