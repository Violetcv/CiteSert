import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

def get_best_authors(df_train, start_date, end_date, k_percent):
    """
    Select top K% authors from df_train between start_date and end_date.
    Score each row as sign(expected_return)*actual_return, sum by author.
    """
    subset = df_train[(df_train['date'] >= start_date) &
    (df_train['date'] <= end_date)].copy()

    if subset.empty:
        return pd.DataFrame(columns=['author'])

    subset['performance'] = np.sign(subset['expected_return']) * subset['actual_return']
    author_perf = (
        subset.groupby('author')['performance']
        .agg(['sum', 'count'])
        .reset_index()
        .rename(columns={'sum': 'total_performance', 'count': 'num_predictions'})
    )

    n_select = max(1, int(np.ceil(len(author_perf) * k_percent / 100.0)))
    top_authors = author_perf.nlargest(n_select, 'total_performance')

    return top_authors[['author']]

def get_best_authors(df_train, start_date, end_date, k_percent):
    """
    Select top K% authors from df_train between start_date and end_date.
    Score each row as sign(expected_return)*actual_return, sum by author.
    """
    subset = df_train[(df_train['date'] >= start_date) &
    (df_train['date'] <= end_date)].copy()


    if subset.empty:
        return pd.DataFrame(columns=['author'])

    subset['performance'] = np.sign(subset['expected_return']) * subset['actual_return']
    author_perf = (
        subset.groupby('author')['performance']
        .agg(['sum', 'count'])
        .reset_index()
        .rename(columns={'sum': 'total_performance', 'count': 'num_predictions'})
    )

    n_select = max(1, int(np.ceil(len(author_perf) * k_percent / 100.0)))
    top_authors = author_perf.nlargest(n_select, 'total_performance')
    return top_authors[['author']]
def calculate_monthly_metrics(df_test, best_authors, target_month):
    """
    For a single month, restricted to the best_authors.
    Returns a dict with:
    - 'year_month': (YYYY-MM format)
    - 'mean_performance'
    - 'num_predictions'
    - 'combined_metric'
    - 'corpus_return'
    """
    # Identify the month's start/end
    month_start = pd.to_datetime(target_month).replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(1)

    # Filter data for the month and best authors
    month_data = df_test[
        (df_test['date'] >= month_start) &
        (df_test['date'] <= month_end) &
        (df_test['author'].isin(best_authors['author']))
    ].copy()

    if month_data.empty:
        return {
            'year_month': month_start.strftime('%Y-%m'),
            'mean_performance': np.nan,
            'num_predictions': 0,
            'combined_metric': np.nan,
            'corpus_return': np.nan
        }

    # Mean performance
    month_data['performance'] = np.sign(month_data['expected_return']) * month_data['actual_return']
    mean_perf = month_data['performance'].mean()
    num_preds = len(month_data)
    combined_metric_val = (mean_perf - 0.0004)*num_preds

    # Corpus logic (day-by-day)
    initial_corpus = 1.0
    current_corpus = initial_corpus
    corpus_fraction = 0.8

    # Group by day; for each day, allocate capital among the trades
    for date_key, day_trades in month_data.groupby('date'):
        num_trades = len(day_trades)
        base_allocation = (corpus_fraction * current_corpus)/num_trades
        max_allocation = 0.25 * initial_corpus
        allocation_per_trade = min(base_allocation, max_allocation)
        
        daily_pl = 0.0
        for _, trade in day_trades.iterrows():
            signed_direction = np.sign(trade['expected_return'])
            trade_return = (signed_direction * trade['actual_return'] - 0.0004) 
            daily_pl += (trade_return * allocation_per_trade)
        
        current_corpus += daily_pl

    corpus_return = (current_corpus - initial_corpus)/initial_corpus

    return {
        'year_month': month_start.strftime('%Y-%m'),
        'mean_performance': mean_perf,
        'num_predictions': num_preds,
        'combined_metric': combined_metric_val,
        'corpus_return': corpus_return
    }

def rolling_analysis_for_all_months(df_train, df_test, k_percent, lookback_months=14):
    """
    From 2018-01 to 2022-12, for each month:
    1) Determine best authors from the preceding lookback_months (in df_train).
    2) Calculate monthly metrics in df_test.
    Returns a DataFrame of monthly rows.
    """
    all_rows = []

    start_date = pd.to_datetime('2018-01-01')
    end_date   = pd.to_datetime('2022-12-31')

    # Build a list of months from 2018-01 to 2022-12
    current = start_date
    while current <= end_date:
        # determine training range
        train_end   = current - timedelta(days=1)
        train_start = train_end - pd.DateOffset(months=lookback_months)
        
        # get best authors
        top_authors = get_best_authors(df_train, train_start, train_end, k_percent)
        
        # calculate that month's metrics
        month_info = calculate_monthly_metrics(df_test, top_authors, current)
        all_rows.append(month_info)
        
        # move to next month
        current += pd.DateOffset(months=1)

    return pd.DataFrame(all_rows)
    
if __name__ == '__main__':
    # Example file path (adjust if needed)
    train_file = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv'
    test_file  = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv'

    df_train = pd.read_csv(train_file).dropna(subset=['expected_return','actual_return'])
    df_test  = pd.read_csv(test_file).dropna(subset=['expected_return','actual_return'])

    # Ensure date columns are datetime
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date']  = pd.to_datetime(df_test['date'])

    print("Starting Analysis from 2018 to 2022 ...")

    # Evaluate exactly k = 15
    k_val = 15.0
    print(f"\nRunning rolling analysis for k={k_val}% ... This may take a bit ...")
    monthly_df = rolling_analysis_for_all_months(df_train, df_test, k_val, lookback_months=14)

    # Print out a snippet of the monthly results
    print("\nFirst 10 rows of monthly results:")
    print(monthly_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Filter for months where mean_performance > 0.001 and corpus_return < 0
    print(f"\nFiltering months for condition: mean_performance > 0.001 and corpus_return < 0")
    filtered = monthly_df[(monthly_df['mean_performance'] > 0.001) & (monthly_df['corpus_return'] < 0)]

    if filtered.empty:
        print("No months found that satisfy these conditions.")
    else:
        print("Months found:")
        print(filtered.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Save results to CSV
    monthly_df.to_csv("rolling_analysis_k15.csv", index=False)
    print("\nAll monthly results for k=15 saved to 'rolling_analysis_k15.csv'")