import numpy as np
import pandas as pd

file_path = "daily_trade_calculations_2022_05.csv"
df = pd.read_csv(file_path, parse_dates=['date'])
# print(df.columns)

# Filter for k=15.5
k15_5 = df

# Calculate performance
k15_5['performance'] = np.sign(k15_5['expected_return']) * k15_5['actual_return']

author_summary = k15_5.groupby('author').agg(
    mean_performance=('performance', 'mean'),
    count=('performance', 'count')
    ).reset_index()

# Calculate metrics
author_summary['monthly_combined'] = (author_summary['mean_performance'] - 0.0004) * author_summary['count']
# print(author_summary.columns)

print("Per-Author Summary for May 2022:")
print(author_summary.to_string(index=False))
overall_mean_performance = author_summary['mean_performance'].mean()
overall_total_count = author_summary['count'].sum()
trade_return_sign_direction_sum = (k15_5['trade_return'] * k15_5['signed_direction']).sum()
trade_return_sign_direction_sum_upon_count_of_trades = trade_return_sign_direction_sum/overall_total_count
overall_monthly_combined = (overall_mean_performance - 0.0004) * overall_total_count

print("\nOverall Monthly Summary for May 2022:")
print(f"Mean of Mean Performance: {overall_mean_performance:.6f}")
print(f"Total Count: {overall_total_count}")
print(f"Combined Metric: {overall_monthly_combined:.6f}")
print(f"trade_return_sign_direction_sum: {trade_return_sign_direction_sum}")
print(f"trade_return_sign_direction_sum_upon_count_of_trades: {trade_return_sign_direction_sum_upon_count_of_trades}")
print(f"Sum of expected return: {k15_5['expected_return'].sum()}")