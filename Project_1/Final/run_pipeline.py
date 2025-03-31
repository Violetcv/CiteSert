import os
import pandas as pd
from config import *
from data_preprocessing import preprocess_data
from calculate_metrics import calculate_metrics
from Search import get_top_or_bottom_n_global
from plot_top_configs import process_top_configs, plot_monthly_time_series


def main():
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Running data preprocessing for {N_DAYS_train} and {N_DAYS_test}")
    df_train = preprocess_data(N_DAYS_train)
    df_test = preprocess_data(N_DAYS_test)

    print(f"[INFO] Loaded train data ({len(df_train)} rows) and test data ({len(df_test)} rows).")

    # k_pct_results = {}  # Dictionary to store the results for later use if needed.
    # print("[INFO] Calculating metrics...")

    # for lookback in lookback_months:
    #     lookback_int = int(lookback)
    #     for max_alloc in max_allocation_values:
    #         print(f"Running lookback: {lookback}, max_alloc: {max_alloc}")
    #         k_pct_df = calculate_metrics(
    #             df_train, df_test, start_date, end_date, k_percentages,
    #             lookback_int, cost_per_trade, corpus_fraction, max_alloc, day_offset, output_dir
    #         )
    #         # Store the resulting dataframe in a dictionary keyed by lookback and max_alloc.
    #         k_pct_results[(lookback, max_alloc)] = k_pct_df
        
    # #  Combine all results into one DataFrame.
    # combined_results_list = []
    # for (lookback, max_alloc), df in k_pct_results.items():
    #     if not df.empty:
    #         temp = df.copy()
    #         temp['lookback'] = lookback
    #         temp['max_alloc'] = max_alloc
    #         combined_results_list.append(temp)

    # if not combined_results_list:
    #     print("[WARNING] No results recorded. Exiting.")
    #     return
    # all_results_df = pd.concat(combined_results_list, ignore_index=True)

    # # Choose a search metric: e.g., higher median_monthly_corpus_return is better.
    # # You can set search_metric_1 = "median_monthly_corpus_return" in your config.
    # # Sort descending by the search metric and pick top_n_results.
    # top_configs = all_results_df.sort_values(by=search_metric, ascending=False).head(top_n_results)

    # if top_configs.empty:
    #     print("[WARNING] No top configurations found. Skipping plots.")
    #     return

    # # Print the global top configurations:
    # print("[INFO] Top configurations based on", search_metric, ":")
    # print(top_configs.to_string(index=False))

    # # For each top configuration, use its lookback and max_alloc parameters to plot its monthly time series.
    # # Note: plot_monthly_time_series is assumed to plot the monthly corpus return for a given parameter combination.
    # print("[INFO] Plotting top configurations...")
    # for _, row in top_configs.iterrows():
    #     lookback_val = int(row['lookback'])
    #     max_alloc_val = float(row['max_alloc'])
    #     k_val = row['k_percent']
    #     # If you have a monthly time series output from calculate_metrics (or save it separately), 
    #     # you can pass that instead. In our earlier implementation, calculate_metrics writes CSV files.
    #     # Here, we will call plot_monthly_time_series with the train data. In many pipelines, you may
    #     # prefer to use the actual monthly_df for that config.
    #     plot_monthly_time_series(df_train, k_val, lookback_val, max_alloc_val, output_dir)

    print("[INFO] Pipeline complete. Results saved in:", output_dir)

if __name__ == "__main__":
    main()
