#!/usr/bin/env python3
import os
import pandas as pd
from calculate_metrics import run_rolling_analysis
from Search import get_top_or_bottom_n_global
from config import *  # Import all paths and parameters from config.py

def plot_monthly_time_series(monthly_df,k , lookback_period, max_alloc, output_dir):
    """
    Plots the monthly corpus return time series and overlays mean/median return.
    
    Params:
      monthly_df: DataFrame containing the train data with a 'month' field
      lookback_period: Lookback period (months) used in the configuration
      max_alloc: Maximum allocation value used in the configuration
      output_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt
    monthly_df = monthly_df.copy()
    monthly_df['month'] = pd.to_datetime(monthly_df['month'])
    monthly_df_sorted = monthly_df.sort_values('month')

    corpus_pct = monthly_df_sorted['corpus_return'] * 100
    overall_mean = corpus_pct.mean()
    overall_median = corpus_pct.median()

    plt.figure(figsize=(14, 7))
    plt.plot(monthly_df_sorted['month'], corpus_pct, marker='o', linestyle='-', color='blue', label='Monthly Corpus Return')
    plt.axhline(y=overall_mean, color='red', linestyle='--', label=f'Mean ({overall_mean:.2f}%)')
    plt.axhline(y=overall_median, color='green', linestyle='--', label=f'Median ({overall_median:.2f}%)')

    plt.title(f'Monthly Corpus Return (%) (Lookback = {lookback_period} months, MaxAlloc = {max_alloc:.2f}, K = {k})')
    plt.xlabel('Month')
    plt.ylabel('Corpus Return (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(output_dir, f'monthly_time_series_{lookback_period}_{max_alloc:.2f}_{k}pct.png')
    plt.savefig(filename)
    plt.close()
    print(f"[INFO] Monthly time series plot saved to: {filename}")

def process_top_configs(df_train):
    """
    Processes and plots top configurations.
    This function searches for the top configurations based on the given search metric,
    and then generates monthly time series plots for each configuration.
    
    Params:
      df_train: Training DataFrame already loaded in the previous file.
    """
    print(f"[INFO] Searching for top {top_n_results} configurations based on {search_metric}...")
    top_configs = get_top_or_bottom_n_global(output_dir, search_metric, top_n=top_n_results)

    if top_configs.empty:
        print("[WARNING] No top configurations found. Skipping plotting.")
        return

    for _, row in top_configs.iterrows():
        lookback = int(row['lookback'])
        max_alloc = float(row['max_alloc'])
        k = row['k_percent']
        print(f"[INFO] Plotting for lookback: {lookback}, max_alloc: {max_alloc}")
        plot_monthly_time_series(df_train, k ,  lookback, max_alloc, output_dir)

    print("[INFO] Completed plotting top configurations.")
