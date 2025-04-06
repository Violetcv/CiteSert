#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from calculate_metrics import run_rolling_analysis
from config import *  # Import parameters only

def plot_monthly_time_series(df_train, df_test, k, lookback, max_alloc, output_dir):
    """
    Generates time series plot for a single configuration
    Only creates new plot if it doesn't already exist
    """
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])

    # Generate unique filename
    plot_filename = f"corpus_ret_lb{lookback}_ma{max_alloc:.2f}_k{k:.1f}.png"
    plot_path = os.path.join(output_dir, "top_config", plot_filename)
    
    # Skip existing plots
    if os.path.exists(plot_path):
        print(f"[SKIP] Plot exists: {plot_filename}")
        return

    # Generate time series data
    try:
        monthly_df = run_rolling_analysis(
            df_train, df_test, start_date, end_date, k,
            lookback, cost_per_trade, corpus_fraction,
            max_alloc, day_offset
        )
    except Exception as e:
        print(f"[ERROR] Failed to generate data for LB{lookback} MA{max_alloc:.2f} K{k:.1f}: {str(e)}")
        return

    # Validate data
    if monthly_df.empty or 'month' not in monthly_df.columns:
        print(f"[WARNING] No data for LB{lookback} MA{max_alloc:.2f} K{k:.1f}")
        return

    # Plotting setup
    plt.figure(figsize=(14, 7))
    param_text = f"Lookback: {lookback} months\nMaxAlloc: {max_alloc:.2f}\nK: {k:.1f}%"
    plt.gcf().text(0.82, 0.15, param_text, bbox=dict(facecolor='white', alpha=0.5))

    # Data processing
    monthly_df['month'] = pd.to_datetime(monthly_df['month'])
    monthly_df = monthly_df.sort_values('month')
    corpus_pct = monthly_df['corpus_return'] * 100

    print(monthly_df)
    
    # Calculate metrics
    mean_return = corpus_pct.mean()
    median_return = corpus_pct.median()

    # Plot construction
    plt.plot(monthly_df['month'], corpus_pct, 
             marker='o', linestyle='-', color='blue', 
             label='Monthly Returns')
    plt.axhline(mean_return, color='red', linestyle='--', 
                label=f'Mean ({mean_return:.2f}%)')
    plt.axhline(median_return, color='green', linestyle='--', 
                label=f'Median ({median_return:.2f}%)')

    # Final touches
    plt.title(f'Monthly Corpus Returns (LB={lookback} MA={max_alloc:.2f} K={k:.1f})')
    plt.xlabel('Month')
    plt.ylabel('Return (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save output
    plt.savefig(plot_path)
    plt.close()
    print(f"[SUCCESS] Saved plot: {plot_filename}")

def process_top_configs(results_df, df_train, df_test, output_dir):
    """Handles top configurations from precomputed results"""
    print(f"\n[PHASE] Processing top {top_n_results} configurations")
    
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])

    # Get top configurations
    top_configs = results_df.sort_values(
        search_metric, 
        ascending=False
    ).head(top_n_results)
    
    if top_configs.empty:
        print("[WARNING] No valid configurations found")
        return
    
    # Plot each configuration
    for _, config in top_configs.iterrows():
        plot_monthly_time_series(
            df_train, df_test,
            k=float(config['k_percent']),
            lookback=int(config['lookback']),
            max_alloc=float(config['max_alloc']),
            output_dir=output_dir
        )