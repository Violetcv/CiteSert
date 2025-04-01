import os
import pandas as pd
from config import *
from data_preprocessing import preprocess_data
from calculate_metrics import calculate_metrics
from Search import get_top_or_bottom_n_global
from plot_top_configs import process_top_configs, plot_monthly_time_series

def get_preprocessed_filename(n_days):
    """Generate standardized filename for preprocessed data"""
    return os.path.join(output_dir, f"preprocessed_{n_days}_days.csv")

def get_metrics_filename(lookback, max_alloc):
    """Generate standardized filename for metrics results"""
    return os.path.join(output_dir, f"metrics_lookback{lookback}_alloc{max_alloc:.2f}.csv")

def get_plot_filename(lookback, max_alloc, k_val):
    """Generate standardized filename for plot"""
    return os.path.join(output_dir, f"plot_lb{lookback}_ma{max_alloc:.2f}_k{k_val:.1f}.png")

def main():
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocessing with cache check
    print(f"[INFO] Checking preprocessed data...")
    train_file = get_preprocessed_filename(N_DAYS_train)
    test_file = get_preprocessed_filename(N_DAYS_test)
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("[INFO] Loading cached preprocessed data")
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
    else:
        print(f"[INFO] Running preprocessing for {N_DAYS_train} and {N_DAYS_test}")
        df_train = preprocess_data(N_DAYS_train)
        df_test = preprocess_data(N_DAYS_test)
        df_train.to_csv(train_file, index=False)
        df_test.to_csv(test_file, index=False)
    
    print(f"[INFO] Loaded train ({len(df_train)} rows) and test ({len(df_test)} rows) data")

    # Metrics calculation with cache check
    k_pct_results = {}
    print("[INFO] Calculating/Rloading metrics...")
    
    for lookback in lookback_months:
        lookback_int = int(lookback)
        for max_alloc in max_allocation_values:
            metrics_file = get_metrics_filename(lookback_int, max_alloc)
            
            if os.path.exists(metrics_file):
                print(f"[CACHE] Loading existing: {metrics_file}")
                k_pct_results[(lookback_int, max_alloc)] = pd.read_csv(metrics_file)
                continue
                
            print(f"[COMPUTE] Running lookback: {lookback}, max_alloc: {max_alloc}")
            k_pct_df = calculate_metrics(
                df_train, df_test, start_date, end_date, lookback_int, k_percentages,
                cost_per_trade, corpus_fraction, max_alloc, day_offset, output_dir
            )
            
            # Save results for future runs
            k_pct_df.to_csv(metrics_file, index=False)
            k_pct_results[(lookback_int, max_alloc)] = k_pct_df

    # Results aggregation
    combined_results_list = []
    for (lookback, max_alloc), df in k_pct_results.items():
        if not df.empty:
            temp = df.copy()
            temp['lookback'] = lookback
            temp['max_alloc'] = max_alloc
            combined_results_list.append(temp)

    if not combined_results_list:
        print("[WARNING] No results recorded. Exiting.")
        return
    
    all_results_df = pd.concat(combined_results_list, ignore_index=True)

    # Top configurations processing
    top_configs = all_results_df.sort_values(by=search_metric, ascending=False).head(top_n_results)
    if top_configs.empty:
        print("[WARNING] No top configurations found. Skipping plots.")
        return

    print("[INFO] Top configurations based on", search_metric)
    print(top_configs.to_string(index=False))

    # Plotting with cache check
    print("[INFO] Plotting top configurations...")
    for _, row in top_configs.iterrows():
        lookback_val = int(row['lookback'])
        max_alloc_val = float(row['max_alloc'])
        k_val = float(row['k_percent'])
        plot_file = get_plot_filename(lookback_val, max_alloc_val, k_val)
        
        if os.path.exists(plot_file):
            print(f"[CACHE] Plot exists: {plot_file}")
            continue
            
        print(f"[PLOTTING] Generating {plot_file}")
        plot_monthly_time_series(df_train, df_test, k_val, lookback_val, max_alloc_val, output_dir)

    print("[INFO] Pipeline complete. Results saved in:", output_dir)

if __name__ == "__main__":
    main()