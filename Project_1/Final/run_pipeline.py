import os
import pandas as pd
from tqdm import tqdm
from config import *
from data_preprocessing import preprocess_data
from calculate_metrics import calculate_metrics
from Search import save_top_configs
from plot_top_configs import plot_monthly_time_series

def get_preprocessed_filename(n_days):
    """Generate standardized filename for preprocessed data"""
    return os.path.join(data_dir, f"filtered_data_{n_days}.csv")

def get_metrics_filename(lookback, max_alloc):
    """Generate standardized filename for metrics results"""
    return os.path.join(output_dir, f"k_percent_performance_results_{lookback}_{max_alloc:.2f}.csv")

def get_plot_filename(lookback, max_alloc, k):
    """Generate standardized filename for plot with parameters"""
    return os.path.join(output_dir, f"corpus_ret_lb{lookback}_ma{max_alloc:.2f}_k{k:.1f}.png")

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
    
    print(f"[INFO] Loaded train data ({len(df_train)} of n = {N_DAYS_train} rows) and test data ({len(df_test)} of n = {N_DAYS_test} rows)")

    # Metrics calculation with cache check
    k_pct_results = {}
    print("[INFO] Calculating/Reloading metrics...")
    
    # Generate all combinations of parameters
    combinations = list(itertools.product(lookback_months, max_allocation_values))
    total_combinations = len(combinations)
    
    # Process each combination with tqdm
    with tqdm(combinations, desc="Processing Configurations", total=total_combinations) as pbar:
        for lookback, max_alloc in pbar:
            lookback_int = int(lookback)
            metrics_file = get_metrics_filename(lookback_int, max_alloc)
            
            # Update progress bar description
            pbar.set_postfix_str(f"LB: {lookback}, MA: {max_alloc}")
            
            if os.path.exists(metrics_file):
                tqdm.write(f"[CACHE] Loading existing: {os.path.basename(metrics_file)}")
                k_pct_results[(lookback_int, max_alloc)] = pd.read_csv(metrics_file)
                continue
                
            # Calculate metrics if not cached
            tqdm.write(f"[COMPUTE] Running lookback: {lookback}, max_alloc: {max_alloc}")
            k_pct_df = calculate_metrics(
                df_train, df_test, start_date, end_date, lookback_int, k_percentages,
                cost_per_trade, corpus_fraction, max_alloc, day_offset, output_dir
            )
            k_pct_df.to_csv(metrics_file, index=False)
            k_pct_results[(lookback_int, max_alloc)] = k_pct_df

    # Aggregate results
    combined_results = [
        df.assign(lookback=lookback, max_alloc=max_alloc)
        for (lookback, max_alloc), df in k_pct_results.items()
        if not df.empty
    ]
    
    if not combined_results:
        print("[WARNING] No results recorded. Exiting.")
        return
    
    all_results_df = pd.concat(combined_results, ignore_index=True)

    # Process top configurations
    top_configs = all_results_df.sort_values(search_metric, ascending=False).head(top_n_results)
    if top_configs.empty:
        print("[WARNING] No top configurations found. Skipping plots.")
        return

    print("\n[INFO] Top configurations based on", search_metric)
    print(top_configs[['lookback', 'max_alloc', 'k_percent', search_metric]].to_string(index=False))

    # Plotting with intelligent caching
    print("\n[INFO] Plotting top configurations...")
    for idx, (_, row) in enumerate(top_configs.iterrows(), 1):
        lb = int(row['lookback'])
        ma = float(row['max_alloc'])
        k = float(row['k_percent'])
        plot_path = get_plot_filename(lb, ma, k)
        
        print(f"\nProcessing config {idx}/{len(top_configs)}")
        print(f"Lookback: {lb}m | MaxAlloc: {ma:.2f} | K: {k:.1f}%")
        
        if os.path.exists(plot_path):
            print(f"|- Plot exists: {os.path.basename(plot_path)}")
            continue
            
        print(f"|- Generating new plot: {os.path.basename(plot_path)}")
        plot_monthly_time_series(df_train, df_test, k, lb, ma, output_dir)

    # Save top configurations
    save_top_configs(all_results_df, output_dir, 
                    ['mean_monthly_corpus_return', 'median_monthly_corpus_return'])

    print("\n[INFO] Pipeline complete. Results saved in:", output_dir)

if __name__ == "__main__":
    main()