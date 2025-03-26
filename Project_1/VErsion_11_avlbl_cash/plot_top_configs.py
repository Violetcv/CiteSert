import os
import pandas as pd
from Avlbl_cash import run_rolling_analysis, plot_monthly_time_series
from Search import get_top_or_bottom_n_global

# --- Load data (same as Avlbl_cash.py) ---
file_path_d5 = r"C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2.csv"
file_path_d1 = r"C:\Users\disch\Desktop\CiteSert\Project_1\Version6_40krows_3metrics\input_data\filtered_data_2_n1.csv"
df_d5 = pd.read_csv(file_path_d5).dropna(subset=['expected_return', 'actual_return'])
df_d5['date'] = pd.to_datetime(df_d5['date'])
df_d1 = pd.read_csv(file_path_d1).dropna(subset=['expected_return', 'actual_return'])

# --- Parameters (match Avlbl_cash.py) ---
start_date = "2018-01-01"
end_date = "2022-12-31"
cost_per_trade = 0.0021
corpus_fraction = 0.8
day_offset = 5
output_dir = os.path.abspath("Output_avbl_cash/Top_Config_Plots")  # Directory for new plots
os.makedirs(output_dir, exist_ok=True)

# --- Get top configurations for mean and median ---
folder_path = r"C:\Users\disch\Desktop\CiteSert\Project_1\VErsion_11_avlbl_cash\Output_avbl_cash"
mean_top = get_top_or_bottom_n_global(folder_path, "mean_monthly_corpus_return", top_n=1, ascending=False)
median_top = get_top_or_bottom_n_global(folder_path, "median_monthly_corpus_return", top_n=1, ascending=False)

# --- Process each top configuration ---
def process_top_configs(top_df, metric_name):
    if top_df.empty:
        print(f"No data found for {metric_name}.")
        return
    
    for idx, row in top_df.iterrows():
        filename = row['filename']
        k_percent = row['k_percent']
        
        # Extract lookback and max_alloc from filename
        try:
            base = filename.split('_results_')[1].replace('.csv', '')
            lookback, max_alloc = base.split('_')
            lookback = int(lookback)
            max_alloc = float(max_alloc)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            continue
        
        # Run analysis for this configuration
        monthly_df = run_rolling_analysis(
            df_d5, df_d5, start_date, end_date, k_percent, lookback,
            cost_per_trade, corpus_fraction, max_alloc, day_offset
        )
        
        if monthly_df.empty:
            print(f"No monthly data for {filename} (k={k_percent}).")
            continue
        
        # Plot and save
        plot_monthly_time_series(
            monthly_df, lookback, max_alloc, output_dir
        )
        print(f"Plot saved for {metric_name}: lookback={lookback}, max_alloc={max_alloc}, k={k_percent}")

# --- Run for mean and median ---
print("\nProcessing top MEAN configurations...")
process_top_configs(mean_top, "mean_monthly_corpus_return")

print("\nProcessing top MEDIAN configurations...")
process_top_configs(median_top, "median_monthly_corpus_return")

print("\nDone. Check output directory:", output_dir)