import os
import numpy as np

# Data Preprocessing 
# --- File Paths ---
BASE_DIR = os.path.abspath(".")  # Adjust if needed
data_dir = os.path.join(BASE_DIR, "data") # 3 files

N_DAYS_train = 3
N_DAYS_test = 3

output_dir = os.path.join(BASE_DIR, f"outputs_n{N_DAYS_train}_d{N_DAYS_test}") # store filtered_data, k percent analysis files and graphs


# Calculate Metrics
# --- Date Range ---
start_date = "2018-01-01"
end_date = "2022-12-31"

# --- Parameters ---
k_percentages = np.arange(1, 15.5, 0.5)
lookback_months = np.arange(19, 25, 1)
corpus_fraction = 0.8
cost_per_trade = 0.0021
max_allocation_values = np.arange(0.10, 0.41, 0.01)  # Fraction of corpus to allocate
day_offset = 5  # Holding period (in business days)



# --- Debugging Parameters ---
DEBUG_MODE = False  # Set to False for full dataset
DEBUG_LIMIT = 2000  # Number of rows to use when debugging


# Top parameters config
top_n_results = 5  # Number of best configurations to consider
search_metric = "median_monthly_corpus_return"  # Can be "mean_monthly_corpus_return" or others
# search_metric_2 = "mean_monthly_corpus_return"

search_metrics = ['mean_monthly_corpus_return', 'median_monthly_corpus_return']