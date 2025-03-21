#!/usr/bin/env python3
import os
import pandas as pd

def get_top_n_global_min(folder_path, top_n=5):
    """
    Reads all CSV files in the given folder, 
    and for those that contain the column 'min_monthly_corpus_return',
    adds a 'filename' column, then combines them. 
    The resulting combined DataFrame is sorted in descending order by the 
    'min_monthly_corpus_return' column. The function returns the top n rows 
    (global across all files) along with the file name from where each row came.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    if not csv_files:
        print("No CSV files found in folder:", folder_path)
        return pd.DataFrame()
    
    dfs = []
    for file in csv_files:
        full_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(full_path)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        if 'min_monthly_corpus_return' not in df.columns:
            print(f"File {file} does not contain 'min_monthly_corpus_return'. Skipping.")
            continue

        if df.empty:
            print(f"File {file} is empty. Skipping.")
            continue

        # Add a column indicating the filename
        df['filename'] = file
        dfs.append(df)

    if not dfs:
        print("No CSV files with the required column were processed.")
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort descending by min_monthly_corpus_return
    combined_df_sorted = combined_df.sort_values('min_monthly_corpus_return', ascending=False)
    
    # Get top n rows
    top_n_df = combined_df_sorted.head(top_n)
    return top_n_df

if __name__ == "__main__":
    # Set the folder path where your CSV files are located.
    folder_path = r"C:\Users\disch\Desktop\CiteSert\Project_1\VErsion10_std_deviation\Max_allocation_grid\Output_max_allocation_grid"
    
    # Set the number of top rows to display (global top n)
    top_n = 5
    
    top_rows = get_top_n_global_min(folder_path, top_n)
    if not top_rows.empty:
        output_file = os.path.join(folder_path, "global_top_min_results.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Top {top_n} rows (global) sorted by min_monthly_corpus_return:\n")
            f.write(top_rows.to_string(index=False))
        print(f"Results saved to {output_file}")
    else:
        print("No results found.") 
