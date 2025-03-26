#!/usr/bin/env python3
import os
import pandas as pd

def get_top_or_bottom_n_global(folder_path, col_name, top_n=5, ascending=False):
    """
    Reads all CSV files in the given folder. For each CSV that contains the specified column (col_name),
    adds a 'filename' column and combines them.
    The combined DataFrame is then sorted by col_name in ascending or descending order
    (ascending=False means top rows; ascending=True means bottom rows).
    
    Returns the top_n rows (global) along with the file name from where each row came.
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

        if col_name not in df.columns:
            print(f"File {file} does not contain '{col_name}'. Skipping.")
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

    # Sort by the specified column; ascending parameter controls order.
    combined_df_sorted = combined_df.sort_values(col_name, ascending=ascending)

    # Get top_n rows based on the sort order
    top_n_df = combined_df_sorted.head(top_n)
    return top_n_df

if __name__ == "__main__":
    # Set the folder path where your CSV files are located.
    folder_path = r"C:\Users\disch\Desktop\CiteSert\Project_1\VErsion_11_avlbl_cash\Output_avbl_cash"
    
    # Specify the column you want to search on:
    # Change the col_name to "max_monthly_corpus_return" for maximum values or
    # "median_monthly_corpus_return" for median, etc.
    # col_name = "max_monthly_corpus_return"
    # col_name = "min_monthly_corpus_return"
    # col_name = "avg_min_available_cash"
    # col_name = "mean_monthly_corpus_return"
    col_name = "median_monthly_corpus_return"
    
    # Set the number of top rows to display (global top n)
    top_n = 5

    # For top rows, sort descending (ascending=False). For bottom rows, set ascending=True.
    result_df = get_top_or_bottom_n_global(folder_path, col_name, top_n, ascending=False)
    
    if not result_df.empty:
        output_file = os.path.join(folder_path, f"global_{col_name}_results.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Top {top_n} rows (global) sorted by {col_name} (descending):\n")
            f.write(result_df.to_string(index=False))
            
            # Also, if you want to show bottom rows, you can compute that too:
            bottom_df = get_top_or_bottom_n_global(folder_path, col_name, top_n, ascending=True)
            f.write("\n\n")
            f.write(f"Bottom {top_n} rows (global) sorted by {col_name} (ascending):\n")
            f.write(bottom_df.to_string(index=False))
        print(f"Results saved to {output_file}")
    else:
        print("No results found.") 