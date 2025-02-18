#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Load the filtered DataFrame from the CSV file
# Define the file path

file_path_train = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version3_40krows_n1\input_data\filtered_data_2.csv'  # 5-day returns data for training
file_path_test = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version3_40krows_n1\input_data\filtered_data_2_n1.csv'  # 1-day returns data for testing

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)

df_train = df_train.dropna(subset=['expected_return', 'actual_return'])
df_test = df_test.dropna(subset=['expected_return', 'actual_return'])

# Display the first few rows of the DataFrame to verify it loaded correctly
print(df_train.head())
print(df_test.head())


# In[3]:

df_train.info()
df_test.info()


# In[4]:


def calculate_correlation(expected_returns, actual_returns):
    """
    Calculate the correlation between expected and actual returns,
    ensuring no division by zero occurs.
    
    Args:
        expected_returns (pd.Series): Expected returns.
        actual_returns (pd.Series): Actual returns.

    Returns:
        float: Correlation coefficient, or NaN if not calculable.
    """
    # Check standard deviation to avoid division by zero
    if np.std(expected_returns) == 0 or np.std(actual_returns) == 0:
        return np.nan
    
    # Calculate correlation
    return np.corrcoef(expected_returns, actual_returns)[0, 1]


# In[5]:

# needs to use df_train
def get_best_authors_by_sector(df, start_date, end_date, corr_threshold, perf_threshold, max_authors=10):
    """
    Identify the best-performing authors per sector within a specific date range based on 
    correlation and mean performance thresholds.
    """
    print("\n### get_best_authors_by_sector ###")
    print(f"Start Date: {start_date}, End Date: {end_date}")
    print(f"Max Authors per Sector: {max_authors}, Correlation Threshold: {corr_threshold}, Performance Threshold: {perf_threshold}")
    
    # Convert date column to datetime if necessary
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for the specified date range
    period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    print(f"Filtered Data Shape: {period_df.shape}")

    if period_df.empty:
        print("No data available for the specified date range.")
        return pd.DataFrame(columns=['Sector', 'author', 'correlation', 'mean_performance'])

    # Calculate performance metrics for each sector-author pair
    author_sector_performance = []
    grouped = period_df.groupby(['Sector', 'author'])
    print(f"Number of Sector-Author Groups: {len(grouped)}")

    for (sector, author), group in grouped:
        print(f"\nProcessing Sector: {sector}, Author: {author}")
        print(f"Group Size: {len(group)}")

        if len(group) == 1:  # Ensure enough data points for correlation calculation
            correlation = -1
            continue

        # Compute correlation between expected and actual returns
        correlation = calculate_correlation(group['expected_return'], group['actual_return'])
        print(f"Correlation: {correlation}")

        # Compute mean performance
        group['performance'] = np.sign(group['expected_return']) * group['actual_return']
        mean_performance = group['performance'].sum()
        print(f"Mean Performance: {mean_performance}")

        # Store results if thresholds are met
        if not np.isnan(correlation) and correlation > corr_threshold and mean_performance > perf_threshold:
            print("Thresholds met. Adding to results.")
            author_sector_performance.append({
                'Sector': sector,
                'author': author,
                'correlation': correlation,
                'mean_performance': mean_performance,
                'number_of_rows_in_group': len(group)
            })
        else:
            print("Thresholds not met. Skipping.")
            continue

    # Create a DataFrame from the results
    performance_df = pd.DataFrame(author_sector_performance)
    print(f"\nPerformance DataFrame Shape: {performance_df.shape}")

    if performance_df.empty:
        print("No authors met the thresholds.")
        return pd.DataFrame(columns=['Sector', 'author', 'correlation', 'mean_performance'])

    # Sort authors by correlation and mean performance, and retain top authors per sector
    print("Sorting and selecting top authors by sector.")

    # top_authors_df = (performance_df
    #                   .sort_values(['correlation', 'mean_performance'], ascending=[False, False])
    #                   .groupby('Sector')
    #                   .head(max_authors)
    #                   .reset_index(drop=True))

    top_authors_df = (performance_df
                      .sort_values(['correlation', 'mean_performance'], ascending=[False, False])
                      .groupby('Sector', group_keys=False)
                      .apply(lambda x: x)  # Keeps all rows per sector
                      .reset_index(drop=True))

    print(f"Top Authors DataFrame Shape: {top_authors_df.shape}")
    print(f"Top Authors DataFrame Sample:\n{top_authors_df}")

    return top_authors_df


# In[6]:

# needs to use df_test
def calculate_monthly_performance(df, best_authors_df, target_month):
    """
    Calculate performance for each author-sector pair for a specific month
    using sign(expected_return) * actual_return
    """
    print(f"\nCalculating monthly performance for: {target_month.strftime('%Y-%m')}")

    # Define month range
    month_start = target_month.replace(day=1)
    month_end = (month_start + pd.offsets.MonthEnd(1))
    print(f"Month Start: {month_start}, Month End: {month_end}")

    # Filter data for the target month
    mask = (df['date'] >= month_start) & (df['date'] <= month_end)
    month_df = df[mask].copy()
    print(f"Filtered Data for Month: {month_df.shape[0]} rows")

    # Merge with best authors
    print(f"Best Authors DataFrame Shape: {best_authors_df.shape}")
    month_df = month_df.merge(
        best_authors_df[['Sector', 'author']],
        on=['Sector', 'author'],
        how='inner'
    )
    print(f"Data After Merging with Best Authors: {month_df.shape[0]} rows")

    # Calculate performance
    month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
    print(f"Performance Column Calculated. Sample Data:\n{month_df[['Sector', 'author', 'performance']]}")

    # Group and summarize performance
    performance_summary = (month_df.groupby(['Sector', 'author'])
                           ['performance']
                           .agg(mean_performance='mean', count='count')
                           .reset_index())
    performance_summary['month'] = month_start
    # print(f"Performance Summary for {target_month.strftime('%Y-%m')}:\n{performance_summary}")

    return performance_summary

def run_rolling_analysis(df_train, df_test, start_date, end_date, a, b, lookback_period=12):
    """
    Run the rolling analysis month by month
    """
    print("\nStarting Rolling Analysis")
    print(f"Start Date: {start_date}, End Date: {end_date}, Lookback Period: {lookback_period} months")
    print(f"Thresholds: Correlation (a={a}), Performance (b={b})")

    # Convert dates if they're strings
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    all_monthly_performance = []
    current_date = start_date

    while current_date <= end_date:
        print(f"\nProcessing Month: {current_date.strftime('%Y-%m')}")
        
        # Define training period
        training_end = current_date - timedelta(days=1)
        training_start = training_end - pd.DateOffset(months=lookback_period)
        print(f"Training Period: Start {training_start}, End {training_end}")

        # Get best authors; df_train
        best_authors = get_best_authors_by_sector(df_train, training_start, training_end, a, b)
        print(f"Best Authors DataFrame Shape: {best_authors.shape}")

        # Calculate performance for the current month; df_test
        if not best_authors.empty:
            monthly_perf = calculate_monthly_performance(df_test, best_authors, current_date)
            all_monthly_performance.append(monthly_perf)

            print(f"Monthly Performance for {current_date.strftime('%Y-%m')}:\n", monthly_perf)
        else:
            print(f"No Best Authors Found for {current_date.strftime('%Y-%m')}")

        # Move to next month
        current_date = current_date + pd.DateOffset(months=1)

    # Combine all results
    if all_monthly_performance:
        combined_results = pd.concat(all_monthly_performance, ignore_index=True)
        print(f"\nCombined Performance DataFrame Shape: {combined_results.shape}")
        # print(f"HERE \n{combined_results}")
        return combined_results
    else:
        print("No performance data collected.")
        return pd.DataFrame()


# In[7]:


def main(df_train, df_test):
    """
    Main function to analyze performance based on correlation and performance thresholds
    using a grid search approach.
    """
    print("\n### MAIN FUNCTION STARTED ###")

    # Convert date column to datetime if it's not already
    print("\nConverting 'date' column to datetime format.")
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')
    df_test['date'] = pd.to_datetime(df_test['date'], errors='coerce')

    
    # Define the grid for correlation and performance thresholds
    print("\nDefining thresholds grid for correlation and performance.")

    correlation_thresholds = np.arange(0, 0.1, 0.01)
    performance_thresholds = np.arange(0, 0.06, 0.005)

    
    print(f"Correlation Thresholds: {correlation_thresholds}")
    print(f"Performance Thresholds: {performance_thresholds}")
    
    # Initialize a dictionary to store results for each iteration
    all_results = {}
    iteration_results = []
    iteration = 0

    print("\n### Starting grid search ###")
    
    # Iterate over the grid of thresholds
    for a in tqdm(correlation_thresholds, desc="Correlation Thresholds"):
        for b in tqdm(performance_thresholds, desc="Performance Thresholds", leave=False):
            print(f"\nGrid Combination - Correlation Threshold: {a}, Performance Threshold: {b}")
            
            # Run the rolling analysis; df_train and df_test
            try:
                print("Running rolling analysis for the current thresholds.")
                results_df = run_rolling_analysis(
                    df_test, #n1_d1
                    df_train, #n1_d5
                    start_date='2014-01-01',
                    end_date='2022-12-31',
                    lookback_period=12,
                    a=a,
                    b=b
                )
                print(f"Rolling Analysis Completed. Results Shape: {results_df.shape}")
                # all_results_df = pd.DataFrame(results_df)
                print(f"HERE \n{results_df}")
            except Exception as e:
                print(f"Error during rolling analysis for a={a}, b={b}: {e}")
                results_df = pd.DataFrame()  # Empty DataFrame in case of errors
            
            # print("HERE - check mean of mean perf and mean of sum of counts")
            # print(results_df)

            # Calculate mean performance for the iteration
            if not results_df.empty and 'mean_performance' in results_df.columns:
                mean_of_mean_performance = results_df['mean_performance'].mean()
                print(f"Mean of Mean Performance for Current Thresholds: {mean_of_mean_performance:.4f}")
            else:
                print("Column 'mean_performance' not found or results_df is empty. Assigning NaN.")
                mean_of_mean_performance = np.nan

            # Calculate rows selected per month
            if not results_df.empty and 'count' in results_df.columns:
                number_of_monthly_predictions = results_df['count'].sum()/108
                print(f"Mean of Counts Per Month for Thresholds a={a}, b={b}:\n{number_of_monthly_predictions}")
            else:
                print("No valid data for rows selected per month calculation.")
                number_of_monthly_predictions = np.nan

            
            # Save the results DataFrame for the iteration
            iteration_key = f"iter_{iteration}_a_{a}_b_{b}"
            all_results[iteration_key] = results_df
            print(f"Iteration {iteration_key}: Results Saved.")
            
            # Append the iteration result
            iteration_results.append({
                'correlation_threshold': a,
                'performance_threshold': b,
                'mean_of_mean_performance': mean_of_mean_performance,
                'number_of_monthly_predictions': number_of_monthly_predictions
            })
            
            # Increment iteration counter
            iteration += 1

    # Convert iteration results to a DataFrame for analysis
    print("\nConverting iteration results to a DataFrame for final analysis.")
    grid_performance_df = pd.DataFrame(iteration_results)
    grid_performance_df.to_csv("input_data/grid_performance_n1_no_limits_n5_d1.csv")

    # If you want to save the dictionary as a simple text file:
    with open("input_data/all_results_n5_d1.txt", "w") as f:
        for key, value in all_results.items():
            f.write(f"### {key} ###\n")
            f.write(value.to_string() + "\n\n")

    # all_results_df.to_csv("input_data/all_results_df_no_limits_2020.csv")

    print(f"Grid Performance DataFrame Shape: {grid_performance_df.shape}")
    print(f"Grid Performance DataFrame:\n{grid_performance_df}")

    # Handle case where no valid data is available
    if grid_performance_df.empty:
        print("\nNo valid data available. Ensure the input data and thresholds are correct.")
    else:
        print("\nGrid Performance DataFrame successfully created.")

    print("\n### MAIN FUNCTION COMPLETED ###")
    print(f"All results: \n{all_results}")
    return all_results, grid_performance_df


# In[ ]:


all_results, grid_performance_df = main(df_train, df_test)


# In[ ]:


grid_performance_df


# In[ ]:




def plot_heatmap(grid_performance_df, x):
    # Pivot the data to prepare for the heatmap
    heatmap_data = grid_performance_df.pivot(
        index='correlation_threshold', 
        columns='performance_threshold', 
        values= x
    )
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".4f", 
        cmap="coolwarm", 
        cbar_kws={'label': 'Average Performance'}
    )
    plt.title('Grid Search Heatmap of '+ x)
    plt.xlabel('Performance Threshold n5_d1')
    plt.ylabel('Correlation Threshold n5_d1')
    plt.savefig(x + "_heatmap.png")
    plt.show()

# Assuming grid_performance_df is already created
plot_heatmap(grid_performance_df, x = 'mean_of_mean_performance')
plot_heatmap(grid_performance_df, x = 'number_of_monthly_predictions')


# In[ ]:




# In[ ]:


# grid_performance_df.describe()


# In[ ]:


all_results


# In[ ]:





# In[ ]:




