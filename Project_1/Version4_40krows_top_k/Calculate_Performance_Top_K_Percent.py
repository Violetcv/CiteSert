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

file_path_d5 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version4_40krows_top_k\input_data\filtered_data_2.csv'  # 5-day returns data for training
file_path_d1 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version4_40krows_top_k\input_data\filtered_data_2_n1.csv'  # 1-day returns data for testing

df_d5 = pd.read_csv(file_path_d5)
df_d1 = pd.read_csv(file_path_d1)

df_d5 = df_d5.dropna(subset=['expected_return', 'actual_return'])
df_d1 = df_d1.dropna(subset=['expected_return', 'actual_return'])

# Display the first few rows of the DataFrame to verify it loaded correctly
print(df_d5.head())
print(df_d1.head())


# In[3]:

df_d5.info()
df_d1.info()


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
def get_best_authors_by_sector(df, start_date, end_date, k_percent):
    """
    Identify the best-performing authors per sector within a specific date range based on 
    their total performance over the period, selecting top K% authors.
    
    Args:
        df (pd.DataFrame): Input DataFrame with author performance data
        start_date (str/datetime): Start date of the analysis period
        end_date (str/datetime): End date of the analysis period
        k_percent (float): Percentage of top performing authors to select (1-100)
    """
    
    print("\n### get_best_authors_by_sector ###")
    print(f"Start Date: {start_date}, End Date: {end_date}")
    print(f"Top K%: {k_percent}%")

    # Convert date column to datetime if necessary
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for the specified date range
    period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    print(f"Filtered Data Shape: {period_df.shape}")

    if period_df.empty:
        print("No data available for the specified date range.")
        return pd.DataFrame(columns=['Sector', 'author', 'total_performance'])

    # Calculate performance for each prediction
    period_df['performance'] = np.sign(period_df['expected_return']) * period_df['actual_return']
    # period_df['performance'] = abs(period_df['actual_return']) * np.sign(period_df['expected_return'])



    # Calculate total performance for each author-sector pair
    author_performance = (period_df.groupby(['Sector', 'author'])
                         .agg(
                             total_performance=('performance', 'sum'),
                             number_of_rows=('performance', 'count')
                         )
                         .reset_index())
    
    # Function to select top K% authors for each sector
    def select_top_k_percent(group, k):
        n = len(group)
        n_select = max(1, int(np.ceil(n * k / 100)))  # At least 1 author
        return group.nlargest(n_select, 'total_performance')
    
    # Select top K% authors for each sector
    top_authors_df = (author_performance
                     .groupby('Sector', group_keys=False)
                     .apply(lambda x: select_top_k_percent(x, k_percent))
                     .reset_index(drop=True))
    
    print(f"Selected {len(top_authors_df)} authors across all sectors")
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
    # month_df['performance'] = abs(month_df['actual_return']) * np.sign(month_df['expected_return'])


    print(f"Performance Column Calculated. Sample Data:\n{month_df[['Sector', 'author', 'performance']]}")

    # Group and summarize performance
    performance_summary = (month_df.groupby(['Sector', 'author'])
                           ['performance']
                           .agg(mean_performance='mean', count='count')
                           .reset_index())
    performance_summary['month'] = month_start
    # print(f"Performance Summary for {target_month.strftime('%Y-%m')}:\n{performance_summary}")

    return performance_summary

def run_rolling_analysis(df_train, df_test, start_date, end_date, k_percent, lookback_period=12):
    """
    Run the rolling analysis month by month, selecting top K% authors
    """
    # Implementation remains largely the same, just pass k_percent instead of a, b
    # to get_best_authors_by_sector
    print("\nStarting Rolling Analysis")
    print(f"Start Date: {start_date}, End Date: {end_date}, Lookback Period: {lookback_period} months")
    print(f"Top K%: {k_percent}")

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
        best_authors = get_best_authors_by_sector(df_train, training_start, training_end, k_percent)
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

    
    # Define the K% values to test
    k_percentages = [1, 2, 5, 10, 15, 20, 25, 50, 75, 100]
    print(f"Testing K percentages: {k_percentages}")

    
    # Initialize a dictionary to store results for each iteration
    all_results = {}
    iteration_results = []
    iteration = 0

    print("\n### Starting K% analysis  ###")
    
   # Iterate over different K% values
    for k in tqdm(k_percentages, desc="Testing K percentages"):
        print(f"\nAnalyzing for top {k}% authors")
        
        try:
            results_df = run_rolling_analysis(
                df_train,
                df_test,
                start_date='2014-01-01',
                end_date='2022-12-31',
                lookback_period=12,
                k_percent=k
            )
            print(f"Analysis Completed. Results Shape: {results_df.shape}")
            
        except Exception as e:
            print(f"Error during analysis for k={k}: {e}")
            results_df = pd.DataFrame()
        
        # Calculate metrics
        if not results_df.empty and 'mean_performance' in results_df.columns:
            mean_of_mean_performance = results_df['mean_performance'].mean()
            number_of_monthly_predictions = results_df['count'].sum()/108
            
            print(f"Mean Performance: {mean_of_mean_performance:.4f}")
            print(f"Average Monthly Predictions: {number_of_monthly_predictions:.2f}")
        else:
            mean_of_mean_performance = np.nan
            number_of_monthly_predictions = np.nan
        
        # Store results
        iteration_key = f"iter_{iteration}_k_{k}"
        all_results[iteration_key] = results_df
        
        iteration_results.append({
            'k_percent': k,
            'mean_of_mean_performance': mean_of_mean_performance,
            'number_of_monthly_predictions': number_of_monthly_predictions
        })
        
        iteration += 1
    
    # Convert iteration results to a DataFrame for analysis
    print("\nConverting iteration results to a DataFrame for final analysis.")
    results_df = pd.DataFrame(iteration_results)
    results_df.to_csv("input_data/k_percent_performance_results.csv")

    # If you want to save the dictionary as a simple text file:
    with open("input_data/k_percent_performance_results.txt", "w") as f:
        for key, value in all_results.items():
            f.write(f"### {key} ###\n")
            f.write(value.to_string() + "\n\n")


    print("\n### MAIN FUNCTION COMPLETED ###")
    print(f"All results: \n{all_results}")
    return all_results, results_df


# In[ ]:


all_results, results_df = main(df_d5, df_d5)


# In[ ]:


results_df


# In[ ]:




# def plot_heatmap(grid_performance_df, x):
#     # Pivot the data to prepare for the heatmap
#     heatmap_data = grid_performance_df.pivot(
#         index='correlation_threshold', 
#         columns='performance_threshold', 
#         values= x
#     )
    
#     # Plot the heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         heatmap_data, 
#         annot=True, 
#         fmt=".4f", 
#         cmap="coolwarm", 
#         cbar_kws={'label': 'Average Performance'}
#     )
#     plt.title('Grid Search Heatmap of '+ x)
#     plt.xlabel('Performance Threshold n5_d1')
#     plt.ylabel('Correlation Threshold n5_d1')
#     plt.savefig(x + "_heatmap.png")
#     plt.show()

# # Assuming grid_performance_df is already created
# plot_heatmap(results_df, x = 'mean_of_mean_performance')
# plot_heatmap(results_df, x = 'number_of_monthly_predictions')


all_results


def plot_results(results_df):
    """
    Plot the results of K% analysis showing mean performance and number of predictions
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot mean performance
    ax1.plot(results_df['k_percent'], results_df['mean_of_mean_performance'], 
             marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Top K%')
    ax1.set_ylabel('Mean Performance')
    ax1.set_title('Mean Performance by K%')
    ax1.grid(True)
    
    # Plot number of monthly predictions
    ax2.plot(results_df['k_percent'], results_df['number_of_monthly_predictions'], 
             marker='o', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Top K%')
    ax2.set_ylabel('Average Monthly Predictions')
    ax2.set_title('Average Monthly Predictions by K%')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("k_percent_analysis_results.png")
    plt.show()

# Replace the heatmap plotting with this new function
plot_results(results_df)


def plot_results(grid_performance_df):
    """
    Plot line graphs for analysis results
    """
    plt.figure(figsize=(12, 6))
    
    # Plot both metrics on the same graph with different y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot mean performance on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('K%')
    ax1.set_ylabel('Mean Performance', color=color)
    line1 = ax1.plot(grid_performance_df['k_percent'], 
                     grid_performance_df['mean_of_mean_performance'], 
                     color=color, 
                     marker='o',
                     label='Mean Performance')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create secondary y-axis for monthly predictions
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Average Monthly Predictions', color=color)
    line2 = ax2.plot(grid_performance_df['k_percent'], 
                     grid_performance_df['number_of_monthly_predictions'], 
                     color=color, 
                     marker='s',
                     label='Monthly Predictions')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and adjust layout
    plt.title('Performance Metrics by K%')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig("k_percent_analysis_line_graph.png")
    plt.show()

# Call the function with your results DataFrame
plot_results(results_df)