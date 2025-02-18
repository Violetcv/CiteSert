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

file_path_d5 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version5_40krows_corpusD\input_data\filtered_data_2.csv'  # 5-day returns data for training
file_path_d1 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version5_40krows_corpusD\input_data\filtered_data_2_n1.csv'  # 1-day returns data for testing

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
def get_best_authors(df, start_date, end_date, k_percent):
    
    """
    Args:
        df (pd.DataFrame): Input DataFrame with author performance data
        start_date (str/datetime): Start date of the analysis period
        end_date (str/datetime): End date of the analysis period
        k_percent (float): Percentage of top performing authors to select (1-100)
    """

    print("\n### get_best_authors ###")
    print(f"Start Date: {start_date}, End Date: {end_date}")
    print(f"Top K%: {k_percent}%")

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for the specified date range
    period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    print(f"Filtered Data Shape: {period_df.shape}")

    if period_df.empty:
        print("No data available for the specified date range.")
        return pd.DataFrame(columns=['author', 'total_performance'])
    
    # Calculate performance for each prediction
    period_df['performance'] = np.sign(period_df['expected_return']) * period_df['actual_return']

    # Aggregate performance for each author
    author_performance = (period_df.groupby('author')
                         .agg(
                             total_performance=('performance', 'sum'),
                             number_of_rows=('performance', 'count')
                         )
                         .reset_index())
    
    # Select top K% authors overall
    n = len(author_performance)
    n_select = max(1, int(np.ceil(n * k_percent / 100)))
    top_authors_df = author_performance.nlargest(n_select, 'total_performance')

    print(f"Selected {len(top_authors_df)} authors.")
    return top_authors_df

# In[6]:

# needs to use df_test
def calculate_monthly_performance(df, best_authors_df, target_month, corpus_fraction):
    """
    Calculate performance for each author for a specific month with detailed metrics
    """
    print(f"\nCalculating monthly performance for: {target_month.strftime('%Y-%m')}")

    # Define month range
    month_start = target_month.replace(day=1)
    month_end = (month_start + pd.offsets.MonthEnd(1))


    print(f"Month Start: {month_start}, Month End: {month_end}")

    # Filter data for the target month
    mask = (df['date'] >= month_start) & (df['date'] <= month_end)
    month_df = df[mask].copy()

    # Calculate mean performance for ALL authors before filtering
    month_df['mean_performance'] = (np.sign(month_df['expected_return']) * 
                                  month_df['actual_return'])

    # Filter for selected authors
    month_df = month_df[month_df['author'].isin(best_authors_df['author'])]
    
    if month_df.empty:
        return pd.DataFrame()
    
    # Initialize corpus D as 1.0 (100%)
    corpus_D = 1.0
    daily_results = []

    # Process each day
    for date, day_data in month_df.groupby('date'):
        N = len(day_data)
        if N == 0:
            continue

        # Get mean performance for all authors on this day
        all_authors_day = df[(df['date'] == date)]
        mean_performance = (np.sign(all_authors_day['expected_return']) * 
                          all_authors_day['actual_return']).mean()

        # Calculate selected authors performance
        selected_performance = (np.sign(day_data['expected_return']) * 
                              day_data['actual_return']).mean()

        # Calculate allocation and returns
        allocation_per_analyst = (corpus_D * corpus_fraction) / N
        day_data['trade_return'] = (np.sign(day_data['expected_return']) * 
                                  day_data['actual_return'] - 0.0004) * allocation_per_analyst
        
        daily_return = day_data['trade_return'].sum()
        corpus_D += daily_return

        daily_results.append({
            'date': date,
            'daily_return': daily_return,
            'num_trades': N,
            'mean_performance': mean_performance,
            'selected_performance': selected_performance,
            'performance_difference': selected_performance - mean_performance,
            'corpus_value': corpus_D
        })

    # Create summary DataFrame
    if daily_results:
        daily_df = pd.DataFrame(daily_results)
        
        performance_summary = pd.DataFrame({
            'month': [month_start],
            'final_corpus_value': [corpus_D],
            'total_return': [corpus_D - 1.0],
            'num_trades': [daily_df['num_trades'].sum()],
            'avg_daily_return': [daily_df['daily_return'].mean()],
            'mean_performance': [daily_df['mean_performance'].mean()],
            'selected_performance': [daily_df['selected_performance'].mean()],
            'avg_performance_difference': [daily_df['performance_difference'].mean()]
        })
        
        return performance_summary

    return pd.DataFrame()

def run_rolling_analysis(df_train, df_test, start_date, end_date, k_percent, lookback_period, corpus_fraction):
    """
    Run the rolling analysis month by month, selecting top K% authors
    """
    # Implementation remains largely the same, just pass k_percent instead of a, b
    # to get_best_authors
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

        # Calculate lookback period with decimal months
        full_months = int(lookback_period)
        partial_month_days = int((lookback_period % 1) * 30)  # Assuming 30 days per month
        
        # Calculate training start date using both full months and additional days
        training_start = (training_end - pd.DateOffset(months=full_months) - 
                         pd.DateOffset(days=partial_month_days))    
        
        print(f"Training Period: Start {training_start}, End {training_end}")

        # Get best authors; df_train
        best_authors = get_best_authors(df_train, training_start, training_end, k_percent)
        print(f"Best Authors DataFrame Shape: {best_authors.shape}")

        # Calculate performance for the current month; df_test
        if not best_authors.empty:
            monthly_perf = calculate_monthly_performance(df_test, best_authors, current_date, corpus_fraction)
            if not monthly_perf.empty:
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
    
    # Define start and end dates
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    
    # Calculate number of months between start and end dates
    num_months = (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year) * 12 + \
                 (pd.to_datetime(end_date).month - pd.to_datetime(start_date).month) + 1
    
    # Define the K% values to test
    k_percentages = np.arange(1, 25.5, 0.5).tolist()
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
                start_date=start_date,
                end_date=end_date,
                lookback_period=14,
                k_percent=k,
                corpus_fraction=0.8  # Add this parameter
            )
            
            # Store results
            iteration_key = f"iter_{iteration}_k_{k}"
            all_results[iteration_key] = results_df
            
            # Always append results to iteration_results, even if empty
            if not results_df.empty and 'total_return' in results_df.columns:
                iteration_results.append({
                    'k_percent': k,
                    'avg_monthly_return': results_df['total_return'].mean(),
                    'avg_trades_per_month': results_df['num_trades'].mean(),
                    'final_corpus_value': results_df['final_corpus_value'].iloc[-1],
                    'mean_performance': results_df['mean_performance'].mean(),
                    'selected_performance': results_df['selected_performance'].mean(),
                    'avg_performance_difference': results_df['avg_performance_difference'].mean()
                })
                
                # Save detailed monthly results
                results_df.to_csv(f"./detailed_results/k_{k}_monthly_results.csv", index=False)
            else:
                iteration_results.append({
                    'k_percent': k,
                    'avg_monthly_return': np.nan,
                    'avg_trades_per_month': np.nan,
                    'final_corpus_value': np.nan,
                    'mean_performance': np.nan  # Add mean performance
                })
            
        except Exception as e:
            print(f"Error during analysis for k={k}: {e}")
            # Still append the k value with NaN results
            iteration_results.append({
                'k_percent': k,
                'avg_monthly_return': np.nan,
                'avg_trades_per_month': np.nan,
                'final_corpus_value': np.nan
            })
        
        iteration += 1
    
    # Convert iteration results to a DataFrame for analysis
    print("\nConverting iteration results to a DataFrame for final analysis.")
    results_df = pd.DataFrame(iteration_results)
    
    # Save results
    results_df.to_csv("./input_data/corpusD_performance_results.csv", index=False)
    
    # Save detailed results
    with open("./input_data/corpusD_performance_results.txt", "w") as f:
        for key, value in all_results.items():
            f.write(f"### {key} ###\n")
            f.write(value.to_string() + "\n\n")

    print("\n### MAIN FUNCTION COMPLETED ###")
    return all_results, results_df

all_results, results_df = main(df_d1, df_d1)

def plot_performance_metrics(results_df):
    """
    Generate comprehensive plots for all available performance metrics
    """
    import os
    # Create a directory for plots if it doesn't exist
    os.makedirs('performance_plots', exist_ok=True)
    
    # Set style for better-looking plots
    plt.style.use('default')  # Using default style instead of seaborn
    
    # Set general plotting parameters
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.grid'] = True
    
    # 1. Combined Performance Metrics
    plt.figure()
    metrics = ['avg_monthly_return', 'mean_performance', 'selected_performance', 
               'avg_performance_difference']
    
    colors = ['blue', 'red', 'green', 'purple']
    for metric, color in zip(metrics, colors):
        if metric in results_df.columns:
            plt.plot(results_df['k_percent'], results_df[metric], 
                    marker='o', label=metric.replace('_', ' ').title(),
                    color=color)
    
    plt.title('Performance Metrics by K%', fontsize=12, pad=20)
    plt.xlabel('Top K Percent', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('performance_plots/all_performance_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Corpus Value and Trading Volume
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    ax1.plot(results_df['k_percent'], results_df['final_corpus_value'], 
             color='green', marker='s')
    ax1.set_title('Final Corpus Value by K%', fontsize=12, pad=20)
    ax1.set_xlabel('Top K Percent', fontsize=10)
    ax1.set_ylabel('Corpus Value', fontsize=10)

    ax2.plot(results_df['k_percent'], results_df['avg_trades_per_month'], 
             color='orange', marker='^')
    ax2.set_title('Average Trading Volume by K%', fontsize=12, pad=20)
    ax2.set_xlabel('Top K Percent', fontsize=10)
    ax2.set_ylabel('Average Trades per Month', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('performance_plots/corpus_and_volume.png', dpi=300)
    plt.close()

    # 3. Correlation Matrix Heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = results_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True)
    plt.title('Correlation Matrix of Performance Metrics', fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig('performance_plots/correlation_matrix.png', dpi=300)
    plt.close()

    # 4. Performance Differences
    plt.figure()
    if 'selected_performance' in results_df.columns and 'mean_performance' in results_df.columns:
        plt.plot(results_df['k_percent'], 
                results_df['selected_performance'] - results_df['mean_performance'],
                color='purple', marker='o')
        plt.title('Performance Difference (Selected - Mean) by K%', fontsize=12, pad=20)
        plt.xlabel('Top K Percent', fontsize=10)
        plt.ylabel('Performance Difference', fontsize=10)
        plt.tight_layout()
        plt.savefig('performance_plots/performance_difference.png', dpi=300)
        plt.close()

    # 5. Rolling Statistics
    window = 5  # Rolling window size
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    
    metrics_to_roll = ['avg_monthly_return', 'mean_performance', 
                      'selected_performance', 'avg_trades_per_month']
    
    for i, (metric, ax) in enumerate(zip(metrics_to_roll, axes)):
        if metric in results_df.columns:
            rolling_mean = results_df[metric].rolling(window=window).mean()
            rolling_std = results_df[metric].rolling(window=window).std()
            
            ax.plot(results_df['k_percent'], rolling_mean, 
                   label=f'Rolling Mean ({window})', color='blue')
            ax.fill_between(results_df['k_percent'], 
                          rolling_mean - rolling_std,
                          rolling_mean + rolling_std,
                          alpha=0.2, color='blue')
            ax.set_title(f'Rolling Statistics: {metric.replace("_", " ").title()}',
                        fontsize=12, pad=20)
            ax.set_xlabel('Top K Percent', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('performance_plots/rolling_statistics.png', dpi=300)
    plt.close()

    print("\nEnhanced plots have been saved in the 'performance_plots' directory.")    
# Call the function
plot_performance_metrics(results_df)