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
    Calculate performance for each author for a specific month
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

    month_df = month_df[month_df['author'].isin(best_authors_df['author'])]
    
    if month_df.empty:
        return pd.DataFrame()
    
    # Initialize corpus D as 1.0 (100%)
    corpus_D = 1.0
    daily_returns = []

    # Group by date to process each day's trades
    for date, day_data in month_df.groupby('date'):
        N = len(day_data)  # Number of analysts for the day
        if N == 0:
            continue

        # Get mean performance for all authors on this day
        all_authors_day = df[(df['date'] == date)]
        mean_performance = (np.sign(all_authors_day['expected_return']) * 
                          all_authors_day['actual_return']).mean()

        # Calculate selected authors performance
        selected_performance = (np.sign(day_data['expected_return']) * 
                              day_data['actual_return']).mean()
        
        # Calculate allocation per analyst
        allocation_per_analyst = (corpus_D * corpus_fraction) / N

        # Calculate daily returns
        day_data['trade_return'] = (np.sign(day_data['expected_return']) * 
                                  (day_data['actual_return'] * allocation_per_analyst) - 0.0004) 


        daily_return = day_data['trade_return'].sum()

        daily_returns.append({
            'date': date,
            'daily_return': daily_return,
            'num_trades': N
        })

        # Add validation checks here
        try:
            assert corpus_D > 0, f"Corpus value should be positive, got {corpus_D}"
            assert abs(daily_return) <= corpus_D * corpus_fraction, \
                f"Daily return ({daily_return}) exceeds allocation ({corpus_D * corpus_fraction})"
            if abs(mean_performance) > 0.0001:  # Avoid division by very small numbers
                performance_ratio = daily_return / mean_performance
                print(f"Return to Performance Ratio: {performance_ratio:.6f}")
            
            print(f"Date: {date}")
            print(f"Corpus: {corpus_D:.6f}")
            print(f"Daily Return: {daily_return:.6f}")
            print(f"Mean Performance: {mean_performance:.6f}")
            print(f"Selected Performance: {selected_performance:.6f}")
            print("-" * 50)
            
        except AssertionError as e:
            print(f"Validation failed for date {date}: {str(e)}")
            print(f"Current state:")
            print(f"Corpus D: {corpus_D}")
            print(f"Daily Return: {daily_return}")
            print(f"Mean Performance: {mean_performance}")
            print(f"Selected Performance: {selected_performance}")
            print(f"Number of trades: {N}")
            raise  # Re-raise the exception after printing debug info

        # Update corpus value
        corpus_D += daily_return

    # Calculate mean performance for all trades in the month
    month_df['performance'] = np.sign(month_df['expected_return']) * month_df['actual_return']
    mean_performance = month_df['performance'].mean()

    # Create summary DataFrame
    if daily_returns:
        daily_returns_df = pd.DataFrame(daily_returns)
        
        performance_summary = pd.DataFrame({
            'month': [month_start],
            'final_corpus_value': [corpus_D],
            'total_return': [corpus_D - 1.0],
            'num_trades': [daily_returns_df['num_trades'].sum()],
            'avg_daily_return': [daily_returns_df['daily_return'].mean()],
            'mean_performance': [mean_performance]  # Add mean performance
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
        print(f"\nCombined Performance DataFrame: {combined_results}")
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
                    'mean_performance': results_df['mean_performance'].mean()  # Add mean performance
                })
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

import os
def analyze_december_2021(df_train, df_test, k_percent=5.0):
    """
    Detailed analysis of December 2021 with daily returns comparison
    """
    print("\n=== Starting analyze_december_2021 function ===")
    print(f"Input parameters: k_percent = {k_percent}")
    print(f"Initial df_train shape: {df_train.shape}")
    print(f"Initial df_test shape: {df_test.shape}")
    
    # Convert dates
    print("\nConverting dates...")
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    print("Dates converted successfully")
    
    # Define analysis period
    print("\nDefining analysis period...")
    target_month = pd.to_datetime('2021-12-01')
    training_end = target_month - pd.Timedelta(days=1)
    training_start = training_end - pd.DateOffset(months=14)
    print(f"Target month: {target_month}")
    print(f"Training period: {training_start} to {training_end}")
    
    # Get best authors
    print("\nGetting best authors...")
    best_authors = get_best_authors(df_train, training_start, training_end, k_percent)
    print(f"Best authors identified: {best_authors}")
    
    # Filter December 2021 data
    print("\nFiltering December 2021 data...")
    dec_data = df_test[
        (df_test['date'].dt.year == 2021) & 
        (df_test['date'].dt.month == 12)
    ].copy()
    print(f"December 2021 data: {dec_data}")
    
    # Create output directory
    print("\nCreating output directory...")
    output_dir = f"december_2021_analysis_{k_percent}%"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    with open(f"{output_dir}/daily_analysis.txt", "w") as f:
        f.write(f"=== December 2021 Analysis (Top {k_percent}% Authors) ===\n\n")
        
        daily_results = []
        corpus_D = 1.0
        print(f"\nInitial corpus value: {corpus_D}")
        
        # Analyze each day
        unique_dates = sorted(dec_data['date'].unique())
        print(f"\nAnalyzing {len(unique_dates)} trading days...")
        
        for date in unique_dates:
            print(f"\nProcessing date: {date.strftime('%Y-%m-%d')}")
            f.write(f"\nDate: {date.strftime('%Y-%m-%d')}\n")
            f.write("-" * 50 + "\n")
            
            # Get data for this day
            day_data = dec_data[dec_data['date'] == date]
            print(f"Number of trades on this day: {len(day_data)}")
            
            
            # Calculate metrics for all authors
            day_data['mean_performance'] = (np.sign(day_data['expected_return']) * 
                              day_data['actual_return'])
            print(f"day data: {day_data}")


            mean_performance = day_data['mean_performance'].mean()
            print(f"Mean performance for all authors: {mean_performance:.6f}")
            
            

            # Filter for best authors
            selected_authors_data = day_data[day_data['author'].isin(best_authors['author'])]
            N = len(selected_authors_data)
            print(f"Selected authors trading today: {selected_authors_data}")
            
            
            
            if N > 0:
                # Initialize corpus tracking for this day
                print(f"\nCorpus value at start of day: {corpus_D:.6f}")

                # Calculate allocation and returns
                allocation_per_analyst = (corpus_D * 0.8) / N
                print(f"Allocation per analyst: {allocation_per_analyst:.6f}")

                # Process each trade individually
                running_corpus = corpus_D
                trade_returns = []

                for idx, trade in selected_authors_data.iterrows():
                    # Calculate individual trade return
                    trade_sign = np.sign(trade['expected_return'])
                    trade_actual = trade['actual_return']
                    trade_cost = 0.0004
                    
                    individual_return = (trade_sign * trade_actual - trade_cost) * allocation_per_analyst
                    trade_returns.append(individual_return)

                    # Update running corpus
                    running_corpus += individual_return
                    
                    # Print detailed breakdown for this trade
                    print(f"\nTrade {idx} breakdown:")
                    print(f"Author: {trade['author']}")
                    print(f"Company: {trade['Company Name']}")
                    print(f"Expected Return: {trade['expected_return']:.6f}")
                    print(f"Actual Return: {trade['actual_return']:.6f}")
                    print(f"Trade Return: {individual_return:.6f}")
                    print(f"Corpus after trade: {running_corpus:.6f}")
                
                # Update final corpus value for the day
                corpus_D = running_corpus
                
                # selected_authors_data['trade_return'] = (
                #     np.sign(selected_authors_data['expected_return']) * 
                #     selected_authors_data['actual_return'] - 0.0004
                # ) * allocation_per_analyst
                
                daily_return = sum(trade_returns)
                print(f"Daily return: {daily_return:.6f}")
                
                selected_authors_data['trade_return'] = trade_returns
                selected_authors_data['selected_authors_performance'] = np.sign(selected_authors_data['expected_return']) * selected_authors_data['actual_return']
                selected_authors_performance = selected_authors_data['selected_authors_performance'].mean()
                print(f"Selected authors performance: {selected_authors_performance:.6f}")
                
                
                print(f"Updated corpus value: {corpus_D:.6f}")
                print(f"selceted authors data : {selected_authors_data}")
                
                # Log details
                f.write(f"Number of Selected Authors Trading: {N}\n")
                f.write(f"All Authors Mean Performance: {mean_performance:.6f}\n")
                f.write(f"Selected Authors Mean Performance: {selected_authors_performance:.6f}\n")
                f.write(f"Performance Difference: {selected_authors_performance - mean_performance:.6f}\n")
                f.write(f"Daily Return: {daily_return:.6f}\n")
                f.write(f"Corpus Value: {corpus_D:.6f}\n")
                
                # Store results
                daily_results.append({
                    'date': date,
                    'num_selected_authors': N,
                    'mean_performance': mean_performance,
                    'selected_authors_performance': selected_authors_performance,
                    'performance_difference': selected_authors_performance - mean_performance,
                    'daily_return': daily_return,
                    'corpus_value': corpus_D
                })
                print("Daily results stored successfully")
            else:
                print("No trades from selected authors today")
                f.write("No trades from selected authors on this day\n")
        
        # Create summary DataFrame
        print("\nCreating summary DataFrame...")
        results_df = pd.DataFrame(daily_results)
        print(f"Results DataFrame shape: {results_df.shape}")
        
        # Calculate and write summary statistics
        print("\nCalculating summary statistics...")
        final_corpus = corpus_D
        avg_daily_return = results_df['daily_return'].mean()
        mean_performance_avg = results_df['mean_performance'].mean()
        avg_performance_diff = results_df['performance_difference'].mean()
        
        print(f"Final Corpus Value: {final_corpus:.6f}")
        print(f"Average Daily Return: {avg_daily_return:.6f}")
        print(f"Mean of Mean Performance: {mean_performance_avg:.6f}")
        print(f"Average Performance Difference: {avg_performance_diff:.6f}")
        
        f.write("\n\n=== Monthly Summary ===\n")
        f.write(f"Final Corpus Value: {final_corpus:.6f}\n")
        f.write(f"Average Daily Return: {avg_daily_return:.6f}\n")
        f.write(f"Mean of Mean Performance: {mean_performance_avg:.6f}\n")
        f.write(f"Average Performance Difference: {avg_performance_diff:.6f}\n")
        
        # Save results
        print("\nSaving results to CSV...")
        results_df.to_csv(f"{output_dir}/daily_results.csv", index=False)
        print("Results saved successfully")
        
        print("\nAnalysis completed successfully")
        return results_df
    
# Run the analysis
# december_results = analyze_december_2021(df_d1, df_d1, k_percent=100.0)

def plot(december_results, k_percent):
    # Create visualizations
    plt.figure(figsize=(15, 15))  # Increased figure size to accommodate three plots

    # Plot 1: Daily Returns and Corpus Value
    plt.subplot(3, 1, 1)  # Changed to 3 rows, 1 column
    plt.plot(december_results['date'], december_results['daily_return'], 
            label='Daily Return', marker='o')
    plt.plot(december_results['date'], december_results['corpus_value'], 
            label='Corpus Value', marker='s')
    plt.title('Daily Returns and Corpus Value - December 2021')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot 2: Performance Comparison
    plt.subplot(3, 1, 2)
    plt.plot(december_results['date'], december_results['mean_performance'], 
            label='All Authors', marker='o')
    plt.plot(december_results['date'], december_results['selected_authors_performance'], 
            label='Selected Authors', marker='s')
    plt.title('Performance Comparison - December 2021')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot 3: Daily Returns vs Mean Performance
    plt.subplot(3, 1, 3)
    plt.plot(december_results['date'], december_results['daily_return'], 
            label='Daily Return', color='blue', marker='o', linestyle='-')
    plt.plot(december_results['date'], december_results['mean_performance'], 
            label='Mean Performance', color='red', marker='s', linestyle='--')
    plt.title('Daily Returns vs Mean Performance - December 2021')
    plt.xlabel('Date')
    plt.ylabel('Return/Performance')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Add value annotations for the third plot
    for i, row in december_results.iterrows():
        plt.annotate(f'{row["daily_return"]:.4f}', 
                    (row['date'], row['daily_return']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=8)
        plt.annotate(f'{row["mean_performance"]:.4f}', 
                    (row['date'], row['mean_performance']),
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center',
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(f'december_2021_analysis_{k_percent}%/performance_comparison.png')
    plt.close()

    print("\nAnalysis completed. Check the 'december_2021_analysis' folder for detailed results.")

# plot(december_results, k_percent=100.0)

import matplotlib.pyplot as plt

def plot_performance_metrics(results_df):
    """
    Generate comprehensive plots for performance analysis including mean performance
    """
    # Create a directory for plots if it doesn't exist
    os.makedirs('performance_plots', exist_ok=True)
    
    # 1. Returns and Performance Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k_percent'], results_df['avg_monthly_return'], 
             color='blue', marker='o', label='Monthly Return')
    plt.plot(results_df['k_percent'], results_df['mean_performance'], 
             color='red', marker='s', label='Mean Performance')
    plt.title('Returns and Performance by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('performance_plots/returns_and_performance.png')
    plt.close()

    # 2. Corpus Value Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k_percent'], results_df['final_corpus_value'], 
             color='green', marker='s')
    plt.title('Final Corpus Value by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Corpus Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_plots/corpus_value.png')
    plt.close()

    # 3. Trading Volume Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k_percent'], results_df['avg_trades_per_month'], 
             color='orange', marker='^')
    plt.title('Average Trading Volume by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Average Trades per Month')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_plots/trading_volume.png')
    plt.close()

    # 4. Multi-Metric Comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Returns and Performance
    ax1.plot(results_df['k_percent'], results_df['avg_monthly_return'], 
             color='blue', marker='o', label='Monthly Return')
    ax1.plot(results_df['k_percent'], results_df['mean_performance'], 
             color='red', marker='s', label='Mean Performance')
    ax1.set_xlabel('Top K%')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Returns and Performance Metrics')
    
    # Corpus Value
    ax2.plot(results_df['k_percent'], results_df['final_corpus_value'], 
             color='green', marker='s', label='Final Corpus')
    ax2.set_xlabel('Top K%')
    ax2.set_ylabel('Corpus Value')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Corpus Value Evolution')
    
    # Trading Volume
    ax3.plot(results_df['k_percent'], results_df['avg_trades_per_month'], 
             color='orange', marker='^', label='Trading Volume')
    ax3.set_xlabel('Top K%')
    ax3.set_ylabel('Trades per Month')
    ax3.grid(True)
    ax3.legend()
    ax3.set_title('Trading Volume Distribution')
    
    plt.tight_layout()
    plt.savefig('performance_plots/combined_metrics.png')
    plt.close()

    # 5. Scatter Plots Matrix
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # Returns vs Trading Volume
    scatter1 = ax1.scatter(results_df['avg_trades_per_month'], 
                          results_df['avg_monthly_return'],
                          c=results_df['k_percent'], cmap='viridis')
    ax1.set_xlabel('Average Trades per Month')
    ax1.set_ylabel('Monthly Return')
    ax1.grid(True)
    plt.colorbar(scatter1, ax=ax1, label='K%')
    
    # Mean Performance vs Trading Volume
    scatter2 = ax2.scatter(results_df['avg_trades_per_month'], 
                          results_df['mean_performance'],
                          c=results_df['k_percent'], cmap='viridis')
    ax2.set_xlabel('Average Trades per Month')
    ax2.set_ylabel('Mean Performance')
    ax2.grid(True)
    plt.colorbar(scatter2, ax=ax2, label='K%')
    
    # Returns vs Corpus Value
    scatter3 = ax3.scatter(results_df['final_corpus_value'], 
                          results_df['avg_monthly_return'],
                          c=results_df['k_percent'], cmap='viridis')
    ax3.set_xlabel('Final Corpus Value')
    ax3.set_ylabel('Monthly Return')
    ax3.grid(True)
    plt.colorbar(scatter3, ax=ax3, label='K%')
    
    # Mean Performance vs Corpus Value
    scatter4 = ax4.scatter(results_df['final_corpus_value'], 
                          results_df['mean_performance'],
                          c=results_df['k_percent'], cmap='viridis')
    ax4.set_xlabel('Final Corpus Value')
    ax4.set_ylabel('Mean Performance')
    ax4.grid(True)
    plt.colorbar(scatter4, ax=ax4, label='K%')
    
    plt.tight_layout()
    plt.savefig('performance_plots/scatter_matrix.png')
    plt.close()

    print("\nPlots have been saved in the 'performance_plots' directory.")
    
# Call the function
# plot_performance_metrics(results_df)

def plot_all_metrics(results_df):
    """
    Plot all performance metrics separately and combined
    """
    os.makedirs('performance_plots', exist_ok=True)
    
    # Calculate combined metric
    if not results_df.empty and 'mean_performance' in results_df.columns:
        number_of_monthly_predictions = results_df['avg_trades_per_month']
        mean_of_mean_performance = results_df['mean_performance']
        combined_metric = (mean_of_mean_performance - 0.0004) * number_of_monthly_predictions
    
    # 1. Mean of Mean Performance Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k_percent'], results_df['mean_performance'], 
             color='blue', marker='o')
    plt.title('Mean of Mean Performance by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Mean Performance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_plots/mean_of_mean_performance.png')
    plt.close()

    # 2. Average Monthly Returns Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k_percent'], results_df['avg_monthly_return'], 
             color='green', marker='o')
    plt.title('Average Monthly Returns by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Average Monthly Return')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_plots/avg_monthly_returns.png')
    plt.close()

    # 3. Combined Metric Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k_percent'], combined_metric, 
             color='red', marker='o')
    plt.title('Combined Metric by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Combined Metric')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_plots/combined_metric.png')
    plt.close()

    # 4. All Metrics Combined
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['k_percent'], results_df['mean_performance'], 
             color='blue', marker='o', label='Mean of Mean Performance')
    plt.plot(results_df['k_percent'], results_df['avg_monthly_return'], 
             color='green', marker='s', label='Avg Monthly Returns')
    plt.plot(results_df['k_percent'], combined_metric, 
             color='red', marker='^', label='Combined Metric')
    
    plt.title('All Performance Metrics by K%')
    plt.xlabel('Top K Percent')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('performance_plots/all_metrics.png')
    plt.close()

    print("\nAll metrics plots have been saved in the 'performance_plots' directory.")

# Call the function
plot_all_metrics(results_df)