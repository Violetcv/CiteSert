#Function that predicts the n-day stock return based on the given parameters


import pandas as pd
import numpy as np
stocksplited_df = pd.read_csv('data/adjusted_stock_data.csv')
 
stocksplited_df['Date'] = pd.to_datetime(stocksplited_df['Date'])

# Filter the DataFrame for July 2018
df_2019 = stocksplited_df[(stocksplited_df['Date'] >= '1900-01-01') & (stocksplited_df['Date'] <= '2024-12-31')]

df_2019

dividend_rows =  pd.read_csv('data/dividend_rows.csv')
dividend_rows

 

def predict_n_day_return(ticker, date_d, n, stock_df, dividend_df):
    """
    Predict the n-day stock return for a given company, including dividends.

    Parameters:
    - ticker (str): Company ticker symbol (e.g., 'RELIANCE').
    - date_d (str): Reference date in 'YYYY-MM-DD' format.
    - n (int): Number of days for the prediction.
    - stock_df (pd.DataFrame): DataFrame containing stock data.
    - dividend_df (pd.DataFrame): DataFrame containing dividend data.

    Returns:
    - start_price (float): Open price at the start of d+1 or closest available.
    - end_price (float): Close price at the end of d+1+n or closest available.
    - total_return (float): Total return including dividends.
    - start_date_used (str): The actual date used for the start price.
    - end_date_used (str): The actual date used for the end price.
    - warnings (str): Warning messages if the exact dates were unavailable.
    """
    # Ensurr the Date columns are in datetime format
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    dividend_df['Date'] = pd.to_datetime(dividend_df['Date'])

    # Filter stock data for the given ticker and sort by Date
    ticker_stock_df = stock_df[stock_df['Symbol'] == ticker].sort_values('Date').reset_index(drop=True)
    
    # Convert the reference date to datetime
    reference_date = pd.to_datetime(date_d)
    
    
    ref_index = ticker_stock_df[ticker_stock_df['Date'] == reference_date].index
    if ref_index.empty:
        raise ValueError(f"Reference date {date_d} not found for ticker {ticker}.")
    ref_index = ref_index[0]
    
    # Determine the start index (d+1)
    start_index = ref_index + 1
    if start_index >= len(ticker_stock_df):
        raise IndexError("Insufficient data to determine d+1.")
    
    # Get the start price at d+1, or find the nearest available date
    start_row = ticker_stock_df.iloc[start_index]
    start_price = start_row['Open']
    start_date_used = start_row['Date']
    warnings = ""

    if start_date_used != reference_date + pd.Timedelta(days=1):
        warnings += f"Data not available for {reference_date + pd.Timedelta(days=1)}. Nearest available date is {start_date_used.date()}.\n"

    # Initialize dividend and total return tracking
    total_dividend = 0.0
    end_price = None  # Initiale end_price
    end_date_used = None  # Initialze the end date used

    # Loop through the next n days to find the end price and calculate dividends
    for i in range(n):
        current_index = start_index + i
        if current_index < len(ticker_stock_df):
            current_row = ticker_stock_df.iloc[current_index]
            current_date = current_row['Date']

            # Filter dividend data for the company in the current date range
            company_dividends = dividend_df[
                (dividend_df['Symbol'] == ticker) & 
                (dividend_df['Date'] >= start_date_used) & 
                (dividend_df['Date'] <= current_date)
            ]
            
            # Calculate total dividends during the period
            #total_dividend += company_dividends['Dividend'].sum() if not company_dividends.empty else 0
            total_dividend += float(company_dividends.values.tolist()[0][1].split('Dividend')[0]) if not company_dividends.empty else 0
            # Set the end price and the end date used if valid data is found
            end_price = current_row['Close']
            end_date_used = current_row['Date']
        else:
            warnings += f"Skipping date {ticker_stock_df.iloc[min(len(ticker_stock_df) - 1, current_index)]['Date'].date()} as no data is available.\n"
            break  # Exit the loop if no data is available for the current index.

    # Handle unavailable end date and notify user of the nearest date, only if end_date_used is not None
    if end_date_used is None:
        warnings += f"Data not available for {reference_date + pd.Timedelta(days=1+n)}. No valid end date found within the range.\n"
    elif end_date_used != reference_date + pd.Timedelta(days=1+n):
        warnings += f"Data not available for {reference_date + pd.Timedelta(days=1+n)}. Nearest available date is {end_date_used.date()}.\n"
    
    # Calculate the total return if end_price is available
    total_return = None
    if end_price is not None:
        total_return = ((end_price - start_price) + total_dividend) / start_price
    
    return start_price, end_price, total_return, start_date_used, end_date_used, warnings


#  testing usage:
#ticker = 'RELIANCE'
#date_d = '2019-12-24'
#n = 5

# ticker = 'RELIANCE'
# date_d = '2024-05-30'
# n = 5

 
    
# try:
#     start_price, end_price, total_return, start_date_used, end_date_used, warnings = predict_n_day_return(ticker, date_d, n, df_2019, dividend_rows)
    
#     if warnings:
#         print("Warnings:")
#         print(warnings)
    
#     print(f"Start Price on {start_date_used}: {start_price}")
    
#     if end_price is not None:
#         print(f"End Price on {end_date_used}: {end_price}")
#         print(f"Total Return over {n} days: {total_return:.2%}")
#     else:
#         print("End price data is unavailable for the given date range.")
# except (ValueError, IndexError) as e:
#     print(e)















