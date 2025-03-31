# %% [markdown]
# # Import librarires

# %%
import pandas as pd
import numpy as np
import re

# %% [markdown]
# # Load Data

# %%
stock_df = pd.read_csv('data/adjusted_stock_data.csv')
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df.head()

# %%
dividend_rows =  pd.read_csv('data/dividend_rows.csv')
dividend_rows['Date'] = pd.to_datetime(dividend_rows['Date'])
dividend_rows.head()

# %%
analyst_ratings =  pd.read_csv('data/2000_analysts.csv')
analyst_ratings['date'] = pd.to_datetime(analyst_ratings['date'])
analyst_ratings.info()

# %% [markdown]
# # Data Exploration

# %%
stock_df.columns

# %%
stock_df.isna().sum()

# %%
dividend_rows.columns

# %%
dividend_rows.isna().sum()

# %%
unique_tickers = dividend_rows['Ticker'].unique() 
unique_tickers

# %%
# Check if the symbols and dates match between datasets - company tickers
common_symbols = set(stock_df['Symbol']).intersection(set(dividend_rows['Symbol']))
print(f"Number of common symbols: {len(common_symbols)}")

# %%
analyst_ratings.columns

# %%
analyst_ratings.isna().sum()

# %% [markdown]
# # Data Preprocessing

# %%
# Define a function to clean author names
def clean_author_name(author):
    # Remove any newline characters and unwanted suffixes
    return re.sub(r'\n.*', '', author).strip()

# Apply the cleaning function to the 'author' column using .loc
analyst_ratings.loc[:, 'author'] = analyst_ratings['author'].apply(clean_author_name)

# Display the cleaned unique author names
print(analyst_ratings['author'].unique())

# %%
stock_df = stock_df.drop(['Volume', 'Market capitalization as on March 28, 2024\n(In lakhs)'], axis=1)

# %%
def clean_price_at_reco(price):
    percentage = None  # Initialize percentage as None

    # Check if the price is a string
    if isinstance(price, str):
        # Split the string by newline
        parts = price.split('\n')
        
        # Check if we have at least two parts
        if len(parts) >= 2:
            # Take the first part as price and second as percentage
            price = parts[0]
            percentage = parts[1]
            
        # Attempt to convert to numeric, ignoring any errors
        price = pd.to_numeric(price, errors='coerce')
    
    return price, percentage
    
# Apply the cleaning function to the 'price_at_reco' column and collect results
analyst_ratings[['cleaned_price_at_reco', 'percentage']] = analyst_ratings['price_at_reco'].apply(
    lambda x: pd.Series(clean_price_at_reco(x))
)

# Print the cleaned DataFrame
analyst_ratings[['price_at_reco', 'cleaned_price_at_reco', 'percentage']]

# %%
# Clean 'price_at_reco' and 'target' to extract numeric values
analyst_ratings['cleaned_price_at_reco'] = analyst_ratings['cleaned_price_at_reco'].astype(float)
analyst_ratings['target'] = analyst_ratings['target'].astype(float)

# %%
#difference between target and price at recommendation - which defines upside% whether target was met or not
analyst_ratings[['target','cleaned_price_at_reco']]

# %% [markdown]
# # Output Variable 1

# %%
# Calculate expected return
analyst_ratings['expected_return'] = ((analyst_ratings['target'] - analyst_ratings['cleaned_price_at_reco']) / analyst_ratings['cleaned_price_at_reco']) * 100

# %%
analyst_ratings.head()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm

# Filter the DataFrame for the date range
df_2019 = stock_df[(stock_df['Date'] >= '1900-01-01') & (stock_df['Date'] <= '2024-12-31')]

# Prepare company names for TF-IDF matching
company_names = [x.lower() for x in stock_df['Company Name'].unique().tolist()]
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)  # Character-based TF-IDF
tfidf_matrix = vectorizer.fit_transform(company_names)

print("Matrix shape", tfidf_matrix.shape)

# Function to get the best match using cosine similarity
def get_closest_company_name(input_name, company_names, tfidf_matrix, vectorizer):
    input_tfidf = vectorizer.transform([input_name.lower()])
    cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()
    best_match_idx = cosine_similarities.argmax()
    best_match_score = cosine_similarities[best_match_idx]
    return company_names[best_match_idx] if best_match_score > 0.5 else None  # Threshold of 0.5 for similarity

# Apply TF-IDF matching with tqdm for progress visualization
tqdm.pandas()  # Initialize tqdm with pandas
analyst_ratings['matched_company_name'] = analyst_ratings['stock_name'].progress_apply(
    lambda x: get_closest_company_name(x, company_names, tfidf_matrix, vectorizer)
)

# Create a mapping of matched company names to symbols
symbol_map = stock_df[['Company Name', 'Symbol']].drop_duplicates()
symbol_map['Company Name'] = symbol_map['Company Name'].str.lower()
analyst_ratings = pd.merge(analyst_ratings, symbol_map, left_on='matched_company_name', right_on='Company Name', how='left')

# %%
print(analyst_ratings)

# %% [markdown]
# ## Import *predict_n_day_return* Function

# %%
import sys
import os

# Get the current directory of the notebook
current_dir = os.path.dirname(os.path.abspath("__file__"))

# Append the directory of script
sys.path.append(os.path.join(current_dir, 'Project-1'))

# call predict_n_day_return function
from nday_stockprediction import predict_n_day_return

# %%
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

n_days = 2  # Set the number of days for predicting return

# Function to get the nearest available date in stock data if the exact date is missing
def get_nearest_date(stock_data, reference_date, days=3):
    stock_data = stock_data.copy()
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    reference_date = pd.to_datetime(reference_date)
    
    # Define a search range of ±days around the reference_date
    start_date = reference_date - timedelta(days=days)
    end_date = reference_date + timedelta(days=days)
    
    # Filter stock_data within the search range
    filtered_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
    
    if reference_date in filtered_data['Date'].values:
        return reference_date  # Exact match found within the range
    else:
        # Find the nearest available date within the filtered range
        if not filtered_data.empty:
            closest_date = filtered_data.loc[filtered_data['Date'].sub(reference_date).abs().idxmin(), 'Date']
            return closest_date
        else:
            return None  # No dates available within the range

# Loop through each row of the analyst ratings with a progress bar
for index, row in tqdm(analyst_ratings.iterrows(), total=analyst_ratings.shape[0], desc="Processing"):
    matched_company = row['matched_company_name']  # Refer to the matched company name in the dataframe
    
    if matched_company:
        # Get the ticker corresponding to the matched company name
        matched_ticker = stock_df[stock_df['Company Name'].str.lower() == matched_company]['Symbol'].iloc[0]
    else:
        matched_ticker = None  # No matching company found
    
    
    # Initialize nearest_date outside the condition to handle cases where matched_ticker is None
    nearest_date = None
    
    # Only call the function if a ticker was found
    if matched_ticker:
        try:
            # Get stock data for the matched ticker
            stock_data = stock_df.loc[stock_df['Symbol'] == matched_ticker].copy()  # Use .copy() to avoid warnings
            
            # Get the nearest available date if the reference date is not found
            nearest_date = get_nearest_date(stock_data, row['date'])
            
            if nearest_date is not None:
                # Call the prediction function with the nearest available date
                _, _, total_return, _, _, _ = predict_n_day_return(matched_ticker, nearest_date, n_days, stock_df, dividend_rows)
                analyst_ratings.loc[index, 'actual_return'] = total_return  # Use .loc to set the value
            else:
                analyst_ratings.loc[index, 'actual_return'] = None  # No nearest date found

        except Exception as e:
            analyst_ratings.loc[index, 'actual_return'] = None  # Store None if there's an error
            tqdm.write(f"Error for {matched_ticker} on {row['date']}: {e}")
    else:
        analyst_ratings.loc[index, 'actual_return'] = None  # No ticker found, store None
    
    # Print a warning if the nearest date is not the same as the reference date and is defined
    if nearest_date is not None and nearest_date != row['date']:
        tqdm.write(f"Reference date {row['date']} not found. Using nearest date: {nearest_date}")


# %%
industry_keywords = {
    'Finance': ['bank', 'financial', 'insurance', 'capital', 'investment', 'wealth', 'lending', 'asset management', 'mutual fund', 'securities', 'credit', 'equity', 'stock exchange', 'private equity', 'venture capital'],
    
    'Technology': ['tech', 'software', 'IT', 'digital', 'cloud', 'data', 'artificial intelligence', 'AI', 'machine learning', 'hardware', 'blockchain', 'cybersecurity', 'semiconductor', 'saas', 'paas', 'technology services', 'internet', 'networking'],
    
    'Healthcare': ['pharma', 'biotech', 'medical', 'healthcare', 'hospital', 'clinical', 'diagnostics', 'life sciences', 'medtech', 'wellness', 'drug', 'therapy', 'genomics', 'vaccines', 'biopharma'],
    
    'Manufacturing': ['steel', 'cement', 'construction', 'engineering', 'manufacturing', 'machinery', 'industrial', 'fabrication', 'metals', 'automotive', 'tools', 'aerospace', 'assembly', 'chemicals', 'plastics', 'production'],
    
    'Energy': ['oil', 'energy', 'gas', 'power', 'coal', 'electricity', 'solar', 'renewable', 'wind', 'hydro', 'nuclear', 'fossil fuels', 'mining', 'petroleum', 'green energy', 'battery'],
    
    'Consumer Goods': ['consumer', 'retail', 'fashion', 'apparel', 'food', 'beverage', 'fmcg', 'personal care', 'household', 'luxury', 'ecommerce', 'cosmetics', 'home appliances', 'groceries', 'sportswear', 'furniture'],
    
    'Telecommunications': ['telecom', 'wireless', 'broadband', 'satellite', 'mobile', 'network', 'fiber', 'telecommunication services', 'cellular', '5G', 'communication', 'ISP'],
    
    'Utilities': ['water', 'electric', 'gas distribution', 'utilities', 'waste management', 'public services', 'sanitation', 'power grid'],
    
    'Real Estate': ['real estate', 'property', 'construction', 'housing', 'commercial real estate', 'residential', 'mortgage', 'land development', 'realty', 'leasing', 'brokerage', 'facility management', 'rental'],
    
    'Transportation': ['logistics', 'shipping', 'transportation', 'rail', 'trucking', 'freight', 'airlines', 'aviation', 'maritime', 'delivery', 'supply chain', 'public transport', 'cargo', 'transport services'],
    
    'Media & Entertainment': ['media', 'entertainment', 'film', 'television', 'music', 'broadcast', 'news', 'publishing', 'streaming', 'advertising', 'social media', 'gaming', 'content creation'],
    
    'Agriculture': ['agriculture', 'farming', 'crop', 'livestock', 'dairy', 'agribusiness', 'fertilizers', 'agrochemicals', 'forestry', 'horticulture', 'fisheries', 'seed', 'organic farming'],
    
    'Retail': ['retail', 'wholesale', 'department store', 'grocery', 'mall', 'shopping', 'online store', 'boutique', 'supermarket', 'convenience store']
}

# %%
from tqdm import tqdm

# Initialize tqdm for pandas
tqdm.pandas()

# Function to identify sector based on description
def identify_sector(description):
    description = description.lower()  # Convert to lowercase for case-insensitive matching
    for sector, keywords in industry_keywords.items():
        if any(keyword in description for keyword in keywords):
            return sector
    return 'Unknown'  # Return 'Unknown' if no match is found

# Create a new DataFrame with unique companies based on 'Symbol'
unique_companies_df = stock_df.drop_duplicates(subset=['Symbol']).copy()

# Apply sector identification on the unique companies
unique_companies_df['Sector'] = unique_companies_df['Description'].progress_apply(identify_sector)

# Display the updated DataFrame with unique entries and their sector
print(unique_companies_df[['Symbol', 'Company Name', 'Description', 'Sector']])

# If needed, you can save this to a new DataFrame, e.g., `unique_sector_df`, without affecting `stock_df`
unique_sector_df = unique_companies_df[['Symbol', 'Company Name', 'Description', 'Sector']].copy()


# %%
# Ensure the DataFrame with unique sectors based on symbols is saved as 'unique_sectors_df'
# Assuming 'Symbol' is the column name in both DataFrames

# Perform an inner join on 'Symbol' column
merged_df = pd.merge(analyst_ratings, unique_sector_df[['Symbol', 'Sector']], on='Symbol', how='inner')

# Display the merged DataFrame to confirm
print(merged_df.head())

# %%
# Select only the desired columns from merged_df
filtered_df = merged_df[['date', 'Company Name', 'Symbol', 'author', 'Sector', 'expected_return', 'actual_return']]

# Display the filtered DataFrame to verify
print(filtered_df.head())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Group by 'Sector' and aggregate company names
sector_group_details = filtered_df.groupby('Sector').agg({
    'Company Name': lambda x: ', '.join(x.unique()),  # Combine company names
    'Symbol': 'count'  # Count the number of companies in each sector
}).reset_index()

# Rename the 'Symbol' column to 'Company Count' for clarity
sector_group_details.rename(columns={'Symbol': 'Company Count'}, inplace=True)

# Display the sector details
print(sector_group_details)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Group by 'Sector' and aggregate company names
sector_group_details = filtered_df.groupby('Sector').agg({
    'Company Name': lambda x: ', '.join(x.unique()),  # Combine company names
    'Symbol': 'count'  # Count the number of companies in each sector
}).reset_index()

# Rename the 'Symbol' column to 'Company Count' for clarity
sector_group_details.rename(columns={'Symbol': 'Company Count'}, inplace=True)

# Display the sector details
print(sector_group_details)

# %% [markdown]
# # Output Dataframe - Final

# %%
# Save filtered_df as a CSV file
filtered_df.to_csv(f'data/filtered_data_{n_days}.csv', index=False)

# %%


# %%


# %%


# %%



