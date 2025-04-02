#!/usr/bin/env python3
"""
This script loads stock, dividend, and analyst ratings data
from hardcoded file paths, performs full preprocessing (cleaning,
company name matching via TF-IDF, prediction using the n-day return 
function, and industry sector identification), and then saves the
final filtered results to a CSV file.
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datetime import timedelta

# Import your prediction function and configurations.
from nday_stockprediction import predict_n_day_return
from config import DEBUG_LIMIT, DEBUG_MODE


def load_data():
    """
    Reads the stock, dividend, and analyst CSV files from hardcoded file paths.
    If DEBUG_MODE is enabled, only a limited number of rows are loaded.
    
    Returns:
       stock_df: DataFrame of stock data
       dividend_rows: DataFrame of dividend data
       analyst_ratings: DataFrame of analyst ratings
    """
    # Hardcoded file paths.
    stock_df = pd.read_csv('data/adjusted_stock_data.csv')
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    dividend_rows = pd.read_csv('data/dividend_rows.csv')
    dividend_rows['Date'] = pd.to_datetime(dividend_rows['Date'])

    analyst_ratings = pd.read_csv('data/2000_analysts.csv')
    analyst_ratings['date'] = pd.to_datetime(analyst_ratings['date'])

    if DEBUG_MODE:
        stock_df = stock_df.head(DEBUG_LIMIT)
        dividend_rows = dividend_rows.head(DEBUG_LIMIT)
        analyst_ratings = analyst_ratings.head(DEBUG_LIMIT)
        print(f"[DEBUG] Using first {DEBUG_LIMIT} rows for debugging.")

    return stock_df, dividend_rows, analyst_ratings


def preprocess_data(n_days):
    """
    Preprocesses the data by reading in hardcoded files, cleaning, matching company names
    using TF-IDF, calculating expected and predicted returns (based on n_days), and
    identifying industry sectors.

    Params:
       n_days: (int) number of days for return prediction.

    Effects:
       Saves the filtered results to a CSV and prints sector group details.
    """
    # Load the required data.
    stock_df, dividend_rows, analyst_ratings = load_data()

    # ====== Cleaning Analyst Ratings ======
    def clean_author_name(author):
        # Remove extra newline characters and unwanted suffix text.
        return re.sub(r'\n.*', '', author).strip()

    analyst_ratings.loc[:, 'author'] = analyst_ratings['author'].apply(clean_author_name)
    print(stock_df.columns)

    # Drop unwanted columns from stock data.
    stock_df = stock_df.drop(['Volume', 'Market capitalization as on March 28 2024(In lakhs)'], axis=1)

    # ====== Cleaning Price Information ======
    def clean_price_at_reco(price):
        """
        Clean the 'price_at_reco' field to separate the numeric price and percentage string.
        Returns a tuple of (price as float, percentage as string or None).
        """
        percentage = None
        if isinstance(price, str):
            parts = price.split('\n')
            if len(parts) >= 2:
                price = parts[0]
                percentage = parts[1]
            price = pd.to_numeric(price, errors='coerce')
        return price, percentage

    analyst_ratings[['cleaned_price_at_reco', 'percentage']] = analyst_ratings['price_at_reco'].apply(
        lambda x: pd.Series(clean_price_at_reco(x))
    )
    analyst_ratings['cleaned_price_at_reco'] = analyst_ratings['cleaned_price_at_reco'].astype(float)
    analyst_ratings['target'] = analyst_ratings['target'].astype(float)

    # ====== Calculating Expected Return ======
    analyst_ratings['expected_return'] = ((analyst_ratings['target'] - analyst_ratings['cleaned_price_at_reco'])
                                          / analyst_ratings['cleaned_price_at_reco']) * 100

    # ====== Company Name Matching using TF-IDF ======
    company_names = [x.lower() for x in stock_df['Company Name'].unique().tolist()]
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(company_names)

    def get_closest_company_name(input_name, company_names, tfidf_matrix, vectorizer):
        input_tfidf = vectorizer.transform([input_name.lower()])
        cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()
        best_match_idx = cosine_similarities.argmax()
        best_match_score = cosine_similarities[best_match_idx]
        return company_names[best_match_idx] if best_match_score > 0.5 else None

    tqdm.pandas()
    analyst_ratings['matched_company_name'] = analyst_ratings['stock_name'].progress_apply(
        lambda x: get_closest_company_name(x, company_names, tfidf_matrix, vectorizer)
    )

    # ====== Mapping Company Symbols ======
    symbol_map = stock_df[['Company Name', 'Symbol']].drop_duplicates()
    symbol_map['Company Name'] = symbol_map['Company Name'].str.lower()
    analyst_ratings = pd.merge(analyst_ratings, symbol_map,
                               left_on='matched_company_name', right_on='Company Name', how='left')

    # ====== Getting the Nearest Stock Date ======
    def get_nearest_date(stock_data, reference_date, days=3):
        stock_data = stock_data.copy()
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        reference_date = pd.to_datetime(reference_date)

        start_date = reference_date - timedelta(days=days)
        end_date = reference_date + timedelta(days=days)

        filtered_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
        if reference_date in filtered_data['Date'].values:
            return reference_date
        else:
            if not filtered_data.empty:
                closest_date = filtered_data.loc[filtered_data['Date'].sub(reference_date).abs().idxmin(), 'Date']
                return closest_date
            else:
                return None

    # ====== Predicting Actual Return Using n-Day Prediction ======
    for index, row in tqdm(analyst_ratings.iterrows(), total=analyst_ratings.shape[0], desc="Processing"):
        matched_company = row['matched_company_name']
        matched_ticker = None

        if matched_company:
            ticker_matches = stock_df[stock_df['Company Name'].str.lower() == matched_company]['Symbol']
            if not ticker_matches.empty:
                matched_ticker = ticker_matches.iloc[0]

        nearest_date = None
        if matched_ticker:
            try:
                stock_data = stock_df.loc[stock_df['Symbol'] == matched_ticker].copy()
                nearest_date = get_nearest_date(stock_data, row['date'])
                if nearest_date is not None:
                    # Only capture total_return from the prediction function.
                    _, _, total_return, _, _, _ = predict_n_day_return(matched_ticker, nearest_date, n_days, stock_df, dividend_rows)
                    analyst_ratings.loc[index, 'actual_return'] = total_return
                else:
                    analyst_ratings.loc[index, 'actual_return'] = None
            except Exception as e:
                analyst_ratings.loc[index, 'actual_return'] = None
                tqdm.write(f"Error for {matched_ticker} on {row['date']}: {e}")
        else:
            analyst_ratings.loc[index, 'actual_return'] = None

        if nearest_date is not None and nearest_date != row['date']:
            tqdm.write(f"Reference date {row['date']} not found. Using nearest date: {nearest_date}")

    # ====== Identify Industry Sector ======
    industry_keywords = {
        'Finance': ['bank', 'financial', 'insurance', 'capital', 'investment', 'wealth', 'lending',
                    'asset management', 'mutual fund', 'securities', 'credit', 'equity', 'stock exchange',
                    'private equity', 'venture capital'],
        'Technology': ['tech', 'software', 'IT', 'digital', 'cloud', 'data', 'artificial intelligence',
                       'AI', 'machine learning', 'hardware', 'blockchain', 'cybersecurity', 'semiconductor',
                       'saas', 'paas', 'technology services', 'internet', 'networking'],
        'Healthcare': ['pharma', 'biotech', 'medical', 'healthcare', 'hospital', 'clinical', 'diagnostics',
                       'life sciences', 'medtech', 'wellness', 'drug', 'therapy', 'genomics', 'vaccines',
                       'biopharma'],
        'Manufacturing': ['steel', 'cement', 'construction', 'engineering', 'manufacturing', 'machinery',
                          'industrial', 'fabrication', 'metals', 'automotive', 'tools', 'aerospace',
                          'assembly', 'chemicals', 'plastics', 'production'],
        'Energy': ['oil', 'energy', 'gas', 'power', 'coal', 'electricity', 'solar', 'renewable',
                   'wind', 'hydro', 'nuclear', 'fossil fuels', 'mining', 'petroleum', 'green energy', 'battery'],
        'Consumer Goods': ['consumer', 'retail', 'fashion', 'apparel', 'food', 'beverage', 'fmcg', 'personal care',
                           'household', 'luxury', 'ecommerce', 'cosmetics', 'home appliances', 'groceries',
                           'sportswear', 'furniture'],
        'Telecommunications': ['telecom', 'wireless', 'broadband', 'satellite', 'mobile', 'network', 'fiber',
                               'telecommunication services', 'cellular', '5G', 'communication', 'ISP'],
        'Utilities': ['water', 'electric', 'gas distribution', 'utilities', 'waste management', 'public services',
                      'sanitation', 'power grid'],
        'Real Estate': ['real estate', 'property', 'construction', 'housing', 'commercial real estate', 'residential',
                        'mortgage', 'land development', 'realty', 'leasing', 'brokerage', 'facility management', 'rental'],
        'Transportation': ['logistics', 'shipping', 'transportation', 'rail', 'trucking', 'freight', 'airlines',
                           'aviation', 'maritime', 'delivery', 'supply chain', 'public transport', 'cargo', 'transport services'],
        'Media & Entertainment': ['media', 'entertainment', 'film', 'television', 'music', 'broadcast', 'news',
                                  'publishing', 'streaming', 'advertising', 'social media', 'gaming', 'content creation'],
        'Agriculture': ['agriculture', 'farming', 'crop', 'livestock', 'dairy', 'agribusiness', 'fertilizers',
                        'agrochemicals', 'forestry', 'horticulture', 'fisheries', 'seed', 'organic farming'],
        'Retail': ['retail', 'wholesale', 'department store', 'grocery', 'mall', 'shopping', 'online store',
                   'boutique', 'supermarket', 'convenience store']
    }

    def identify_sector(description):
        description = description.lower()
        for sector, keywords in industry_keywords.items():
            if any(keyword in description for keyword in keywords):
                return sector
        return 'Unknown'

    unique_companies_df = stock_df.drop_duplicates(subset=['Symbol']).copy()
    unique_companies_df['Sector'] = unique_companies_df['Description'].progress_apply(identify_sector)
    unique_sector_df = unique_companies_df[['Symbol', 'Company Name', 'Description', 'Sector']].copy()

    merged_df = pd.merge(analyst_ratings, unique_sector_df[['Symbol', 'Sector']], on='Symbol', how='inner')
    filtered_df = merged_df[['date', 'Company Name', 'Symbol', 'author', 'Sector', 'expected_return', 'actual_return']]

    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    filtered_df = filtered_df.dropna()

    sector_group_details = filtered_df.groupby('Sector').agg({
        'Company Name': lambda x: ', '.join(x.unique()),
        'Symbol': 'count'
    }).reset_index()
    sector_group_details.rename(columns={'Symbol': 'Company Count'}, inplace=True)

    output_filename = f'data/filtered_data_{n_days}.csv'
    filtered_df.to_csv(output_filename, index=False)
    print(f"\nFiltered data saved to {output_filename}")
    print("\nSector Group Details:")
    print(sector_group_details)

    filtered_df['expected_return'] = pd.to_numeric(filtered_df['expected_return'], errors='coerce')
    filtered_df['actual_return'] = pd.to_numeric(filtered_df['actual_return'], errors='coerce')

    filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')


    # Optionally return DataFrames for further processing.
    return filtered_df


# if __name__ == '__main__':
#     N_DAYS = 1  # Set the desired number of days for return prediction.
#     preprocess_data(N_DAYS)