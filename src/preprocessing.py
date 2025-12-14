"""
Data preprocessing and aggregation.

Responsibilities:
- Remove rows with missing price or variety
- Filter data by unit (kg only)
- Aggregate data by (variety, year, week)
- Calculate mean price and standard deviation
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw agricultural price data.
    
    Steps:
    - Drop rows with missing price or variety
    - Filter to keep only unit == "kg"
    
    Args:
        df: Raw DataFrame from load_data()
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    logger.info(f"Starting with {len(df)} rows")
    
    # Drop rows with missing price or variety
    df_clean = df.dropna(subset=['price', 'variety'])
    logger.info(f"After dropping missing price/variety: {len(df_clean)} rows")
    
    # Filter to keep only kg units
    df_clean = df_clean[df_clean['unit'] == 'kg'].copy()
    logger.info(f"After filtering to unit='kg': {len(df_clean)} rows")
    
    return df_clean


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data by (variety, year, week).
    
    Calculates:
    - mean_price: average price per product per week
    - price_std: standard deviation of price per product per week
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        pd.DataFrame: Aggregated data with columns:
            [variety, year, week, publication_date, mean_price, price_std]
    """
    # Keep publication_date for later use in feature engineering
    agg_df = df.groupby(['variety', 'year', 'week']).agg({
        'price': ['mean', 'std'],
        'publication_date': 'first'
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ['variety', 'year', 'week', 'mean_price', 'price_std', 'publication_date']
    
    # Fill NaN std (when only 1 sample per group)
    agg_df['price_std'] = agg_df['price_std'].fillna(0.0)
    
    logger.info(f"Aggregated to {len(agg_df)} unique (variety, year, week) combinations")
    
    return agg_df


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw DataFrame from load_data()
        
    Returns:
        pd.DataFrame: Clean, aggregated data ready for feature engineering
    """
    df_clean = clean_data(df)
    df_agg = aggregate_weekly(df_clean)
    return df_agg
