"""
Feature engineering for time-series price prediction.

Responsibilities:
- Create week_of_year feature
- Calculate rolling mean price (4-week window)
- Calculate rolling std price (4-week window)
- Handle missing rolling values (fillna with mean)
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from publication_date.
    
    Features:
    - week_of_year: ISO week number (1-53)
    
    Args:
        df: Aggregated DataFrame with publication_date
        
    Returns:
        pd.DataFrame: DataFrame with temporal features added
    """
    df = df.copy()
    
    # Extract week of year from publication_date
    df['week_of_year'] = df['publication_date'].dt.isocalendar().week
    
    logger.info("Created temporal features: week_of_year")
    
    return df


def create_rolling_features(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Create rolling window features per product.
    
    Rolling features (computed per variety, sorted by year-week):
    - rolling_mean_price: mean price over last N weeks
    - rolling_std_price: std dev of price over last N weeks
    
    Args:
        df: DataFrame with aggregated prices
        window: Rolling window size in weeks (default: 4)
        
    Returns:
        pd.DataFrame: DataFrame with rolling features added
    """
    df = df.copy()
    
    # Create a time key for proper sorting
    df['time_key'] = df['year'] * 100 + df['week']
    
    # Sort by variety and time
    df = df.sort_values(['variety', 'time_key']).reset_index(drop=True)
    
    # Calculate rolling features per variety
    df['rolling_mean_price'] = df.groupby('variety')['mean_price'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    df['rolling_std_price'] = df.groupby('variety')['mean_price'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )
    
    # Fill NaN rolling std with 0 (when all values in window are the same)
    df['rolling_std_price'] = df['rolling_std_price'].fillna(0.0)
    
    logger.info(f"Created rolling features with window={window}")
    
    return df


def feature_engineering_pipeline(df: pd.DataFrame, rolling_window: int = 4) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Aggregated DataFrame from preprocessing
        rolling_window: Window size for rolling features
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """
    df = create_temporal_features(df)
    df = create_rolling_features(df, window=rolling_window)
    
    # Drop the temporary time_key column
    df = df.drop(columns=['time_key'])
    
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    
    return df
