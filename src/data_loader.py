"""
Load agricultural market price CSV data.

Responsibilities:
- Read CSV file
- Parse publication_date as datetime
- Basic validation
- Return pandas DataFrame
"""

from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_prices.csv"


def load_data() -> pd.DataFrame:
    """
    Load and parse agricultural market price data from CSV.
    
    Returns:
        pd.DataFrame: Raw price data with publication_date parsed as datetime.
        
    Raises:
        FileNotFoundError: If CSV file does not exist.
        pd.errors.ParserError: If CSV is malformed.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    logger.info(f"Loading data from {DATA_PATH}")
    
    df = pd.read_csv(
        DATA_PATH,
        dtype={
            'year': 'int32',
            'week': 'int32',
            'NOMBRE': 'string',
            'variety': 'string',
            'quality': 'string',
            'size': 'string',
            'sale_format': 'string',
            'unit': 'string',
            'price': 'float32'
        },
        parse_dates=['publication_date']
    )
    
    logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    return df
