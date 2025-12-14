# Agricultural Market Price Prediction

A production-ready ML pipeline for predicting agricultural market prices using time-series data, developed in collaboration with the National Production Council of Costa Rica (CNP) and the University of Costa Rica (UCR).

## Project Structure

```
src/
├── __init__.py              # Package initialization
├── data_loader.py          # CSV loading and parsing
├── preprocessing.py        # Data cleaning and aggregation
├── features.py             # Feature engineering
├── model.py                # Model training and evaluation
└── train.py                # Main orchestration

data/
└── raw_prices.csv          # Input agricultural price data
```

## Pipeline Overview

The complete ML pipeline consists of 5 modular stages:

### 1. **Data Loading** (`data_loader.py`)
- Reads CSV file with proper dtype specification
- Parses `publication_date` as datetime
- Returns cleaned DataFrame

### 2. **Preprocessing** (`preprocessing.py`)
- **Clean**: Drops rows with missing `price` or `variety`
- **Filter**: Keeps only `unit == "kg"`
- **Aggregate**: Groups by `(variety, year, week)` and calculates:
  - `mean_price`: Average price per product-week
  - `price_std`: Price volatility

### 3. **Feature Engineering** (`features.py`)
Creates time-based features for each product:
- **Temporal**: `week_of_year` from `publication_date`
- **Rolling Mean** (4-week window): `rolling_mean_price`
- **Rolling Std** (4-week window): `rolling_std_price`

### 4. **Model Training** (`model.py`)
- **Time-aware split**: Chronological train/test split (no shuffling)
- **Models**: RandomForest or LinearRegression
- **Features**: `year`, `week`, `week_of_year`, `rolling_mean_price`, `rolling_std_price`
- **Target**: `mean_price`
- **Metrics**: MAE and RMSE

### 5. **Orchestration** (`train.py`)
- Integrates all modules
- Configurable model type and parameters
- Comprehensive logging
- Sample prediction output

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Training Pipeline

```bash
python -m src.train
```

### Use in Code

```python
from src.data_loader import load_data
from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline
from src.model import train_and_evaluate

# Load and process data
df = load_data()
df = preprocess_pipeline(df)
df = feature_engineering_pipeline(df, rolling_window=4)

# Train model
model, eval_results, test_df = train_and_evaluate(
    df,
    model_type='random_forest',
    test_size=0.2
)

# Access results
print(f"MAE: {eval_results['mae']}")
print(f"RMSE: {eval_results['rmse']}")
```

## Dataset Columns

Input CSV columns:
- `year` (int): Year of observation
- `week` (int): Week number
- `publication_date` (date): Publication date
- `NOMBRE` (str): Product name
- `variety` (str): Product identifier
- `quality` (str, optional): Quality grade
- `size` (str, optional): Size category
- `sale_format` (str, optional): Sale format
- `unit` (str): Unit (kg, u, etc.)
- `price` (float): Price value (target)

## Key Design Principles

✅ **Modularity**: Each file has a single responsibility
✅ **Readability**: Clear function names, docstrings, and type hints
✅ **Production-Ready**: Proper error handling, logging, and documentation
✅ **Reproducibility**: Time-aware splitting prevents data leakage
✅ **Extensibility**: Easy to add new models or features
✅ **No Notebook Code**: Everything is in reusable modules

## Model Performance

The pipeline evaluates using:
- **MAE** (Mean Absolute Error): Average prediction error in absolute price units
- **RMSE** (Root Mean Squared Error): Penalizes larger errors more heavily

## Configuration

Modify parameters in `train.py`:

```python
main(
    model_type='random_forest',  # or 'linear_regression'
    test_size=0.2,              # 20% test set
    rolling_window=4            # 4-week rolling window
)
```

## Next Steps

- Add cross-validation for more robust evaluation
- Implement feature importance analysis
- Add hyperparameter tuning
- Save/load trained models
- Create unit tests
