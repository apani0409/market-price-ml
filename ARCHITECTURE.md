"""
ARCHITECTURE GUIDE - Agricultural Price Prediction ML Pipeline

This document explains the modular architecture and how each component works together.
"""

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

"""
ferias/
├── data/
│   └── raw_prices.csv                  # Input dataset (9,184 rows)
│
├── src/                                # Production code
│   ├── __init__.py                     # Package initialization
│   ├── data_loader.py                  # CSV loading and parsing
│   ├── preprocessing.py                # Data cleaning and aggregation
│   ├── features.py                     # Feature engineering
│   ├── model.py                        # Model training and evaluation
│   └── train.py                        # Main orchestration
│
├── notebooks/                          # Jupyter notebooks for exploration
│   └── agricultural_price_prediction.ipynb
│
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── ARCHITECTURE.md                     # This file
"""

# ============================================================================
# PIPELINE FLOW
# ============================================================================

"""
INPUT (raw_prices.csv: 9,184 rows)
    ↓
[1] LOAD (data_loader.py)
    └─→ Read CSV with proper dtypes
    └─→ Parse publication_date as datetime
    └─→ Return DataFrame: (9,184 rows)
    ↓
[2] PREPROCESS (preprocessing.py)
    ├─→ clean_data()
    │   ├─ Drop missing price/variety → 8,924 rows
    │   ├─ Filter unit == 'kg' → 5,075 rows
    │
    └─→ aggregate_weekly()
        ├─ Group by (variety, year, week)
        ├─ Calculate mean_price
        ├─ Calculate price_std
        └─ Return DataFrame: (4,509 rows)
    ↓
[3] FEATURE ENGINEERING (features.py)
    ├─→ create_temporal_features()
    │   └─ Add week_of_year from publication_date
    │
    └─→ create_rolling_features()
        ├─ Calculate rolling_mean_price (4-week window)
        ├─ Calculate rolling_std_price (4-week window)
        └─ Sort by variety and time within each product
    ↓
[4] DATA SPLITTING (model.py)
    ├─→ time_aware_split()
    │   ├─ Sort by (variety, year, week)
    │   ├─ No shuffling (preserves temporal order)
    │   ├─ Train: 3,607 samples (80%)
    │   └─ Test: 902 samples (20%)
    ↓
[5] MODEL TRAINING (model.py)
    ├─→ prepare_features()
    │   └─ Extract: [year, week, week_of_year, rolling_mean_price, rolling_std_price]
    │   └─ Target: mean_price
    │
    └─→ train_model()
        ├─ RandomForestRegressor (default)
        │   └─ 100 trees, max_depth=15, random_state=42
        │
        └─ LinearRegression (alternative)
    ↓
[6] EVALUATION (model.py)
    ├─→ evaluate_model()
    │   ├─ MAE: Mean Absolute Error
    │   ├─ RMSE: Root Mean Squared Error
    │   └─ Return predictions and actuals
    ↓
OUTPUT (trained model + evaluation metrics)
"""

# ============================================================================
# MODULE RESPONSIBILITIES
# ============================================================================

"""
1. data_loader.py
   ─────────────
   Responsibility: Read and parse the CSV file
   
   Functions:
   - load_data() → pd.DataFrame
     ├─ Input: CSV path (hardcoded as DATA_PATH)
     ├─ Output: DataFrame with 10 columns
     └─ Parsing: publication_date as datetime
   
   Design:
   ✓ Single responsibility (only loading)
   ✓ Error handling for missing files
   ✓ Logging for traceability
   ✓ Type hints for clarity
   
   Dependencies: pandas, pathlib, logging


2. preprocessing.py
   ─────────────────
   Responsibility: Clean and aggregate raw data
   
   Functions:
   - clean_data(df) → pd.DataFrame
     ├─ Drop missing price/variety
     └─ Filter unit == 'kg'
   
   - aggregate_weekly(df) → pd.DataFrame
     ├─ Group by (variety, year, week)
     ├─ Compute mean_price
     ├─ Compute price_std (volatility)
     └─ Keep publication_date for feature engineering
   
   - preprocess_pipeline(df) → pd.DataFrame
     └─ Chains clean_data() → aggregate_weekly()
   
   Design:
   ✓ Clear separation of concerns
   ✓ Logging for debugging
   ✓ NaN handling for std (when n=1)
   
   Dependencies: pandas, logging


3. features.py
   ────────────
   Responsibility: Engineer time-series features
   
   Functions:
   - create_temporal_features(df) → pd.DataFrame
     └─ Add week_of_year from ISO calendar
   
   - create_rolling_features(df, window=4) → pd.DataFrame
     ├─ Sort by variety, year, week
     ├─ Per-variety rolling mean (4-week window)
     ├─ Per-variety rolling std (4-week window)
     └─ Fill NaN rolling_std with 0
   
   - feature_engineering_pipeline(df, rolling_window=4) → pd.DataFrame
     └─ Chains temporal → rolling features
   
   Design:
   ✓ Per-product rolling calculations (groupby)
   ✓ Proper handling of series start (min_periods=1)
   ✓ Configurable window size
   ✓ NaN handling for constant-price periods
   
   Dependencies: pandas, logging


4. model.py
   ─────────
   Responsibility: Train, evaluate, and manage models
   
   Functions:
   - time_aware_split(df, test_size=0.2) → (train_df, test_df)
     ├─ Chronological split (no shuffling)
     ├─ Preserves temporal order
     └─ Prevents future-to-past leakage
   
   - prepare_features(df) → (X, y)
     ├─ Extract feature columns
     ├─ Convert to numpy arrays
     └─ Return (features, target)
   
   - train_model(X_train, y_train, model_type) → model
     ├─ RandomForestRegressor (100 trees, depth=15)
     └─ LinearRegression (baseline)
   
   - evaluate_model(model, X_test, y_test) → dict
     ├─ Generate predictions
     ├─ Calculate MAE
     ├─ Calculate RMSE
     └─ Return results dict
   
   - train_and_evaluate(df, model_type, test_size) → (model, results, test_df)
     └─ Complete pipeline in one call
   
   Design:
   ✓ Time-aware splitting (critical for time-series)
   ✓ Flexible model selection
   ✓ Comprehensive error metrics
   ✓ Returns predictions for analysis
   
   Dependencies: pandas, numpy, sklearn, logging


5. train.py
   ──────────
   Responsibility: Orchestrate the complete pipeline
   
   Functions:
   - main(model_type, test_size, rolling_window) → (model, results, test_df)
     ├─ [1/5] load_data()
     ├─ [2/5] preprocess_pipeline()
     ├─ [3/5] feature_engineering_pipeline()
     ├─ [4/5] train_and_evaluate()
     └─ [5/5] Logging and summary
   
   Design:
   ✓ Clear pipeline stages
   ✓ Detailed logging at each step
   ✓ Configurable parameters
   ✓ Runnable as script or importable module
   ✓ No hardcoded magic numbers
   
   Usage:
   - As script: python -m src.train
   - As module: from src.train import main; model, _, _ = main()
   
   Dependencies: all src modules, logging
"""

# ============================================================================
# DATA TRANSFORMATIONS
# ============================================================================

"""
STAGE 1: LOAD
Input:  CSV file (raw_prices.csv)
        Columns: year, week, publication_date, NOMBRE, variety, quality,
                 size, sale_format, unit, price
        Rows: 9,184

Output: pd.DataFrame
        Columns: same as input
        Rows: 9,184
        Change: publication_date converted to datetime64

Example:
┌────┬──────┬────────────────┬──────────────┬────────────────┬───┐
│year│ week │publication_date│  NOMBRE      │   variety      │...│
├────┼──────┼────────────────┼──────────────┼────────────────┼───┤
│2021│  1   │2021-01-08      │aguacate Hass │aguacate_hass   │...│
│2021│  1   │2021-01-08      │apio verde    │apio_verde      │...│
└────┴──────┴────────────────┴──────────────┴────────────────┴───┘

───────────────────────────────────────────────────────────────────

STAGE 2: PREPROCESS
Input:  DataFrame from LOAD (9,184 rows)

Step 2a: Clean
  - Drop rows where price is NULL: 9,184 → 8,924 rows
  - Drop rows where variety is NULL: 8,924 → 8,924 rows
  - Filter unit == 'kg': 8,924 → 5,075 rows
  
Output: Clean DataFrame (5,075 rows)

Step 2b: Aggregate
Input:  Clean DataFrame (5,075 rows)

Groupby: (variety, year, week)
Aggregate:
  - price: mean() → mean_price
  - price: std() → price_std
  - publication_date: first() → publication_date (preserved)

Output: Aggregated DataFrame (4,509 unique combinations)

Example aggregation:
┌───────────────┬──────┬──────┬────────────────┬───────────┬──────────────┐
│   variety     │ year │ week │publication_date│mean_price │  price_std   │
├───────────────┼──────┼──────┼────────────────┼───────────┼──────────────┤
│aguacate_hass  │ 2021 │  1   │2021-01-08      │  1903.00  │   0.00       │
│apio_verde     │ 2021 │  1   │2021-01-08      │  1000.00  │   0.00       │
│brocoli        │ 2021 │  1   │2021-01-08      │  2569.00  │   0.00       │
│cebolla_blanca │ 2021 │  1   │2021-01-08      │  1829.33  │  352.00      │
└───────────────┴──────┴──────┴────────────────┴───────────┴──────────────┘

───────────────────────────────────────────────────────────────────

STAGE 3: FEATURE ENGINEERING
Input:  Aggregated DataFrame (4,509 rows)

Step 3a: Temporal Features
  - week_of_year: ISO calendar week (1-53)
    Extract from publication_date
    
Step 3b: Rolling Features (per variety, 4-week window)
  Sort by: variety, year, week (time order)
  
  For each variety:
    rolling_mean_price = mean(price) over last 4 weeks
    rolling_std_price = std(price) over last 4 weeks
  
  At edges (first < 4 weeks):
    min_periods=1: use available data (e.g., 1, 2, 3 weeks)

Output: DataFrame with features (4,509 rows, 9 columns)

Example features:
┌───────────────┬──────┬──────┬────────────┬───────────────┬──────────────┐
│   variety     │ year │ week │week_of_year│rolling_mean_pr│rolling_std_pr│
├───────────────┼──────┼──────┼────────────┼───────────────┼──────────────┤
│aguacate_hass  │ 2021 │  1   │     1      │   1903.00     │    0.00      │
│aguacate_hass  │ 2021 │  2   │     2      │   1903.00     │    0.00      │
│aguacate_hass  │ 2021 │  3   │     3      │   1903.00     │    0.00      │
│aguacate_hass  │ 2021 │  4   │     4      │   1903.00     │    0.00      │
│aguacate_hass  │ 2021 │  5   │     5      │   1903.00     │    0.00      │
└───────────────┴──────┴──────┴────────────┴───────────────┴──────────────┘

───────────────────────────────────────────────────────────────────

STAGE 4: SPLIT & PREPARE
Input:  Featured DataFrame (4,509 rows)

Time-aware split (no shuffle):
  Sort by: variety, year, week
  Split at index 3,607 (80%)
  
  Train: rows 0-3,606 (3,607 samples)
  Test:  rows 3,607-4,508 (902 samples)

Feature preparation:
  X_features: [year, week, week_of_year, rolling_mean_price, rolling_std_price]
  y_target: [mean_price]
  
  Output: X_train (3607, 5), y_train (3607,)
          X_test (902, 5), y_test (902,)

───────────────────────────────────────────────────────────────────

STAGE 5: TRAIN & EVALUATE
Input:  X_train, y_train, X_test, y_test

Train (RandomForest):
  RandomForestRegressor(n_estimators=100, max_depth=15, ...)
  fit(X_train, y_train)

Predict & Evaluate:
  y_pred = model.predict(X_test)
  
  MAE = mean(|y_test - y_pred|)
  RMSE = sqrt(mean((y_test - y_pred)^2))

Output: Trained model, evaluation metrics, predictions
"""

# ============================================================================
# FEATURE ENGINEERING DETAILS
# ============================================================================

"""
FINAL FEATURE SET (5 features):

1. year (int32)
   Source: Original data
   Range: 2021-2023
   Purpose: Captures annual patterns (inflation, seasonal shifts)

2. week (int32)
   Source: Original data
   Range: 1-52
   Purpose: Captures weekly patterns within year

3. week_of_year (int32)
   Source: publication_date ISO calendar
   Range: 1-53
   Purpose: Captures seasonal patterns across calendar year
   Note: Similar to week, but cross-year consistent

4. rolling_mean_price (float32)
   Source: Aggregated mean_price with 4-week rolling window
   Window: Past 4 weeks per product
   Per-variety: Independent rolling window for each product
   Purpose: Captures recent price trends, momentum
   Edge handling: min_periods=1 (use available data at start)

5. rolling_std_price (float32)
   Source: Aggregated mean_price with 4-week rolling window (std)
   Window: Past 4 weeks per product
   Per-variety: Independent rolling window for each product
   Purpose: Captures price volatility, uncertainty
   Edge handling: min_periods=1 (use available data at start)
   NaN → 0: When all values in window are identical (no variation)

TARGET VARIABLE:

mean_price (float32)
  Source: Aggregated price data per (variety, year, week)
  Calculation: mean(price) over all quality/size/format variants
  Range: Varies by product (100-3000+)
  Purpose: Prediction target (what we're forecasting)
"""

# ============================================================================
# ERROR METRICS INTERPRETATION
# ============================================================================

"""
MAE (Mean Absolute Error):
  Formula: mean(|actual - predicted|)
  Units: Same as target (price in local currency)
  Interpretation: Average prediction error
  Example: MAE = 96.20 means predictions are off by ~96.20 units on average
  Use: Easy to interpret in business terms

RMSE (Root Mean Squared Error):
  Formula: sqrt(mean((actual - predicted)^2))
  Units: Same as target (price in local currency)
  Interpretation: Punishes larger errors more heavily
  Example: RMSE = 169.41 means typical prediction error is ~169.41 units
  Use: More sensitive to outliers, better for optimization

MAPE (Mean Absolute Percentage Error - optional):
  Formula: mean(|actual - predicted| / |actual|) * 100
  Units: Percentage (%)
  Interpretation: Relative error
  Use: Compare models across different price ranges

For this project:
  MAE ≈ 96: On average, predictions are off by ~96 price units
  RMSE ≈ 169: Considering outliers, typical error is ~169 units
  Trade-off: LinearRegression (simpler, faster) vs RandomForest (more accurate)
"""

# ============================================================================
# EXTENDING THE PIPELINE
# ============================================================================

"""
1. ADD NEW FEATURES:
   ─────────────────
   Location: features.py
   
   Example - Add lag features:
     def create_lag_features(df, lags=[1, 2, 4]):
         for lag in lags:
             df[f'lag_{lag}_price'] = df.groupby('variety')['mean_price'].shift(lag)
         return df
   
   Then add to feature_engineering_pipeline()

2. ADD NEW MODELS:
   ───────────────
   Location: model.py
   
   Example - Add Gradient Boosting:
     def train_model(X_train, y_train, model_type='random_forest'):
         ...
         elif model_type == 'gradient_boosting':
             model = GradientBoostingRegressor(...)
         ...

3. ADD CROSS-VALIDATION:
   ────────────────────
   Location: model.py
   
   Example - Time-series CV:
     from sklearn.model_selection import TimeSeriesSplit
     tscv = TimeSeriesSplit(n_splits=5)
     for train_idx, test_idx in tscv.split(X):
         X_train, X_test = X[train_idx], X[test_idx]
         ...

4. SAVE TRAINED MODELS:
   ───────────────────
   Location: train.py
   
   Example:
     import pickle
     with open('models/rf_model.pkl', 'wb') as f:
         pickle.dump(model, f)
     
     # Later, load:
     with open('models/rf_model.pkl', 'rb') as f:
         loaded_model = pickle.load(f)

5. HYPERPARAMETER TUNING:
   ─────────────────────
   Location: model.py
   
   Example - Grid Search:
     from sklearn.model_selection import GridSearchCV
     params = {'n_estimators': [50, 100, 200], 'max_depth': [10, 15, 20]}
     gs = GridSearchCV(RandomForestRegressor(), params, cv=5)
     gs.fit(X_train, y_train)
     best_model = gs.best_estimator_
"""

# ============================================================================
# PRODUCTION CHECKLIST
# ============================================================================

"""
Before deploying to production:

✓ Data Validation
  □ Check for unexpected null values
  □ Validate feature ranges
  □ Monitor data distribution drift

✓ Model Management
  □ Version control trained models
  □ Document model parameters and performance
  □ Track model lineage (data → preprocessing → model)

✓ Error Handling
  □ Handle missing data gracefully
  □ Validate input shapes and types
  □ Log errors for debugging

✓ Performance
  □ Monitor prediction latency
  □ Check memory usage
  □ Optimize for batch vs real-time predictions

✓ Testing
  □ Unit tests for each module
  □ Integration tests for pipeline
  □ Regression tests on hold-out set

✓ Monitoring
  □ Track prediction accuracy over time
  □ Alert if model performance degrades
  □ Monitor input data quality
  □ Log all predictions for audit trail

✓ Documentation
  □ API documentation
  □ Training data documentation
  □ Model card with limitations
  □ Deployment runbook
"""

print(__doc__)
