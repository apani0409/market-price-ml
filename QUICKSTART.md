# Quick Start Guide

## Installation

```bash
# Clone or navigate to project directory
cd ferias

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run the Pipeline

### Option 1: Run as Script

```bash
python -m src.train
```

Output:
```
============================================================
Starting Agricultural Price Prediction Pipeline
============================================================

[1/5] Loading data...
[2/5] Preprocessing data...
[3/5] Engineering features...
[4/5] Training random_forest model...
[5/5] Results Summary
────────────────────────────────────────────────────────────
Model Type: random_forest
Mean Absolute Error (MAE): 96.1983
Root Mean Squared Error (RMSE): 169.4086
Test set samples: 902
```

### Option 2: Use as Module in Python

```python
from src.train import main

# Run with default parameters
model, eval_results, test_df = main()

# Or customize
model, eval_results, test_df = main(
    model_type='linear_regression',
    test_size=0.2,
    rolling_window=4
)

# Access results
print(f"MAE: {eval_results['mae']:.4f}")
print(f"RMSE: {eval_results['rmse']:.4f}")
```

### Option 3: Use Individual Modules

```python
from src.data_loader import load_data
from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline
from src.model import train_and_evaluate

# Build pipeline step-by-step
df = load_data()
df = preprocess_pipeline(df)
df = feature_engineering_pipeline(df, rolling_window=4)
model, results, test_df = train_and_evaluate(df, model_type='random_forest')
```

### Option 4: Run Jupyter Notebook

```bash
jupyter notebook notebooks/agricultural_price_prediction.ipynb
```

Then run cells in order. The notebook includes:
- Data exploration
- Complete pipeline execution
- Model comparison (RandomForest vs LinearRegression)
- Visualization of results

## Project Structure

```
ferias/
├── src/                                # Production code
│   ├── data_loader.py                  # Load CSV
│   ├── preprocessing.py                # Clean & aggregate
│   ├── features.py                     # Feature engineering
│   ├── model.py                        # Train & evaluate
│   └── train.py                        # Orchestration
│
├── notebooks/
│   └── agricultural_price_prediction.ipynb
│
├── data/
│   └── raw_prices.csv                  # Input data
│
├── requirements.txt                    # Dependencies
├── README.md                           # Documentation
├── ARCHITECTURE.md                     # Design details
└── QUICKSTART.md                       # This file
```

## Key Features

✅ **Modular Design** - Each module has a single responsibility  
✅ **Time-Aware Split** - Chronological train/test split, no data leakage  
✅ **Multiple Models** - RandomForest and LinearRegression  
✅ **Feature Engineering** - Temporal and rolling statistics  
✅ **Production-Ready** - Logging, error handling, type hints  
✅ **Well-Documented** - Docstrings, comments, examples  

## Configuration

Modify parameters in `src/train.py`:

```python
main(
    model_type='random_forest',  # 'linear_regression' for baseline
    test_size=0.2,              # 20% test set
    rolling_window=4            # 4-week rolling window
)
```

## Expected Performance

On the full dataset (9,184 raw records → 4,509 aggregated):

- **RandomForest**: MAE ≈ 96, RMSE ≈ 169
- **LinearRegression**: MAE ≈ 115, RMSE ≈ 205

(Results vary slightly with different rolling window sizes and data splits)

## Common Tasks

### Add a New Feature

Edit `src/features.py`:

```python
def create_your_feature(df):
    """Add your custom feature"""
    df['your_feature'] = df['mean_price'].transform(...)
    return df
```

Then add to `feature_engineering_pipeline()`.

### Try a Different Model

Edit `src/model.py` in `train_model()`:

```python
elif model_type == 'xgboost':
    model = XGBRegressor(...)
```

Then use: `model_type='xgboost'` in `main()`.

### Save Trained Model

```python
import pickle

model, _, _ = main()
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Later, load:
with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Analyze Feature Importance (RandomForest)

```python
model, _, _ = main(model_type='random_forest')

feature_names = ['year', 'week', 'week_of_year', 'rolling_mean_price', 'rolling_std_price']
importances = model.feature_importances_

for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name:25s}: {imp:.4f}")
```

## Troubleshooting

**Error: "Data file not found"**
- Check that `data/raw_prices.csv` exists
- Verify CSV has correct columns: year, week, publication_date, NOMBRE, variety, quality, size, sale_format, unit, price

**Error: "ModuleNotFoundError: No module named 'src'"**
- Run from project root directory: `cd ferias`
- Or use: `python -m src.train`

**Slow execution**
- Default: RandomForest with 100 trees
- To speed up: Reduce `n_estimators` in `model.py`
- Or use LinearRegression: `main(model_type='linear_regression')`

**Low accuracy**
- Check data quality: `python -c "from src.data_loader import load_data; print(load_data().describe())"`
- Try different rolling window: `main(rolling_window=8)`
- Consider adding more features in `features.py`

## Next Steps

1. **Explore the Data**: Run the Jupyter notebook
2. **Understand the Code**: Read `ARCHITECTURE.md`
3. **Customize Features**: Add domain-specific features
4. **Tune Hyperparameters**: Experiment with model parameters
5. **Deploy to Production**: Save and version models
6. **Monitor Performance**: Track metrics over time

## Resources

- **Pandas Time Series**: https://pandas.pydata.org/docs/user_guide/timeseries.html
- **Scikit-learn RandomForest**: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
- **Time Series ML**: https://towardsdatascience.com/time-series-machine-learning-regression-framework-9ea33929009a

## Support

For issues or questions:
1. Check the logs (info/debug messages)
2. Review `ARCHITECTURE.md` for design details
3. Look at the Jupyter notebook for usage examples
4. Check `README.md` for comprehensive documentation
