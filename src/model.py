"""
Model training, evaluation, and time-aware splitting.

Responsibilities:
- Time-aware train/test split (no shuffling)
- Train RandomForest and LinearRegression models
- Evaluate using MAE and RMSE metrics
- Return trained model and evaluation results
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def time_aware_split(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets using time-aware split (no shuffling).
    
    Strategy:
    - Sort by (variety, year, week)
    - Take first (1 - test_size)% as training
    - Take last test_size% as testing
    
    Args:
        df: Features and target DataFrame
        test_size: Fraction of data to use for testing (default: 0.2)
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    df = df.sort_values(['variety', 'year', 'week']).reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    
    return train_df, test_df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and target vector.
    
    Features:
    - year, week, week_of_year, rolling_mean_price, rolling_std_price
    
    Target:
    - mean_price
    
    Args:
        df: DataFrame with all features and target
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y)
    """
    feature_cols = ['year', 'week', 'week_of_year', 'rolling_mean_price', 'rolling_std_price']
    target_col = 'mean_price'
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    
    return X, y


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'random_forest'
) -> Any:
    """
    Train regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Either 'random_forest' or 'linear_regression'
        
    Returns:
        Trained model (sklearn estimator)
    """
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        logger.info("Training RandomForestRegressor...")
    elif model_type == 'linear_regression':
        model = LinearRegression()
        logger.info("Training LinearRegression...")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.fit(X_train, y_train)
    logger.info(f"Model training complete")
    
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model on test set using MAE and RMSE.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dict with 'mae' and 'rmse' metrics
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results = {
        'mae': mae,
        'rmse': rmse,
        'predictions': y_pred,
        'actuals': y_test
    }
    
    logger.info(f"Evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    return results


def train_and_evaluate(
    df: pd.DataFrame,
    model_type: str = 'random_forest',
    test_size: float = 0.2
) -> Tuple[Any, Dict[str, float], pd.DataFrame]:
    """
    Complete training and evaluation pipeline.
    
    Args:
        df: DataFrame with all features and target
        model_type: Type of model to train
        test_size: Fraction of data for testing
        
    Returns:
        Tuple[model, eval_results, test_df]:
            - model: Trained sklearn model
            - eval_results: Dict with MAE, RMSE, predictions, actuals
            - test_df: Test set DataFrame with predictions
    """
    # Time-aware split
    train_df, test_df = time_aware_split(df, test_size=test_size)
    
    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)
    
    # Train model
    model = train_model(X_train, y_train, model_type=model_type)
    
    # Evaluate
    eval_results = evaluate_model(model, X_test, y_test)
    
    # Add predictions to test set
    test_df = test_df.copy()
    test_df['predicted_price'] = eval_results['predictions']
    
    return model, eval_results, test_df
