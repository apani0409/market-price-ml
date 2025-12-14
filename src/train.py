"""
Main training pipeline orchestration.

Integrates all modules:
- data_loader: Load CSV
- preprocessing: Clean and aggregate
- features: Engineer features
- model: Train and evaluate

Run as: python -m src.train
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data
from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline
from src.model import train_and_evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    rolling_window: int = 4
) -> None:
    """
    Execute complete ML pipeline.
    
    Args:
        model_type: 'random_forest' or 'linear_regression'
        test_size: Fraction of data for testing
        rolling_window: Window size for rolling features
    """
    logger.info("="*60)
    logger.info("Starting Agricultural Price Prediction Pipeline")
    logger.info("="*60)
    
    # Step 1: Load data
    logger.info("\n[1/5] Loading data...")
    df = load_data()
    
    # Step 2: Preprocess
    logger.info("\n[2/5] Preprocessing data...")
    df = preprocess_pipeline(df)
    
    # Step 3: Feature engineering
    logger.info("\n[3/5] Engineering features...")
    df = feature_engineering_pipeline(df, rolling_window=rolling_window)
    
    logger.info(f"\nDataset shape before training: {df.shape}")
    logger.info(f"Features: {df.columns.tolist()}")
    
    # Step 4: Train and evaluate
    logger.info(f"\n[4/5] Training {model_type} model...")
    model, eval_results, test_df = train_and_evaluate(
        df,
        model_type=model_type,
        test_size=test_size
    )
    
    # Step 5: Results summary
    logger.info("\n[5/5] Results Summary")
    logger.info("-"*60)
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Mean Absolute Error (MAE): {eval_results['mae']:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {eval_results['rmse']:.4f}")
    logger.info(f"Test set samples: {len(test_df)}")
    
    # Show sample predictions
    logger.info("\nSample predictions (first 5 test samples):")
    sample_cols = ['variety', 'year', 'week', 'mean_price', 'predicted_price']
    for _, row in test_df[sample_cols].head(5).iterrows():
        logger.info(
            f"  {row['variety']:20s} | Week {row['week']:2d}/{row['year']:4d} | "
            f"Actual: {row['mean_price']:8.2f} | Predicted: {row['predicted_price']:8.2f}"
        )
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline execution completed successfully!")
    logger.info("="*60)
    
    return model, eval_results, test_df


if __name__ == '__main__':
    # Run pipeline with default parameters
    # Modify these to experiment with different configurations
    model, results, test_data = main(
        model_type='random_forest',
        test_size=0.2,
        rolling_window=4
    )
