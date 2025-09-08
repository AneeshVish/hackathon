#!/usr/bin/env python3
"""
Training script for patient deterioration prediction models
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.train import ModelTrainer
from data.pipeline import DataPipeline
from data.preprocessing import FeatureEngineer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Train patient deterioration prediction models')
    parser.add_argument('--data-path', type=str, default='data/processed', 
                       help='Path to processed data')
    parser.add_argument('--model-type', type=str, default='all',
                       choices=['logistic', 'random_forest', 'lightgbm', 'xgboost', 'all'],
                       help='Model type to train')
    parser.add_argument('--output-dir', type=str, default='models/saved',
                       help='Output directory for trained models')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training pipeline...")
    
    try:
        # Initialize components
        trainer = ModelTrainer()
        
        # Load and prepare data
        logger.info("Loading training data...")
        X_train, X_test, y_train, y_test = trainer.load_data(
            data_path=args.data_path,
            test_size=args.test_size
        )
        
        # Train models
        if args.model_type == 'all':
            models_to_train = ['logistic', 'random_forest', 'lightgbm', 'xgboost']
        else:
            models_to_train = [args.model_type]
        
        results = {}
        for model_type in models_to_train:
            logger.info(f"Training {model_type} model...")
            result = trainer.train_model(
                X_train, y_train, X_test, y_test,
                model_type=model_type,
                cv_folds=args.cv_folds
            )
            results[model_type] = result
            
            # Save model
            model_path = os.path.join(args.output_dir, f"{model_type}_model.joblib")
            trainer.save_model(result['model'], model_path)
            logger.info(f"Model saved to {model_path}")
        
        # Select best model
        best_model_name = trainer.select_best_model(results)
        logger.info(f"Best model: {best_model_name}")
        
        # Save best model as default
        best_model_path = os.path.join(args.output_dir, "best_model.joblib")
        trainer.save_model(results[best_model_name]['model'], best_model_path)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
