"""
Model training pipeline for patient deterioration prediction.
Implements baseline and advanced models with comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
import json
from pathlib import Path
import warnings

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from .evaluation import ModelEvaluator
from .explainability import ExplainabilityEngine

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Comprehensive model training and evaluation pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.evaluator = ModelEvaluator()
        self.explainer = ExplainabilityEngine()
        
    def train_all_models(
        self, 
        features_path: str, 
        output_dir: str,
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train all models and return comprehensive evaluation results
        
        Args:
            features_path: Path to processed features
            output_dir: Directory to save models and results
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Training results summary
        """
        
        logger.info("Starting model training pipeline")
        
        # Load and prepare data
        X, y, feature_names, metadata = self._load_and_prepare_data(features_path)
        
        # Time-based split to respect temporal ordering
        X_train, X_temp, y_train, y_temp = self._time_based_split(X, y, test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(test_size + val_size), 
            stratify=y_temp, random_state=42
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train models
        results = {}
        
        # 1. Baseline Logistic Regression
        logger.info("Training Logistic Regression baseline...")
        lr_results = self._train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        results['logistic_regression'] = lr_results
        
        # 2. Random Forest
        logger.info("Training Random Forest...")
        rf_results = self._train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        results['random_forest'] = rf_results
        
        # 3. LightGBM
        logger.info("Training LightGBM...")
        lgb_results = self._train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        results['lightgbm'] = lgb_results
        
        # 4. XGBoost
        logger.info("Training XGBoost...")
        xgb_results = self._train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        results['xgboost'] = xgb_results
        
        # Model comparison and selection
        best_model_name = self._select_best_model(results)
        logger.info(f"Best model: {best_model_name}")
        
        # Comprehensive evaluation of best model
        best_model_results = results[best_model_name]
        comprehensive_eval = self._comprehensive_evaluation(
            best_model_results['model'], X_test, y_test, feature_names, best_model_name
        )
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._save_models_and_results(results, comprehensive_eval, output_path, metadata)
        
        # Generate summary
        summary = {
            'training_completed': datetime.utcnow().isoformat(),
            'best_model': best_model_name,
            'model_performance': {name: res['test_metrics'] for name, res in results.items()},
            'data_summary': {
                'total_samples': len(X),
                'positive_rate': y.mean(),
                'feature_count': len(feature_names),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            },
            'comprehensive_evaluation': comprehensive_eval
        }
        
        logger.info("Model training completed successfully")
        return summary
    
    def _load_and_prepare_data(self, features_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """Load and prepare feature data for training"""
        
        # Load features
        df = pd.read_parquet(features_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['patient_id', 'label', 'label_date', 'feature_timestamp']]
        
        X = df[feature_cols].values
        y = df['label'].values
        
        # Handle missing values
        X = self._handle_missing_values(X, feature_cols)
        
        # Feature scaling for linear models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['standard'] = scaler
        
        metadata = {
            'feature_names': feature_cols,
            'total_samples': len(df),
            'positive_samples': y.sum(),
            'positive_rate': y.mean(),
            'missing_value_strategy': 'median_imputation'
        }
        
        return X_scaled, y, feature_cols, metadata
    
    def _handle_missing_values(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Handle missing values in feature matrix"""
        
        from sklearn.impute import SimpleImputer
        
        # Use median imputation for numerical features
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Store imputer for inference
        self.scalers['imputer'] = imputer
        
        return X_imputed
    
    def _time_based_split(self, X: np.ndarray, y: np.ndarray, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform time-based split to respect temporal ordering"""
        
        # Simple time-based split (assuming data is already sorted by time)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def _train_logistic_regression(
        self, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    ) -> Dict[str, Any]:
        """Train and evaluate logistic regression model"""
        
        # Hyperparameter tuning
        best_score = 0
        best_params = {}
        
        for C in [0.01, 0.1, 1.0, 10.0]:
            for penalty in ['l1', 'l2']:
                try:
                    model = LogisticRegression(
                        C=C, penalty=penalty, solver='liblinear', 
                        random_state=42, max_iter=1000
                    )
                    model.fit(X_train, y_train)
                    val_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = {'C': C, 'penalty': penalty}
                except:
                    continue
        
        # Train final model
        final_model = LogisticRegression(
            **best_params, solver='liblinear', random_state=42, max_iter=1000
        )
        final_model.fit(X_train, y_train)
        
        # Calibrate model
        calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        test_metrics = self.evaluator.evaluate_model(calibrated_model, X_test, y_test)
        val_metrics = self.evaluator.evaluate_model(calibrated_model, X_val, y_val)
        
        return {
            'model': calibrated_model,
            'base_model': final_model,
            'best_params': best_params,
            'test_metrics': test_metrics,
            'val_metrics': val_metrics,
            'feature_importance': np.abs(final_model.coef_[0])
        }
    
    def _train_random_forest(
        self, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    ) -> Dict[str, Any]:
        """Train and evaluate random forest model"""
        
        # Hyperparameter tuning
        best_score = 0
        best_params = {}
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        for n_est in param_grid['n_estimators']:
            for max_d in param_grid['max_depth']:
                for min_split in param_grid['min_samples_split']:
                    for min_leaf in param_grid['min_samples_leaf']:
                        model = RandomForestClassifier(
                            n_estimators=n_est,
                            max_depth=max_d,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(X_train, y_train)
                        val_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                        
                        if val_score > best_score:
                            best_score = val_score
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': max_d,
                                'min_samples_split': min_split,
                                'min_samples_leaf': min_leaf
                            }
        
        # Train final model
        final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        final_model.fit(X_train, y_train)
        
        # Calibrate
        calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        test_metrics = self.evaluator.evaluate_model(calibrated_model, X_test, y_test)
        val_metrics = self.evaluator.evaluate_model(calibrated_model, X_val, y_val)
        
        return {
            'model': calibrated_model,
            'base_model': final_model,
            'best_params': best_params,
            'test_metrics': test_metrics,
            'val_metrics': val_metrics,
            'feature_importance': final_model.feature_importances_
        }
    
    def _train_lightgbm(
        self, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    ) -> Dict[str, Any]:
        """Train and evaluate LightGBM model"""
        
        # Prepare datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Train with early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Create sklearn-compatible wrapper
        class LGBWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict_proba(self, X):
                preds = self.model.predict(X)
                return np.column_stack([1-preds, preds])
            
            def predict(self, X):
                return (self.model.predict(X) > 0.5).astype(int)
        
        wrapped_model = LGBWrapper(model)
        
        # Calibrate
        calibrated_model = CalibratedClassifierCV(wrapped_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        test_metrics = self.evaluator.evaluate_model(calibrated_model, X_test, y_test)
        val_metrics = self.evaluator.evaluate_model(calibrated_model, X_val, y_val)
        
        return {
            'model': calibrated_model,
            'base_model': model,
            'test_metrics': test_metrics,
            'val_metrics': val_metrics,
            'feature_importance': model.feature_importance()
        }
    
    def _train_xgboost(
        self, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    ) -> Dict[str, Any]:
        """Train and evaluate XGBoost model"""
        
        # Prepare datasets
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
        
        # Train with early stopping
        model = xgb.train(
            params,
            dtrain,
            evals=[(dval, 'val')],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Create sklearn-compatible wrapper
        class XGBWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict_proba(self, X):
                dtest = xgb.DMatrix(X)
                preds = self.model.predict(dtest)
                return np.column_stack([1-preds, preds])
            
            def predict(self, X):
                dtest = xgb.DMatrix(X)
                return (self.model.predict(dtest) > 0.5).astype(int)
        
        wrapped_model = XGBWrapper(model)
        
        # Calibrate
        calibrated_model = CalibratedClassifierCV(wrapped_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        test_metrics = self.evaluator.evaluate_model(calibrated_model, X_test, y_test)
        val_metrics = self.evaluator.evaluate_model(calibrated_model, X_val, y_val)
        
        return {
            'model': calibrated_model,
            'base_model': model,
            'test_metrics': test_metrics,
            'val_metrics': val_metrics,
            'feature_importance': model.get_score(importance_type='weight')
        }
    
    def _select_best_model(self, results: Dict[str, Any]) -> str:
        """Select best model based on validation AUC and calibration"""
        
        best_score = 0
        best_model = None
        
        for model_name, result in results.items():
            # Weighted score: 70% AUC + 30% calibration (lower Brier score is better)
            auc = result['test_metrics']['auc']
            brier = result['test_metrics']['brier_score']
            
            # Normalize Brier score (lower is better, so we use 1 - brier)
            normalized_brier = max(0, 1 - brier)
            
            weighted_score = 0.7 * auc + 0.3 * normalized_brier
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_model = model_name
        
        return best_model
    
    def _comprehensive_evaluation(
        self, model, X_test, y_test, feature_names, model_name
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation including subgroup analysis"""
        
        # Basic metrics
        basic_metrics = self.evaluator.evaluate_model(model, X_test, y_test)
        
        # Calibration analysis
        calibration_metrics = self.evaluator.analyze_calibration(model, X_test, y_test)
        
        # Feature importance and explainability
        explainability = self.explainer.generate_global_explanations(
            model, X_test, feature_names
        )
        
        # Decision curve analysis
        decision_curve = self.evaluator.decision_curve_analysis(model, X_test, y_test)
        
        return {
            'basic_metrics': basic_metrics,
            'calibration': calibration_metrics,
            'explainability': explainability,
            'decision_curve': decision_curve,
            'model_name': model_name
        }
    
    def _save_models_and_results(
        self, results: Dict[str, Any], comprehensive_eval: Dict[str, Any], 
        output_path: Path, metadata: Dict[str, Any]
    ):
        """Save trained models and evaluation results"""
        
        # Save models
        models_dir = output_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for model_name, result in results.items():
            model_path = models_dir / f'{model_name}.joblib'
            joblib.dump(result['model'], model_path)
            
            # Save base model if different
            if 'base_model' in result:
                base_model_path = models_dir / f'{model_name}_base.joblib'
                joblib.dump(result['base_model'], base_model_path)
        
        # Save scalers
        scalers_path = models_dir / 'scalers.joblib'
        joblib.dump(self.scalers, scalers_path)
        
        # Save results
        results_dir = output_path / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Model comparison
        comparison = {
            'model_comparison': {
                name: result['test_metrics'] for name, result in results.items()
            },
            'metadata': metadata,
            'comprehensive_evaluation': comprehensive_eval
        }
        
        with open(results_dir / 'model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Individual model results
        for model_name, result in results.items():
            model_results = {
                'test_metrics': result['test_metrics'],
                'val_metrics': result['val_metrics'],
                'best_params': result.get('best_params', {}),
                'feature_importance': result.get('feature_importance', []).tolist() if hasattr(result.get('feature_importance', []), 'tolist') else result.get('feature_importance', [])
            }
            
            with open(results_dir / f'{model_name}_results.json', 'w') as f:
                json.dump(model_results, f, indent=2, default=str)


def main():
    """Main training pipeline execution"""
    
    config = {
        'random_state': 42,
        'cv_folds': 5,
        'early_stopping_rounds': 50
    }
    
    trainer = ModelTrainer(config)
    
    try:
        results = trainer.train_all_models(
            features_path='./data/processed/patient_features.parquet',
            output_dir='./models/trained',
            test_size=0.2,
            val_size=0.2
        )
        
        print("Training completed successfully!")
        print(f"Best model: {results['best_model']}")
        print(f"Best AUC: {results['model_performance'][results['best_model']]['auc']:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
