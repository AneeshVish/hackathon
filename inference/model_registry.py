"""
Model registry for managing trained models and their metadata.
Handles model loading, versioning, and performance tracking.
"""

import joblib
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata structure"""
    name: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    model_type: str
    calibration_status: str
    file_path: str


class ModelRegistry:
    """Registry for managing ML models and their metadata"""
    
    def __init__(self, models_dir: str = "./models/trained"):
        self.models_dir = Path(models_dir)
        self.current_model = None
        self.current_metadata = None
        self.feature_names = None
        self.model_version = None
        self.scalers = None
        
        # Model cache
        self._model_cache = {}
        self._metadata_cache = {}
        
    def load_default_model(self) -> bool:
        """Load the default/best performing model"""
        try:
            # Look for model comparison results
            comparison_file = self.models_dir / "results" / "model_comparison.json"
            
            if comparison_file.exists():
                with open(comparison_file, 'r') as f:
                    comparison_data = json.load(f)
                
                # Find best model based on AUC
                best_model_name = None
                best_auc = 0
                
                for model_name, metrics in comparison_data.get('model_comparison', {}).items():
                    auc = metrics.get('auc', 0)
                    if auc > best_auc:
                        best_auc = auc
                        best_model_name = model_name
                
                if best_model_name:
                    return self.load_model(best_model_name)
            
            # Fallback: try to load any available model
            model_files = list(self.models_dir.glob("models/*.joblib"))
            if model_files:
                # Load the first available model
                model_name = model_files[0].stem
                return self.load_model(model_name)
            
            logger.warning("No trained models found")
            return False
            
        except Exception as e:
            logger.error(f"Error loading default model: {e}")
            return False
    
    def load_model(self, model_name: str, version: str = "latest") -> bool:
        """
        Load a specific model by name and version
        
        Args:
            model_name: Name of the model to load
            version: Model version (default: latest)
            
        Returns:
            True if model loaded successfully
        """
        try:
            # Check cache first
            cache_key = f"{model_name}_{version}"
            if cache_key in self._model_cache:
                self.current_model = self._model_cache[cache_key]
                self.current_metadata = self._metadata_cache[cache_key]
                self._set_model_attributes()
                return True
            
            # Load model file
            model_path = self.models_dir / "models" / f"{model_name}.joblib"
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load the model
            model = joblib.load(model_path)
            
            # Load scalers
            scalers_path = self.models_dir / "models" / "scalers.joblib"
            if scalers_path.exists():
                self.scalers = joblib.load(scalers_path)
            
            # Load metadata
            metadata = self._load_model_metadata(model_name)
            
            if metadata is None:
                logger.warning(f"No metadata found for model {model_name}")
                # Create basic metadata
                metadata = ModelMetadata(
                    name=model_name,
                    version="1.0.0",
                    training_date=datetime.utcnow(),
                    performance_metrics={},
                    feature_names=[],
                    model_type="unknown",
                    calibration_status="unknown",
                    file_path=str(model_path)
                )
            
            # Cache the model and metadata
            self._model_cache[cache_key] = model
            self._metadata_cache[cache_key] = metadata
            
            # Set current model
            self.current_model = model
            self.current_metadata = metadata
            self._set_model_attributes()
            
            logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def _load_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Load model metadata from results files"""
        try:
            # Try to load from individual model results
            results_file = self.models_dir / "results" / f"{model_name}_results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                
                # Load feature names from metadata
                metadata_file = self.models_dir / "pipeline_metadata.json"
                feature_names = []
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        pipeline_metadata = json.load(f)
                        feature_names = pipeline_metadata.get('feature_columns', [])
                
                return ModelMetadata(
                    name=model_name,
                    version="1.0.0",
                    training_date=datetime.utcnow(),
                    performance_metrics=results_data.get('test_metrics', {}),
                    feature_names=feature_names,
                    model_type=model_name,
                    calibration_status="calibrated",
                    file_path=str(self.models_dir / "models" / f"{model_name}.joblib")
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading metadata for {model_name}: {e}")
            return None
    
    def _set_model_attributes(self):
        """Set model attributes from current metadata"""
        if self.current_metadata:
            self.feature_names = self.current_metadata.feature_names
            self.model_version = self.current_metadata.version
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction using current model
        
        Args:
            features: Input features for prediction
            
        Returns:
            Prediction results with probabilities and metadata
        """
        if self.current_model is None:
            raise ValueError("No model loaded")
        
        try:
            # Ensure features are 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Apply preprocessing if scalers available
            if self.scalers and 'imputer' in self.scalers:
                features = self.scalers['imputer'].transform(features)
            
            if self.scalers and 'standard' in self.scalers:
                features = self.scalers['standard'].transform(features)
            
            # Make prediction
            prediction_proba = self.current_model.predict_proba(features)[0]
            
            # Extract probabilities
            if len(prediction_proba) == 2:
                risk_score = prediction_proba[1]  # Positive class probability
            else:
                risk_score = prediction_proba[0]
            
            # For calibrated models, the probability should already be calibrated
            calibrated_risk = risk_score
            
            # Calculate confidence interval (simplified)
            confidence_interval = self._calculate_confidence_interval(risk_score)
            
            return {
                'risk_score': float(risk_score),
                'calibrated_risk': float(calibrated_risk),
                'confidence_interval': confidence_interval,
                'model_version': self.model_version,
                'prediction_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Prediction failed: {e}")
    
    def _calculate_confidence_interval(self, probability: float, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for prediction (simplified)"""
        
        # Simplified confidence interval calculation
        # In production, this would use proper statistical methods
        margin = 0.05  # 5% margin
        
        lower = max(0.0, probability - margin)
        upper = min(1.0, probability + margin)
        
        return {
            'lower': lower,
            'upper': upper,
            'confidence_level': confidence
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.current_metadata is None:
            raise ValueError("No model loaded")
        
        return {
            'name': self.current_metadata.name,
            'version': self.current_metadata.version,
            'training_date': self.current_metadata.training_date,
            'metrics': self.current_metadata.performance_metrics,
            'feature_count': len(self.current_metadata.feature_names),
            'model_type': self.current_metadata.model_type,
            'calibration_status': self.current_metadata.calibration_status
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models in the registry"""
        models = []
        
        try:
            models_dir = self.models_dir / "models"
            if models_dir.exists():
                for model_file in models_dir.glob("*.joblib"):
                    if not model_file.name.endswith("_base.joblib"):  # Skip base models
                        model_name = model_file.stem
                        metadata = self._load_model_metadata(model_name)
                        
                        model_info = {
                            'name': model_name,
                            'file_path': str(model_file),
                            'file_size': model_file.stat().st_size,
                            'modified_date': datetime.fromtimestamp(model_file.stat().st_mtime)
                        }
                        
                        if metadata:
                            model_info.update({
                                'version': metadata.version,
                                'training_date': metadata.training_date,
                                'performance_metrics': metadata.performance_metrics,
                                'model_type': metadata.model_type
                            })
                        
                        models.append(model_info)
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def validate_model(self, model_name: str) -> Dict[str, Any]:
        """Validate a model's integrity and performance"""
        try:
            # Load model temporarily
            model_path = self.models_dir / "models" / f"{model_name}.joblib"
            
            if not model_path.exists():
                return {'valid': False, 'error': 'Model file not found'}
            
            # Try to load the model
            model = joblib.load(model_path)
            
            # Basic validation checks
            validation_results = {
                'valid': True,
                'model_name': model_name,
                'has_predict_method': hasattr(model, 'predict'),
                'has_predict_proba_method': hasattr(model, 'predict_proba'),
                'model_type': type(model).__name__,
                'file_size_mb': model_path.stat().st_size / (1024 * 1024)
            }
            
            # Load metadata if available
            metadata = self._load_model_metadata(model_name)
            if metadata:
                validation_results['metadata_available'] = True
                validation_results['feature_count'] = len(metadata.feature_names)
                validation_results['performance_metrics'] = metadata.performance_metrics
            else:
                validation_results['metadata_available'] = False
            
            return validation_results
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def get_model_performance_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get performance history for a model (if available)"""
        # This would typically connect to a model performance tracking system
        # For now, return basic information
        
        try:
            metadata = self._load_model_metadata(model_name)
            if metadata and metadata.performance_metrics:
                return [{
                    'timestamp': metadata.training_date,
                    'metrics': metadata.performance_metrics,
                    'version': metadata.version
                }]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting performance history for {model_name}: {e}")
            return []
    
    def register_new_model(
        self, 
        model, 
        model_name: str, 
        metadata: Dict[str, Any],
        replace_current: bool = False
    ) -> bool:
        """
        Register a new model in the registry
        
        Args:
            model: Trained model object
            model_name: Name for the model
            metadata: Model metadata
            replace_current: Whether to make this the current model
            
        Returns:
            True if registration successful
        """
        try:
            # Create models directory if it doesn't exist
            models_dir = self.models_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = models_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            
            # Create metadata object
            model_metadata = ModelMetadata(
                name=model_name,
                version=metadata.get('version', '1.0.0'),
                training_date=metadata.get('training_date', datetime.utcnow()),
                performance_metrics=metadata.get('performance_metrics', {}),
                feature_names=metadata.get('feature_names', []),
                model_type=metadata.get('model_type', type(model).__name__),
                calibration_status=metadata.get('calibration_status', 'unknown'),
                file_path=str(model_path)
            )
            
            # Save metadata
            results_dir = self.models_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_file = results_dir / f"{model_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'name': model_metadata.name,
                    'version': model_metadata.version,
                    'training_date': model_metadata.training_date.isoformat(),
                    'performance_metrics': model_metadata.performance_metrics,
                    'feature_names': model_metadata.feature_names,
                    'model_type': model_metadata.model_type,
                    'calibration_status': model_metadata.calibration_status
                }, f, indent=2)
            
            # Update cache
            cache_key = f"{model_name}_latest"
            self._model_cache[cache_key] = model
            self._metadata_cache[cache_key] = model_metadata
            
            # Set as current model if requested
            if replace_current:
                self.current_model = model
                self.current_metadata = model_metadata
                self._set_model_attributes()
            
            logger.info(f"Successfully registered model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
            return False
