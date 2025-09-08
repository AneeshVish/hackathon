"""
FastAPI inference service for patient deterioration prediction.
Provides REST API endpoints for predictions, explanations, and model management.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import joblib
import json
from pathlib import Path
import os
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from data.schemas import PredictionRequest, PredictionResult, CohortRequest, CohortResponse
from models.explainability import ExplainabilityEngine
from .model_registry import ModelRegistry
from .security import SecurityManager
from .monitoring import MonitoringService
from .audit import AuditLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and services
model_registry = None
explainer = None
security_manager = None
monitoring_service = None
audit_logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global model_registry, explainer, security_manager, monitoring_service, audit_logger
    
    # Startup
    logger.info("Starting Patient Deterioration Prediction API")
    
    # Initialize services
    model_registry = ModelRegistry()
    explainer = ExplainabilityEngine()
    security_manager = SecurityManager()
    monitoring_service = MonitoringService()
    audit_logger = AuditLogger()
    
    # Load default model
    try:
        model_registry.load_default_model()
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Patient Deterioration Prediction API")


# Initialize FastAPI app
app = FastAPI(
    title="Patient Deterioration Prediction API",
    description="Secure, explainable ML service for predicting patient deterioration risk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-dashboard-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


# Request/Response Models
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    model_status: str


class PredictionResponse(BaseModel):
    patient_id: str
    timestamp: datetime
    risk_score: float = Field(..., ge=0, le=1)
    calibrated_risk: float = Field(..., ge=0, le=1)
    risk_bucket: str
    confidence_interval: Optional[Dict[str, float]] = None
    top_drivers: List[Dict[str, Any]]
    recommended_actions: List[str]
    urgency_level: str
    model_version: str
    explanation_id: str


class BatchPredictionRequest(BaseModel):
    patient_ids: List[str]
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    include_explanations: bool = True


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    feature_count: int
    calibration_status: str


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate JWT token and return user info"""
    try:
        user_info = security_manager.validate_token(credentials.credentials)
        return user_info
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


async def check_patient_access(patient_id: str, user_info: dict):
    """Check if user has access to specific patient"""
    if not security_manager.check_patient_access(user_info, patient_id):
        raise HTTPException(status_code=403, detail="Access denied for this patient")


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_status = "healthy" if model_registry and model_registry.current_model else "no_model"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        model_status=model_status
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_deterioration(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(get_current_user)
):
    """
    Predict deterioration risk for a single patient
    """
    try:
        # Check patient access
        await check_patient_access(request.patient_id, user_info)
        
        # Audit log
        audit_logger.log_prediction_request(user_info['user_id'], request.patient_id)
        
        # Get patient features
        patient_features = await _get_patient_features(request.patient_id, request.timestamp)
        
        if patient_features is None:
            raise HTTPException(status_code=404, detail="Patient data not found")
        
        # Make prediction
        prediction_result = model_registry.predict(patient_features)
        
        # Generate explanations
        explanations = explainer.generate_local_explanations(
            model_registry.current_model,
            patient_features,
            model_registry.feature_names,
            request.patient_id
        )
        
        # Generate recommended actions
        recommended_actions = _generate_recommended_actions(
            prediction_result['risk_score'],
            explanations['top_contributors']
        )
        
        # Create response
        response = PredictionResponse(
            patient_id=request.patient_id,
            timestamp=datetime.utcnow(),
            risk_score=prediction_result['risk_score'],
            calibrated_risk=prediction_result['calibrated_risk'],
            risk_bucket=_categorize_risk(prediction_result['calibrated_risk']),
            top_drivers=explanations['top_contributors'][:5],
            recommended_actions=recommended_actions,
            urgency_level=_determine_urgency(prediction_result['calibrated_risk']),
            model_version=model_registry.model_version,
            explanation_id=f"exp_{request.patient_id}_{int(datetime.utcnow().timestamp())}"
        )
        
        # Background tasks
        background_tasks.add_task(
            monitoring_service.log_prediction,
            request.patient_id,
            prediction_result['calibrated_risk'],
            user_info['user_id']
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for patient {request.patient_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch")
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(get_current_user)
):
    """
    Batch prediction for multiple patients
    """
    try:
        results = []
        
        for patient_id in request.patient_ids:
            try:
                # Check access for each patient
                await check_patient_access(patient_id, user_info)
                
                # Get features and predict
                patient_features = await _get_patient_features(patient_id, request.timestamp)
                
                if patient_features is not None:
                    prediction_result = model_registry.predict(patient_features)
                    
                    result = {
                        'patient_id': patient_id,
                        'risk_score': prediction_result['risk_score'],
                        'calibrated_risk': prediction_result['calibrated_risk'],
                        'risk_bucket': _categorize_risk(prediction_result['calibrated_risk']),
                        'status': 'success'
                    }
                    
                    if request.include_explanations:
                        explanations = explainer.generate_local_explanations(
                            model_registry.current_model,
                            patient_features,
                            model_registry.feature_names,
                            patient_id
                        )
                        result['top_drivers'] = explanations['top_contributors'][:3]
                    
                    results.append(result)
                else:
                    results.append({
                        'patient_id': patient_id,
                        'status': 'no_data',
                        'error': 'Patient data not found'
                    })
                    
            except Exception as e:
                results.append({
                    'patient_id': patient_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Log batch prediction
        background_tasks.add_task(
            audit_logger.log_batch_prediction,
            user_info['user_id'],
            len(request.patient_ids),
            len([r for r in results if r['status'] == 'success'])
        )
        
        return {
            'timestamp': datetime.utcnow(),
            'total_requested': len(request.patient_ids),
            'successful_predictions': len([r for r in results if r['status'] == 'success']),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.get("/cohort")
async def get_cohort_predictions(
    clinic_id: Optional[str] = None,
    care_team_id: Optional[str] = None,
    risk_bucket: Optional[str] = None,
    min_risk: Optional[float] = None,
    max_risk: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    user_info: dict = Depends(get_current_user)
):
    """
    Get cohort-level predictions with filtering
    """
    try:
        # Build filters based on user permissions
        filters = security_manager.build_cohort_filters(user_info, {
            'clinic_id': clinic_id,
            'care_team_id': care_team_id,
            'risk_bucket': risk_bucket,
            'min_risk': min_risk,
            'max_risk': max_risk
        })
        
        # Get cohort data
        cohort_data = await _get_cohort_data(filters, limit, offset)
        
        # Calculate risk distribution
        risk_distribution = _calculate_risk_distribution(cohort_data)
        
        response = CohortResponse(
            patients=cohort_data,
            total_count=len(cohort_data),
            risk_distribution=risk_distribution
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Cohort query error: {e}")
        raise HTTPException(status_code=500, detail="Cohort query failed")


@app.get("/patient/{patient_id}/features")
async def get_patient_features(
    patient_id: str,
    timestamp: Optional[datetime] = None,
    user_info: dict = Depends(get_current_user)
):
    """
    Get patient features used for prediction
    """
    try:
        await check_patient_access(patient_id, user_info)
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        features = await _get_patient_features(patient_id, timestamp)
        
        if features is None:
            raise HTTPException(status_code=404, detail="Patient features not found")
        
        # Convert to readable format
        feature_dict = {}
        for i, feature_name in enumerate(model_registry.feature_names):
            feature_dict[feature_name] = float(features[i])
        
        return {
            'patient_id': patient_id,
            'timestamp': timestamp,
            'features': feature_dict,
            'feature_count': len(feature_dict)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting features for patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve patient features")


@app.post("/feedback")
async def submit_clinical_feedback(
    prediction_id: str,
    patient_id: str,
    feedback_type: str,
    actual_outcome: Optional[bool] = None,
    outcome_date: Optional[datetime] = None,
    feedback_text: Optional[str] = None,
    rating: Optional[int] = None,
    user_info: dict = Depends(get_current_user)
):
    """
    Submit clinical feedback for model improvement
    """
    try:
        await check_patient_access(patient_id, user_info)
        
        feedback_data = {
            'prediction_id': prediction_id,
            'patient_id': patient_id,
            'clinician_id': user_info['user_id'],
            'feedback_type': feedback_type,
            'actual_outcome': actual_outcome,
            'outcome_date': outcome_date,
            'feedback_text': feedback_text,
            'rating': rating,
            'timestamp': datetime.utcnow()
        }
        
        # Store feedback
        await _store_clinical_feedback(feedback_data)
        
        # Log feedback submission
        audit_logger.log_feedback_submission(
            user_info['user_id'], patient_id, feedback_type
        )
        
        return {
            'status': 'success',
            'message': 'Feedback submitted successfully',
            'timestamp': datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(user_info: dict = Depends(get_current_user)):
    """
    Get current model information
    """
    try:
        model_info = model_registry.get_model_info()
        
        return ModelInfoResponse(
            model_name=model_info['name'],
            model_version=model_info['version'],
            training_date=model_info['training_date'],
            performance_metrics=model_info['metrics'],
            feature_count=len(model_registry.feature_names),
            calibration_status=model_info['calibration_status']
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@app.get("/metrics")
async def get_service_metrics(user_info: dict = Depends(get_current_user)):
    """
    Get service performance metrics
    """
    try:
        # Check admin permissions
        if not security_manager.is_admin(user_info):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        metrics = monitoring_service.get_metrics()
        
        return {
            'timestamp': datetime.utcnow(),
            'metrics': metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


# Helper functions
async def _get_patient_features(patient_id: str, timestamp: datetime) -> Optional[np.ndarray]:
    """
    Retrieve patient features from feature store
    This would typically connect to your feature store (Redis, database, etc.)
    """
    try:
        # Mock implementation - replace with actual feature store connection
        # In production, this would query your feature store
        
        # For demo purposes, return mock features
        if patient_id.startswith("DEMO"):
            # Generate mock features for demo
            np.random.seed(hash(patient_id) % 2**32)
            features = np.random.randn(len(model_registry.feature_names) if model_registry.feature_names else 50)
            return features
        
        # In production, implement actual feature retrieval
        # Example:
        # features = feature_store.get_patient_features(patient_id, timestamp)
        # return features
        
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving features for patient {patient_id}: {e}")
        return None


async def _get_cohort_data(filters: Dict[str, Any], limit: int, offset: int) -> List[Dict[str, Any]]:
    """
    Get cohort data based on filters
    """
    # Mock implementation - replace with actual database query
    cohort_data = []
    
    # Generate mock cohort data for demo
    for i in range(min(limit, 50)):
        patient_id = f"DEMO_PATIENT_{i + offset:03d}"
        risk_score = np.random.beta(2, 5)  # Skewed towards lower risk
        
        cohort_data.append({
            'patient_id': patient_id,
            'name': f"Patient {i + offset:03d}",
            'age': np.random.randint(45, 85),
            'last_visit': (datetime.utcnow() - timedelta(days=np.random.randint(1, 30))).isoformat(),
            'risk_score': risk_score,
            'risk_bucket': _categorize_risk(risk_score),
            'top_driver': np.random.choice(['BNP elevation', 'Weight gain', 'Poor adherence', 'Recent ED visit'])
        })
    
    return cohort_data


def _calculate_risk_distribution(cohort_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate risk distribution for cohort"""
    distribution = {'Low': 0, 'Medium': 0, 'High': 0}
    
    for patient in cohort_data:
        risk_bucket = patient.get('risk_bucket', 'Low')
        distribution[risk_bucket] += 1
    
    return distribution


def _categorize_risk(risk_score: float) -> str:
    """Categorize risk score into buckets"""
    if risk_score >= 0.30:
        return "High"
    elif risk_score >= 0.10:
        return "Medium"
    else:
        return "Low"


def _determine_urgency(risk_score: float) -> str:
    """Determine urgency level based on risk score"""
    if risk_score >= 0.50:
        return "urgent"
    elif risk_score >= 0.30:
        return "high"
    elif risk_score >= 0.15:
        return "medium"
    else:
        return "low"


def _generate_recommended_actions(risk_score: float, top_contributors: List[Dict]) -> List[str]:
    """Generate clinical recommendations based on risk and drivers"""
    actions = []
    
    if risk_score >= 0.30:
        actions.append("Schedule urgent clinical assessment within 48 hours")
        actions.append("Consider telehealth check-in within 24 hours")
    elif risk_score >= 0.15:
        actions.append("Schedule routine follow-up within 1-2 weeks")
        actions.append("Monitor patient remotely")
    
    # Add specific actions based on top contributors
    for contributor in top_contributors[:3]:
        feature = contributor.get('feature', '').lower()
        
        if 'bnp' in feature and contributor.get('shap_value', 0) > 0:
            actions.append("Consider cardiology consultation for elevated BNP")
        elif 'adherence' in feature and contributor.get('shap_value', 0) > 0:
            actions.append("Medication adherence counseling recommended")
        elif 'weight' in feature and 'change' in feature:
            actions.append("Monitor fluid status and consider diuretic adjustment")
    
    return list(set(actions))  # Remove duplicates


async def _store_clinical_feedback(feedback_data: Dict[str, Any]):
    """Store clinical feedback in database"""
    # Mock implementation - replace with actual database storage
    logger.info(f"Storing feedback: {feedback_data}")
    # In production: database.store_feedback(feedback_data)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
