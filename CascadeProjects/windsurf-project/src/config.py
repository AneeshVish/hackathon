"""Configuration management for the patient deterioration prediction system."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database Configuration
    database_url: str = Field(default="postgresql://user:password@localhost:5432/patient_ml_db")
    redis_url: str = Field(default="redis://localhost:6379/0")
    
    # MLflow Configuration
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_s3_endpoint_url: Optional[str] = Field(default="http://localhost:9000")
    mlflow_artifact_root: str = Field(default="s3://mlflow-artifacts")
    
    # API Configuration
    api_secret_key: str = Field(default="your-secret-key-here")
    api_algorithm: str = Field(default="HS256")
    api_access_token_expire_minutes: int = Field(default=30)
    
    # Feature Store
    feast_repo_path: str = Field(default="./feature_repo")
    
    # Monitoring
    prometheus_port: int = Field(default=8000)
    grafana_port: int = Field(default=3000)
    
    # Security
    encryption_key: str = Field(default="your-encryption-key-here")
    hipaa_audit_log_path: str = Field(default="./logs/audit.log")
    
    # External Services
    ehr_api_url: Optional[str] = Field(default="https://ehr-system.hospital.com/api")
    fhir_base_url: Optional[str] = Field(default="https://fhir.hospital.com/R4")
    
    # OAuth2 / SSO
    oauth2_client_id: Optional[str] = Field(default=None)
    oauth2_client_secret: Optional[str] = Field(default=None)
    oauth2_redirect_uri: str = Field(default="http://localhost:8000/auth/callback")
    
    # Model Configuration
    model_registry_path: str = Field(default="./models")
    default_model_version: str = Field(default="v1.0.0")
    prediction_threshold_high: float = Field(default=0.30)
    prediction_threshold_medium: float = Field(default=0.10)
    
    # Data Pipeline
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    airflow_home: str = Field(default="./airflow")
    data_lake_path: str = Field(default="./data/raw")
    processed_data_path: str = Field(default="./data/processed")
    
    # Clinical Configuration
    deterioration_lookback_days: int = Field(default=90)
    feature_window_days: int = Field(default=180)
    prediction_horizon_days: int = Field(default=90)
    
    # Compliance
    audit_retention_days: int = Field(default=2555)  # 7 years
    de_identification_enabled: bool = Field(default=True)
    consent_required: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


class ClinicalConfig:
    """Clinical domain-specific configuration."""
    
    # Deterioration event definitions
    DETERIORATION_EVENTS = {
        "unplanned_hospitalization": True,
        "ed_visit": True,
        "mortality": True,
        "egfr_drop_20_percent": True,
        "custom_clinical_events": []
    }
    
    # Vital sign normal ranges
    VITAL_RANGES = {
        "heart_rate": {"min": 60, "max": 100},
        "systolic_bp": {"min": 90, "max": 140},
        "diastolic_bp": {"min": 60, "max": 90},
        "temperature": {"min": 36.1, "max": 37.2},
        "respiratory_rate": {"min": 12, "max": 20},
        "spo2": {"min": 95, "max": 100}
    }
    
    # Lab reference ranges (example values)
    LAB_RANGES = {
        "creatinine": {"min": 0.6, "max": 1.2, "units": "mg/dL"},
        "bnp": {"min": 0, "max": 100, "units": "pg/mL"},
        "hba1c": {"min": 4.0, "max": 5.6, "units": "%"},
        "hemoglobin": {"min": 12.0, "max": 16.0, "units": "g/dL"}
    }
    
    # Risk stratification thresholds
    RISK_BUCKETS = {
        "low": {"min": 0.0, "max": 0.10, "color": "green"},
        "medium": {"min": 0.10, "max": 0.30, "color": "yellow"},
        "high": {"min": 0.30, "max": 1.0, "color": "red"}
    }
    
    # Feature aggregation windows
    AGGREGATION_WINDOWS = [7, 14, 30, 60, 90]  # days
    
    # Medication adherence thresholds
    ADHERENCE_THRESHOLDS = {
        "poor": 0.6,
        "moderate": 0.8,
        "good": 0.9
    }


clinical_config = ClinicalConfig()
