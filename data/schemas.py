"""
Data schemas for patient deterioration prediction system.
Defines Pydantic models for all data types following clinical standards.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class GenderEnum(str, Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class RiskBucketEnum(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class EventTypeEnum(str, Enum):
    HOSPITALIZATION = "hospitalization"
    ED_VISIT = "ed_visit"
    MORTALITY = "mortality"
    CLINICAL_EVENT = "clinical_event"


# Core Patient Demographics
class PatientDemographics(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    date_of_birth: date = Field(..., description="Patient date of birth")
    gender: GenderEnum = Field(..., description="Patient gender")
    zip_code: Optional[str] = Field(None, description="Patient zip code")
    care_team_id: Optional[str] = Field(None, description="Assigned care team")
    comorbidities: List[str] = Field(default_factory=list, description="ICD-10 codes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Vital Signs
class VitalSigns(BaseModel):
    patient_id: str
    timestamp: datetime
    heart_rate: Optional[float] = Field(None, ge=30, le=300, description="BPM")
    systolic_bp: Optional[float] = Field(None, ge=60, le=300, description="mmHg")
    diastolic_bp: Optional[float] = Field(None, ge=30, le=200, description="mmHg")
    weight: Optional[float] = Field(None, ge=20, le=500, description="kg")
    spo2: Optional[float] = Field(None, ge=70, le=100, description="Oxygen saturation %")
    respiratory_rate: Optional[float] = Field(None, ge=8, le=60, description="breaths/min")
    temperature: Optional[float] = Field(None, ge=32, le=45, description="Celsius")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Laboratory Results
class LabResult(BaseModel):
    patient_id: str
    timestamp: datetime
    lab_name: str = Field(..., description="Standardized lab name")
    value: float = Field(..., description="Numeric lab value")
    units: str = Field(..., description="Units of measurement")
    reference_range_low: Optional[float] = None
    reference_range_high: Optional[float] = None
    abnormal_flag: Optional[str] = Field(None, description="H/L/N for High/Low/Normal")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Medications
class Medication(BaseModel):
    patient_id: str
    medication_name: str = Field(..., description="Generic medication name")
    dose: str = Field(..., description="Dose and frequency")
    start_date: date = Field(..., description="Medication start date")
    end_date: Optional[date] = Field(None, description="Medication end date")
    adherence_pct: Optional[float] = Field(None, ge=0, le=100, description="Adherence percentage")
    prescriber_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Healthcare Encounters
class Encounter(BaseModel):
    patient_id: str
    encounter_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    encounter_type: str = Field(..., description="outpatient/inpatient/emergency")
    start_datetime: datetime
    end_datetime: Optional[datetime] = None
    primary_diagnosis: Optional[str] = Field(None, description="ICD-10 code")
    secondary_diagnoses: List[str] = Field(default_factory=list)
    provider_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Lifestyle and Patient-Reported Outcomes
class LifestyleData(BaseModel):
    patient_id: str
    timestamp: datetime
    activity_minutes: Optional[int] = Field(None, ge=0, description="Daily activity minutes")
    steps: Optional[int] = Field(None, ge=0, description="Daily step count")
    smoking_status: Optional[str] = Field(None, description="never/former/current")
    sleep_hours: Optional[float] = Field(None, ge=0, le=24, description="Hours of sleep")
    sleep_quality: Optional[int] = Field(None, ge=1, le=10, description="Self-reported 1-10")
    diet_quality: Optional[int] = Field(None, ge=1, le=10, description="Self-reported 1-10")
    stress_level: Optional[int] = Field(None, ge=1, le=10, description="Self-reported 1-10")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Device Data (Wearables, CGM, etc.)
class DeviceData(BaseModel):
    patient_id: str
    device_type: str = Field(..., description="cgm/fitbit/apple_watch/etc")
    timestamp: datetime
    metric_name: str = Field(..., description="glucose/heart_rate/steps/etc")
    value: float
    units: str
    device_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Target Labels for ML
class DeteriorationEvent(BaseModel):
    patient_id: str
    event_date: datetime
    event_type: EventTypeEnum
    event_description: str
    severity: Optional[str] = Field(None, description="mild/moderate/severe")
    outcome: Optional[str] = Field(None, description="Additional outcome details")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Feature Store Schema
class PatientFeatures(BaseModel):
    patient_id: str
    feature_timestamp: datetime
    
    # Demographic features
    age: float
    gender_encoded: int
    comorbidity_count: int
    charlson_score: Optional[float] = None
    
    # Vital trends (last 30 days)
    hr_mean: Optional[float] = None
    hr_std: Optional[float] = None
    hr_trend: Optional[float] = None
    bp_systolic_mean: Optional[float] = None
    bp_systolic_trend: Optional[float] = None
    weight_change_30d: Optional[float] = None
    weight_trend: Optional[float] = None
    
    # Lab features
    bnp_last: Optional[float] = None
    bnp_change_pct: Optional[float] = None
    creatinine_last: Optional[float] = None
    creatinine_change_pct: Optional[float] = None
    hba1c_last: Optional[float] = None
    
    # Medication features
    medication_count: int = 0
    adherence_mean_30d: Optional[float] = None
    recent_med_changes: int = 0
    
    # Utilization features
    ed_visits_90d: int = 0
    admissions_365d: int = 0
    missed_appointments_90d: int = 0
    
    # Lifestyle features
    activity_mean_30d: Optional[float] = None
    sleep_quality_mean_30d: Optional[float] = None
    
    # Temporal features
    days_since_last_visit: Optional[int] = None
    season: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Prediction Output Schema
class PredictionResult(BaseModel):
    patient_id: str
    timestamp: datetime
    risk_score: float = Field(..., ge=0, le=1, description="Raw model probability")
    calibrated_risk: float = Field(..., ge=0, le=1, description="Calibrated probability")
    risk_bucket: RiskBucketEnum
    confidence_interval: Optional[Dict[str, float]] = None
    
    # Explainability
    top_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    shap_values: Optional[Dict[str, float]] = None
    
    # Model metadata
    model_version: str
    model_name: str
    feature_count: int
    
    # Clinical recommendations
    recommended_actions: List[str] = Field(default_factory=list)
    urgency_level: Optional[str] = Field(None, description="low/medium/high/urgent")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


# API Request/Response Models
class PredictionRequest(BaseModel):
    patient_id: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    feature_override: Optional[Dict[str, Any]] = None


class CohortRequest(BaseModel):
    clinic_id: Optional[str] = None
    care_team_id: Optional[str] = None
    risk_bucket: Optional[RiskBucketEnum] = None
    min_risk: Optional[float] = Field(None, ge=0, le=1)
    max_risk: Optional[float] = Field(None, ge=0, le=1)
    last_visit_days: Optional[int] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class CohortResponse(BaseModel):
    patients: List[Dict[str, Any]]
    total_count: int
    risk_distribution: Dict[str, int]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# Feedback and Audit
class ClinicalFeedback(BaseModel):
    prediction_id: str
    patient_id: str
    clinician_id: str
    feedback_type: str = Field(..., description="outcome/correction/rating")
    actual_outcome: Optional[bool] = None
    outcome_date: Optional[datetime] = None
    feedback_text: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AuditLog(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    patient_id: Optional[str] = None
    action: str = Field(..., description="predict/view/export/etc")
    resource: str = Field(..., description="API endpoint or resource accessed")
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
