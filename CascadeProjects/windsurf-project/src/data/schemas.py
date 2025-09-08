"""Data schemas for patient deterioration prediction system."""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class GenderEnum(str, Enum):
    """Patient gender enumeration."""
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class EventTypeEnum(str, Enum):
    """Clinical event types."""
    HOSPITALIZATION = "hospitalization"
    ED_VISIT = "ed_visit"
    OUTPATIENT = "outpatient"
    MORTALITY = "mortality"
    LAB_CRITICAL = "lab_critical"


class AdherenceSourceEnum(str, Enum):
    """Medication adherence data sources."""
    REFILL = "refill"
    DIGITAL_PILL = "digital_pill"
    SELF_REPORT = "self_report"
    PROVIDER_ASSESSMENT = "provider_assessment"


# Core Patient Data Models

class Patient(BaseModel):
    """Patient demographic and baseline information."""
    patient_id: str = Field(..., description="Unique patient identifier")
    date_of_birth: date = Field(..., description="Patient date of birth")
    gender: GenderEnum = Field(..., description="Patient gender")
    zip_code: Optional[str] = Field(None, description="Patient zip code")
    care_team_id: Optional[str] = Field(None, description="Assigned care team")
    enrollment_date: datetime = Field(..., description="System enrollment date")
    
    # Comorbidities (ICD-10 codes)
    comorbidities: List[str] = Field(default_factory=list, description="ICD-10 diagnosis codes")
    
    # Social determinants
    living_situation: Optional[str] = Field(None, description="Living arrangement")
    insurance_type: Optional[str] = Field(None, description="Insurance coverage type")
    
    @validator('patient_id')
    def validate_patient_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Patient ID must be at least 3 characters')
        return v


class VitalSigns(BaseModel):
    """Patient vital signs measurement."""
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: datetime = Field(..., description="Measurement timestamp")
    
    # Core vitals
    heart_rate: Optional[float] = Field(None, ge=0, le=300, description="Heart rate (bpm)")
    systolic_bp: Optional[float] = Field(None, ge=50, le=300, description="Systolic BP (mmHg)")
    diastolic_bp: Optional[float] = Field(None, ge=30, le=200, description="Diastolic BP (mmHg)")
    temperature: Optional[float] = Field(None, ge=30, le=45, description="Temperature (°C)")
    respiratory_rate: Optional[float] = Field(None, ge=5, le=60, description="Respiratory rate (breaths/min)")
    spo2: Optional[float] = Field(None, ge=50, le=100, description="Oxygen saturation (%)")
    weight: Optional[float] = Field(None, ge=20, le=300, description="Weight (kg)")
    
    # Metadata
    measurement_source: Optional[str] = Field(None, description="Device or manual entry")
    quality_flag: Optional[str] = Field(None, description="Data quality indicator")


class LabResult(BaseModel):
    """Laboratory test result."""
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: datetime = Field(..., description="Collection timestamp")
    
    lab_name: str = Field(..., description="Laboratory test name")
    value: float = Field(..., description="Numeric result value")
    units: str = Field(..., description="Units of measurement")
    reference_range_low: Optional[float] = Field(None, description="Lower reference limit")
    reference_range_high: Optional[float] = Field(None, description="Upper reference limit")
    
    # Flags
    abnormal_flag: Optional[str] = Field(None, description="H/L/Critical flags")
    lab_source: Optional[str] = Field(None, description="Laboratory system")


class Medication(BaseModel):
    """Medication prescription and administration."""
    patient_id: str = Field(..., description="Patient identifier")
    medication_name: str = Field(..., description="Generic medication name")
    
    # Prescription details
    dose: Optional[str] = Field(None, description="Dose strength")
    frequency: Optional[str] = Field(None, description="Dosing frequency")
    route: Optional[str] = Field(None, description="Administration route")
    
    # Dates
    start_date: datetime = Field(..., description="Prescription start date")
    end_date: Optional[datetime] = Field(None, description="Prescription end date")
    
    # Clinical context
    indication: Optional[str] = Field(None, description="Indication for prescription")
    prescriber_id: Optional[str] = Field(None, description="Prescribing provider")


class MedicationAdherence(BaseModel):
    """Medication adherence tracking."""
    patient_id: str = Field(..., description="Patient identifier")
    medication_name: str = Field(..., description="Medication name")
    
    # Adherence metrics
    measurement_period_start: datetime = Field(..., description="Measurement period start")
    measurement_period_end: datetime = Field(..., description="Measurement period end")
    adherence_percentage: float = Field(..., ge=0, le=100, description="Adherence percentage")
    
    # Source and method
    adherence_source: AdherenceSourceEnum = Field(..., description="Data source")
    days_covered: Optional[int] = Field(None, description="Days with medication available")
    total_days: Optional[int] = Field(None, description="Total days in period")


class ClinicalEncounter(BaseModel):
    """Clinical encounter or visit."""
    patient_id: str = Field(..., description="Patient identifier")
    encounter_id: str = Field(..., description="Unique encounter identifier")
    
    # Encounter details
    encounter_type: EventTypeEnum = Field(..., description="Type of encounter")
    start_datetime: datetime = Field(..., description="Encounter start time")
    end_datetime: Optional[datetime] = Field(None, description="Encounter end time")
    
    # Clinical context
    primary_diagnosis: Optional[str] = Field(None, description="Primary ICD-10 diagnosis")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses")
    provider_id: Optional[str] = Field(None, description="Attending provider")
    department: Optional[str] = Field(None, description="Hospital department")
    
    # Outcome
    discharge_disposition: Optional[str] = Field(None, description="Discharge disposition")


class LifestyleData(BaseModel):
    """Patient lifestyle and self-reported data."""
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: datetime = Field(..., description="Data collection timestamp")
    
    # Activity and exercise
    activity_minutes: Optional[int] = Field(None, ge=0, description="Daily activity minutes")
    steps: Optional[int] = Field(None, ge=0, description="Daily step count")
    
    # Health behaviors
    smoking_status: Optional[str] = Field(None, description="Current smoking status")
    alcohol_use: Optional[str] = Field(None, description="Alcohol consumption pattern")
    
    # Sleep and wellness
    sleep_hours: Optional[float] = Field(None, ge=0, le=24, description="Hours of sleep")
    sleep_quality: Optional[int] = Field(None, ge=1, le=10, description="Sleep quality score")
    
    # Self-reported symptoms
    pain_score: Optional[int] = Field(None, ge=0, le=10, description="Pain level (0-10)")
    fatigue_score: Optional[int] = Field(None, ge=0, le=10, description="Fatigue level (0-10)")


class DeviceData(BaseModel):
    """Wearable device and remote monitoring data."""
    patient_id: str = Field(..., description="Patient identifier")
    device_type: str = Field(..., description="Device type (CGM, fitness tracker, etc.)")
    timestamp: datetime = Field(..., description="Measurement timestamp")
    
    # Glucose monitoring
    glucose_value: Optional[float] = Field(None, ge=20, le=600, description="Blood glucose (mg/dL)")
    
    # Cardiac monitoring
    heart_rate_variability: Optional[float] = Field(None, description="HRV measurement")
    
    # Activity tracking
    activity_level: Optional[str] = Field(None, description="Activity intensity level")
    
    # Device metadata
    device_id: Optional[str] = Field(None, description="Unique device identifier")
    battery_level: Optional[float] = Field(None, ge=0, le=100, description="Device battery %")


# Target Labels and Outcomes

class DeteriorationEvent(BaseModel):
    """Patient deterioration event (target label)."""
    patient_id: str = Field(..., description="Patient identifier")
    event_timestamp: datetime = Field(..., description="When deterioration occurred")
    event_type: EventTypeEnum = Field(..., description="Type of deterioration event")
    
    # Event details
    severity: Optional[str] = Field(None, description="Event severity classification")
    description: Optional[str] = Field(None, description="Clinical description")
    
    # Outcome tracking
    length_of_stay: Optional[int] = Field(None, description="Hospital LOS (days)")
    icu_admission: Optional[bool] = Field(None, description="ICU admission required")
    
    # Validation
    prediction_window_start: datetime = Field(..., description="Start of prediction window")
    prediction_window_end: datetime = Field(..., description="End of prediction window")
    
    @validator('event_timestamp')
    def validate_event_in_window(cls, v, values):
        if 'prediction_window_start' in values and 'prediction_window_end' in values:
            if not (values['prediction_window_start'] <= v <= values['prediction_window_end']):
                raise ValueError('Event must occur within prediction window')
        return v


# Feature Engineering Schemas

class PatientFeatures(BaseModel):
    """Engineered features for a patient at a specific time point."""
    patient_id: str = Field(..., description="Patient identifier")
    feature_timestamp: datetime = Field(..., description="Feature computation timestamp")
    
    # Demographic features
    age_years: float = Field(..., description="Patient age in years")
    gender_encoded: int = Field(..., description="Gender encoding")
    
    # Vital trends (30-day windows)
    weight_trend_30d: Optional[float] = Field(None, description="Weight slope (kg/day)")
    weight_change_30d: Optional[float] = Field(None, description="Weight change (kg)")
    bp_systolic_mean_30d: Optional[float] = Field(None, description="Mean systolic BP")
    bp_variability_30d: Optional[float] = Field(None, description="BP standard deviation")
    hr_mean_30d: Optional[float] = Field(None, description="Mean heart rate")
    
    # Lab deltas
    creatinine_change_90d: Optional[float] = Field(None, description="Creatinine change")
    bnp_change_90d: Optional[float] = Field(None, description="BNP change")
    bnp_last_value: Optional[float] = Field(None, description="Most recent BNP")
    
    # Medication features
    medication_count: int = Field(default=0, description="Active medication count")
    adherence_mean_30d: Optional[float] = Field(None, description="Mean adherence %")
    recent_med_changes: int = Field(default=0, description="Med changes in 30d")
    
    # Utilization features
    ed_visits_90d: int = Field(default=0, description="ED visits in 90 days")
    hospitalizations_365d: int = Field(default=0, description="Hospitalizations in 1 year")
    missed_appointments_90d: int = Field(default=0, description="Missed appointments")
    
    # Comorbidity scores
    charlson_score: Optional[float] = Field(None, description="Charlson comorbidity index")
    
    # Data completeness
    vitals_completeness_30d: float = Field(default=0.0, description="% vitals available")
    labs_completeness_90d: float = Field(default=0.0, description="% labs available")
    
    # Target label
    deterioration_90d: Optional[bool] = Field(None, description="Deterioration in 90 days")


# API Response Schemas

class RiskPrediction(BaseModel):
    """Risk prediction API response."""
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    # Risk scores
    risk: float = Field(..., ge=0, le=1, description="Raw model probability")
    calibrated_risk: float = Field(..., ge=0, le=1, description="Calibrated probability")
    risk_bucket: str = Field(..., description="Risk stratification (Low/Medium/High)")
    
    # Explanations
    top_drivers: List[Dict[str, Any]] = Field(..., description="Top risk drivers")
    
    # Model metadata
    model_version: str = Field(..., description="Model version used")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Prediction CI")
    
    # Clinical context
    recommended_actions: List[str] = Field(default_factory=list, description="Suggested actions")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P1234",
                "timestamp": "2025-09-08T10:02:00Z",
                "risk": 0.72,
                "calibrated_risk": 0.70,
                "risk_bucket": "High",
                "top_drivers": [
                    {
                        "feature": "BNP_last_change",
                        "impact": 0.18,
                        "explanation": "BNP increased 40% over 14 days"
                    }
                ],
                "model_version": "v1.2.0",
                "recommended_actions": [
                    "Schedule nurse phone visit within 48 hours",
                    "Consider CHF medication review"
                ]
            }
        }


class CohortSummary(BaseModel):
    """Cohort-level summary statistics."""
    total_patients: int = Field(..., description="Total patients in cohort")
    high_risk_count: int = Field(..., description="High risk patients")
    medium_risk_count: int = Field(..., description="Medium risk patients")
    low_risk_count: int = Field(..., description="Low risk patients")
    
    # Summary statistics
    mean_risk: float = Field(..., description="Average risk score")
    last_updated: datetime = Field(..., description="Last update timestamp")
    
    # Recent events
    recent_admissions: int = Field(..., description="Admissions in last 7 days")
    pending_actions: int = Field(..., description="Pending clinical actions")
