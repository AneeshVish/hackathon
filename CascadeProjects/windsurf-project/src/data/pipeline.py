"""Data ingestion and ETL pipeline for patient deterioration prediction."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import redis

from ..config import settings, clinical_config
from .schemas import (
    Patient, VitalSigns, LabResult, Medication, MedicationAdherence,
    ClinicalEncounter, LifestyleData, DeviceData, DeteriorationEvent,
    PatientFeatures
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Data quality validation and cleaning."""
    
    def __init__(self):
        self.vital_ranges = clinical_config.VITAL_RANGES
        self.lab_ranges = clinical_config.LAB_RANGES
    
    def validate_vitals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean vital signs data."""
        logger.info(f"Validating {len(df)} vital sign records")
        
        # Remove implausible values
        for vital, ranges in self.vital_ranges.items():
            if vital in df.columns:
                mask = (df[vital] >= ranges['min']) & (df[vital] <= ranges['max'])
                invalid_count = (~mask & df[vital].notna()).sum()
                if invalid_count > 0:
                    logger.warning(f"Removing {invalid_count} invalid {vital} values")
                    df.loc[~mask, vital] = np.nan
        
        # Flag suspicious patterns
        df['quality_flag'] = 'normal'
        
        # Detect potential measurement errors (e.g., HR = 600 instead of 60)
        if 'heart_rate' in df.columns:
            hr_high_mask = df['heart_rate'] > 200
            df.loc[hr_high_mask, 'quality_flag'] = 'suspicious_high'
        
        return df
    
    def validate_labs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate laboratory results."""
        logger.info(f"Validating {len(df)} lab records")
        
        # Check for negative values where inappropriate
        negative_mask = df['value'] < 0
        if negative_mask.any():
            logger.warning(f"Found {negative_mask.sum()} negative lab values")
            # Keep negative values for some labs (e.g., base excess) but flag others
            
        # Flag extreme values
        df['abnormal_flag'] = 'normal'
        
        # Lab-specific validation
        for _, row in df.iterrows():
            lab_name = row['lab_name'].lower()
            value = row['value']
            
            if lab_name in self.lab_ranges:
                ranges = self.lab_ranges[lab_name]
                if value < ranges['min'] * 0.1 or value > ranges['max'] * 10:
                    df.loc[df.index == row.name, 'abnormal_flag'] = 'extreme'
        
        return df
    
    def check_data_completeness(self, df: pd.DataFrame, 
                              required_columns: List[str]) -> Dict[str, float]:
        """Calculate data completeness metrics."""
        completeness = {}
        for col in required_columns:
            if col in df.columns:
                completeness[col] = (1 - df[col].isnull().mean()) * 100
            else:
                completeness[col] = 0.0
        
        return completeness


class DataProcessor:
    """Core data processing and transformation."""
    
    def __init__(self):
        self.validator = DataValidator()
        
        # Database connection
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Redis for caching
        self.redis_client = redis.from_url(settings.redis_url)
    
    def ingest_raw_data(self, data_source: str, file_path: str) -> pd.DataFrame:
        """Ingest raw data from various sources."""
        logger.info(f"Ingesting data from {data_source}: {file_path}")
        
        if data_source == "csv":
            df = pd.read_csv(file_path)
        elif data_source == "json":
            df = pd.read_json(file_path)
        elif data_source == "ehr_api":
            # Placeholder for EHR API integration
            df = self._fetch_from_ehr_api(file_path)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def _fetch_from_ehr_api(self, endpoint: str) -> pd.DataFrame:
        """Fetch data from EHR API (placeholder)."""
        # This would integrate with actual EHR systems
        logger.info(f"Fetching from EHR API: {endpoint}")
        # Return empty DataFrame for now
        return pd.DataFrame()
    
    def process_vitals(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Process vital signs data."""
        logger.info("Processing vital signs data")
        
        # Standardize column names
        df = raw_df.copy()
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate data
        df = self.validator.validate_vitals(df)
        
        # Unit conversions
        if 'temperature_f' in df.columns:
            df['temperature'] = (df['temperature_f'] - 32) * 5/9
            df.drop('temperature_f', axis=1, inplace=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['patient_id', 'timestamp'])
        
        return df
    
    def process_labs(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Process laboratory results."""
        logger.info("Processing laboratory data")
        
        df = raw_df.copy()
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Standardize lab names
        df['lab_name'] = df['lab_name'].str.lower().str.strip()
        
        # Validate data
        df = self.validator.validate_labs(df)
        
        # Unit standardization (example)
        creatinine_mask = df['lab_name'].str.contains('creatinine', na=False)
        if creatinine_mask.any():
            # Convert μmol/L to mg/dL if needed
            umol_mask = df['units'].str.contains('μmol/L|umol/L', na=False)
            if umol_mask.any():
                df.loc[creatinine_mask & umol_mask, 'value'] *= 0.0113
                df.loc[creatinine_mask & umol_mask, 'units'] = 'mg/dL'
        
        return df
    
    def process_medications(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Process medication data."""
        logger.info("Processing medication data")
        
        df = raw_df.copy()
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Convert dates
        date_columns = ['start_date', 'end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Standardize medication names
        df['medication_name'] = df['medication_name'].str.lower().str.strip()
        
        # Calculate duration
        if 'start_date' in df.columns and 'end_date' in df.columns:
            df['duration_days'] = (df['end_date'] - df['start_date']).dt.days
        
        return df
    
    def create_deterioration_labels(self, 
                                  encounters_df: pd.DataFrame,
                                  prediction_date: datetime,
                                  lookback_days: int = 90) -> pd.DataFrame:
        """Create deterioration event labels."""
        logger.info("Creating deterioration labels")
        
        # Define prediction window
        window_start = prediction_date
        window_end = prediction_date + timedelta(days=lookback_days)
        
        # Filter encounters in prediction window
        mask = (encounters_df['start_datetime'] >= window_start) & \
               (encounters_df['start_datetime'] <= window_end)
        
        prediction_encounters = encounters_df[mask].copy()
        
        # Define deterioration events
        deterioration_types = [
            'hospitalization',
            'ed_visit'
        ]
        
        deterioration_mask = prediction_encounters['encounter_type'].isin(deterioration_types)
        deterioration_events = prediction_encounters[deterioration_mask]
        
        # Create labels DataFrame
        labels = []
        for patient_id in encounters_df['patient_id'].unique():
            patient_events = deterioration_events[
                deterioration_events['patient_id'] == patient_id
            ]
            
            has_deterioration = len(patient_events) > 0
            first_event_time = patient_events['start_datetime'].min() if has_deterioration else None
            
            labels.append({
                'patient_id': patient_id,
                'prediction_date': prediction_date,
                'deterioration_90d': has_deterioration,
                'first_event_timestamp': first_event_time,
                'event_count': len(patient_events)
            })
        
        return pd.DataFrame(labels)
    
    def save_to_database(self, df: pd.DataFrame, table_name: str):
        """Save processed data to database."""
        logger.info(f"Saving {len(df)} records to {table_name}")
        
        try:
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            logger.info(f"Successfully saved to {table_name}")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            raise
    
    def cache_features(self, patient_id: str, features: Dict):
        """Cache computed features in Redis."""
        cache_key = f"features:{patient_id}"
        self.redis_client.setex(
            cache_key, 
            timedelta(hours=24), 
            json.dumps(features, default=str)
        )


class FeatureEngineer:
    """Feature engineering for ML models."""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.aggregation_windows = clinical_config.AGGREGATION_WINDOWS
    
    def create_vital_features(self, vitals_df: pd.DataFrame, 
                            patient_id: str, 
                            reference_date: datetime) -> Dict:
        """Create vital sign features for a patient."""
        patient_vitals = vitals_df[vitals_df['patient_id'] == patient_id].copy()
        
        if len(patient_vitals) == 0:
            return {}
        
        features = {}
        
        # Sort by timestamp
        patient_vitals = patient_vitals.sort_values('timestamp')
        
        for window_days in self.aggregation_windows:
            window_start = reference_date - timedelta(days=window_days)
            window_data = patient_vitals[
                (patient_vitals['timestamp'] >= window_start) & 
                (patient_vitals['timestamp'] <= reference_date)
            ]
            
            if len(window_data) == 0:
                continue
            
            # Aggregations for each vital
            for vital in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'weight']:
                if vital in window_data.columns:
                    values = window_data[vital].dropna()
                    if len(values) > 0:
                        features[f'{vital}_mean_{window_days}d'] = values.mean()
                        features[f'{vital}_std_{window_days}d'] = values.std()
                        features[f'{vital}_min_{window_days}d'] = values.min()
                        features[f'{vital}_max_{window_days}d'] = values.max()
                        features[f'{vital}_last_{window_days}d'] = values.iloc[-1]
                        
                        # Trend (slope)
                        if len(values) >= 2:
                            x = np.arange(len(values))
                            slope = np.polyfit(x, values, 1)[0]
                            features[f'{vital}_trend_{window_days}d'] = slope
        
        # Weight change features
        if 'weight' in patient_vitals.columns:
            weight_data = patient_vitals[patient_vitals['weight'].notna()]
            if len(weight_data) >= 2:
                # 30-day weight change
                recent_30d = weight_data[
                    weight_data['timestamp'] >= reference_date - timedelta(days=30)
                ]
                if len(recent_30d) >= 2:
                    weight_change = recent_30d['weight'].iloc[-1] - recent_30d['weight'].iloc[0]
                    features['weight_change_30d'] = weight_change
        
        return features
    
    def create_lab_features(self, labs_df: pd.DataFrame, 
                          patient_id: str, 
                          reference_date: datetime) -> Dict:
        """Create laboratory features for a patient."""
        patient_labs = labs_df[labs_df['patient_id'] == patient_id].copy()
        
        if len(patient_labs) == 0:
            return {}
        
        features = {}
        
        # Key labs to focus on
        key_labs = ['creatinine', 'bnp', 'hba1c', 'hemoglobin']
        
        for lab in key_labs:
            lab_data = patient_labs[
                patient_labs['lab_name'].str.contains(lab, case=False, na=False)
            ].sort_values('timestamp')
            
            if len(lab_data) == 0:
                continue
            
            # Most recent value
            recent_data = lab_data[lab_data['timestamp'] <= reference_date]
            if len(recent_data) > 0:
                features[f'{lab}_last_value'] = recent_data['value'].iloc[-1]
                features[f'{lab}_days_since_last'] = (
                    reference_date - recent_data['timestamp'].iloc[-1]
                ).days
            
            # Change from baseline (90 days ago)
            baseline_date = reference_date - timedelta(days=90)
            baseline_data = lab_data[
                (lab_data['timestamp'] >= baseline_date - timedelta(days=30)) &
                (lab_data['timestamp'] <= baseline_date + timedelta(days=30))
            ]
            
            if len(baseline_data) > 0 and len(recent_data) > 0:
                baseline_value = baseline_data['value'].mean()
                recent_value = recent_data['value'].iloc[-1]
                features[f'{lab}_change_90d'] = recent_value - baseline_value
                features[f'{lab}_pct_change_90d'] = (
                    (recent_value - baseline_value) / baseline_value * 100
                )
        
        return features
    
    def create_medication_features(self, meds_df: pd.DataFrame,
                                 adherence_df: pd.DataFrame,
                                 patient_id: str,
                                 reference_date: datetime) -> Dict:
        """Create medication-related features."""
        patient_meds = meds_df[meds_df['patient_id'] == patient_id].copy()
        patient_adherence = adherence_df[adherence_df['patient_id'] == patient_id].copy()
        
        features = {}
        
        # Active medications count
        active_meds = patient_meds[
            (patient_meds['start_date'] <= reference_date) &
            ((patient_meds['end_date'].isna()) | (patient_meds['end_date'] >= reference_date))
        ]
        features['medication_count'] = len(active_meds)
        
        # Recent medication changes (30 days)
        recent_changes = patient_meds[
            patient_meds['start_date'] >= reference_date - timedelta(days=30)
        ]
        features['recent_med_changes'] = len(recent_changes)
        
        # Adherence features
        if len(patient_adherence) > 0:
            recent_adherence = patient_adherence[
                patient_adherence['measurement_period_end'] >= reference_date - timedelta(days=30)
            ]
            
            if len(recent_adherence) > 0:
                features['adherence_mean_30d'] = recent_adherence['adherence_percentage'].mean()
                features['adherence_min_30d'] = recent_adherence['adherence_percentage'].min()
                
                # Poor adherence flag
                poor_adherence_threshold = clinical_config.ADHERENCE_THRESHOLDS['poor'] * 100
                features['has_poor_adherence'] = (
                    recent_adherence['adherence_percentage'] < poor_adherence_threshold
                ).any()
        
        return features
    
    def create_utilization_features(self, encounters_df: pd.DataFrame,
                                  patient_id: str,
                                  reference_date: datetime) -> Dict:
        """Create healthcare utilization features."""
        patient_encounters = encounters_df[encounters_df['patient_id'] == patient_id].copy()
        
        features = {}
        
        # ED visits in various windows
        ed_encounters = patient_encounters[
            patient_encounters['encounter_type'] == 'ed_visit'
        ]
        
        for window_days in [30, 90, 365]:
            window_start = reference_date - timedelta(days=window_days)
            window_encounters = ed_encounters[
                (ed_encounters['start_datetime'] >= window_start) &
                (ed_encounters['start_datetime'] <= reference_date)
            ]
            features[f'ed_visits_{window_days}d'] = len(window_encounters)
        
        # Hospitalizations
        hosp_encounters = patient_encounters[
            patient_encounters['encounter_type'] == 'hospitalization'
        ]
        
        for window_days in [90, 365]:
            window_start = reference_date - timedelta(days=window_days)
            window_encounters = hosp_encounters[
                (hosp_encounters['start_datetime'] >= window_start) &
                (hosp_encounters['start_datetime'] <= reference_date)
            ]
            features[f'hospitalizations_{window_days}d'] = len(window_encounters)
        
        # Time since last hospitalization
        if len(hosp_encounters) > 0:
            last_hosp = hosp_encounters['start_datetime'].max()
            features['days_since_last_hospitalization'] = (reference_date - last_hosp).days
        
        return features
    
    def create_patient_features(self, patient_id: str, 
                              reference_date: datetime,
                              vitals_df: pd.DataFrame,
                              labs_df: pd.DataFrame,
                              meds_df: pd.DataFrame,
                              adherence_df: pd.DataFrame,
                              encounters_df: pd.DataFrame,
                              patients_df: pd.DataFrame) -> PatientFeatures:
        """Create comprehensive feature set for a patient."""
        logger.info(f"Creating features for patient {patient_id} at {reference_date}")
        
        # Get patient demographics
        patient_info = patients_df[patients_df['patient_id'] == patient_id].iloc[0]
        age_years = (reference_date.date() - patient_info['date_of_birth']).days / 365.25
        
        # Initialize features
        all_features = {
            'patient_id': patient_id,
            'feature_timestamp': reference_date,
            'age_years': age_years,
            'gender_encoded': 1 if patient_info['gender'] == 'M' else 0
        }
        
        # Add feature groups
        vital_features = self.create_vital_features(vitals_df, patient_id, reference_date)
        lab_features = self.create_lab_features(labs_df, patient_id, reference_date)
        med_features = self.create_medication_features(meds_df, adherence_df, patient_id, reference_date)
        util_features = self.create_utilization_features(encounters_df, patient_id, reference_date)
        
        all_features.update(vital_features)
        all_features.update(lab_features)
        all_features.update(med_features)
        all_features.update(util_features)
        
        # Calculate data completeness
        vitals_30d = vitals_df[
            (vitals_df['patient_id'] == patient_id) &
            (vitals_df['timestamp'] >= reference_date - timedelta(days=30)) &
            (vitals_df['timestamp'] <= reference_date)
        ]
        all_features['vitals_completeness_30d'] = min(len(vitals_30d) / 30.0, 1.0) * 100
        
        labs_90d = labs_df[
            (labs_df['patient_id'] == patient_id) &
            (labs_df['timestamp'] >= reference_date - timedelta(days=90)) &
            (labs_df['timestamp'] <= reference_date)
        ]
        all_features['labs_completeness_90d'] = min(len(labs_90d) / 10.0, 1.0) * 100
        
        return PatientFeatures(**all_features)


def main():
    """Main pipeline execution."""
    logger.info("Starting data pipeline")
    
    processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    
    # Example pipeline execution
    try:
        # This would be replaced with actual data sources
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
