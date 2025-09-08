"""
Data ingestion and ETL pipeline for patient deterioration prediction.
Handles validation, cleaning, normalization, and feature preparation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass

from .schemas import (
    PatientDemographics, VitalSigns, LabResult, Medication, 
    Encounter, LifestyleData, DeviceData, DeteriorationEvent,
    PatientFeatures
)
from .validation import DataValidator
from .preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for data pipeline"""
    lookback_days: int = 180
    prediction_horizon_days: int = 90
    min_data_completeness: float = 0.3
    vital_imputation_method: str = "forward_fill"
    lab_imputation_method: str = "mice"
    feature_aggregation_windows: List[int] = None
    
    def __post_init__(self):
        if self.feature_aggregation_windows is None:
            self.feature_aggregation_windows = [7, 14, 30, 90]


class DataPipeline:
    """Main data pipeline for processing patient data"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor(config)
        
    def process_patient_cohort(
        self, 
        raw_data_path: str,
        output_path: str,
        label_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process entire patient cohort through the pipeline
        
        Args:
            raw_data_path: Path to raw data files
            output_path: Path to save processed features
            label_definition: Definition of deterioration events
            
        Returns:
            Pipeline execution summary
        """
        logger.info("Starting patient cohort processing pipeline")
        
        # Step 1: Load and validate raw data
        raw_data = self._load_raw_data(raw_data_path)
        validation_report = self.validator.validate_cohort(raw_data)
        
        if validation_report['critical_errors'] > 0:
            raise ValueError(f"Critical validation errors: {validation_report}")
        
        # Step 2: Create labels
        labels_df = self._create_labels(raw_data, label_definition)
        
        # Step 3: Generate features for each patient
        features_list = []
        processed_patients = 0
        
        for patient_id in raw_data['demographics']['patient_id'].unique():
            try:
                patient_features = self._process_single_patient(
                    patient_id, raw_data, labels_df
                )
                if patient_features is not None:
                    features_list.append(patient_features)
                    processed_patients += 1
                    
                if processed_patients % 100 == 0:
                    logger.info(f"Processed {processed_patients} patients")
                    
            except Exception as e:
                logger.error(f"Error processing patient {patient_id}: {e}")
                continue
        
        # Step 4: Combine and save features
        if features_list:
            final_features = pd.concat(features_list, ignore_index=True)
            final_features.to_parquet(f"{output_path}/patient_features.parquet")
            
            # Save metadata
            metadata = {
                'total_patients': len(raw_data['demographics']),
                'processed_patients': processed_patients,
                'feature_columns': list(final_features.columns),
                'pipeline_config': self.config.__dict__,
                'validation_report': validation_report,
                'created_at': datetime.utcnow().isoformat()
            }
            
            with open(f"{output_path}/pipeline_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Pipeline completed. Processed {processed_patients} patients")
            return metadata
        else:
            raise ValueError("No patients successfully processed")
    
    def _load_raw_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load raw data from various sources"""
        data_path = Path(data_path)
        raw_data = {}
        
        # Expected data files
        data_files = {
            'demographics': 'demographics.csv',
            'vitals': 'vitals.csv',
            'labs': 'labs.csv',
            'medications': 'medications.csv',
            'encounters': 'encounters.csv',
            'lifestyle': 'lifestyle.csv',
            'devices': 'devices.csv',
            'events': 'deterioration_events.csv'
        }
        
        for data_type, filename in data_files.items():
            file_path = data_path / filename
            if file_path.exists():
                raw_data[data_type] = pd.read_csv(file_path)
                logger.info(f"Loaded {len(raw_data[data_type])} records from {filename}")
            else:
                logger.warning(f"File not found: {filename}")
                raw_data[data_type] = pd.DataFrame()
        
        return raw_data
    
    def _create_labels(
        self, 
        raw_data: Dict[str, pd.DataFrame], 
        label_definition: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create binary labels based on deterioration events"""
        
        if raw_data['events'].empty:
            logger.warning("No deterioration events found")
            return pd.DataFrame()
        
        events_df = raw_data['events'].copy()
        events_df['event_date'] = pd.to_datetime(events_df['event_date'])
        
        # Filter events based on definition
        valid_event_types = label_definition.get('event_types', [
            'hospitalization', 'ed_visit', 'mortality', 'clinical_event'
        ])
        
        events_df = events_df[events_df['event_type'].isin(valid_event_types)]
        
        # Create labels with prediction windows
        labels = []
        for _, event in events_df.iterrows():
            # Create label for prediction window before event
            label_date = event['event_date'] - timedelta(days=self.config.prediction_horizon_days)
            
            labels.append({
                'patient_id': event['patient_id'],
                'label_date': label_date,
                'label': 1,
                'event_date': event['event_date'],
                'event_type': event['event_type']
            })
        
        return pd.DataFrame(labels)
    
    def _process_single_patient(
        self, 
        patient_id: str, 
        raw_data: Dict[str, pd.DataFrame],
        labels_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Process a single patient's data"""
        
        # Get patient demographics
        demo = raw_data['demographics'][
            raw_data['demographics']['patient_id'] == patient_id
        ]
        if demo.empty:
            return None
        
        # Get patient's time series data
        patient_data = self._extract_patient_timeseries(patient_id, raw_data)
        
        if not self._check_data_completeness(patient_data):
            return None
        
        # Generate features at multiple time points
        feature_rows = []
        
        # Get label dates for this patient
        patient_labels = labels_df[labels_df['patient_id'] == patient_id]
        
        # Generate features for label dates
        for _, label_row in patient_labels.iterrows():
            features = self._generate_features_at_timepoint(
                patient_id, patient_data, label_row['label_date'], demo.iloc[0]
            )
            if features is not None:
                features['label'] = label_row['label']
                features['label_date'] = label_row['label_date']
                feature_rows.append(features)
        
        # Also generate features for negative samples (no events)
        # Sample random dates for negative examples
        if not patient_data['vitals'].empty:
            date_range = pd.date_range(
                start=patient_data['vitals']['timestamp'].min() + timedelta(days=self.config.lookback_days),
                end=patient_data['vitals']['timestamp'].max() - timedelta(days=self.config.prediction_horizon_days),
                freq='30D'  # Monthly samples
            )
            
            for sample_date in date_range:
                # Check if this date is too close to any positive event
                if not self._is_near_positive_event(sample_date, patient_labels):
                    features = self._generate_features_at_timepoint(
                        patient_id, patient_data, sample_date, demo.iloc[0]
                    )
                    if features is not None:
                        features['label'] = 0
                        features['label_date'] = sample_date
                        feature_rows.append(features)
        
        if feature_rows:
            return pd.DataFrame(feature_rows)
        return None
    
    def _extract_patient_timeseries(
        self, 
        patient_id: str, 
        raw_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Extract time series data for a single patient"""
        
        patient_data = {}
        
        for data_type in ['vitals', 'labs', 'medications', 'encounters', 'lifestyle', 'devices']:
            if data_type in raw_data and not raw_data[data_type].empty:
                df = raw_data[data_type][raw_data[data_type]['patient_id'] == patient_id].copy()
                
                # Convert timestamp columns
                timestamp_cols = ['timestamp', 'start_datetime', 'start_date']
                for col in timestamp_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.sort_values(col)
                        break
                
                patient_data[data_type] = df
            else:
                patient_data[data_type] = pd.DataFrame()
        
        return patient_data
    
    def _check_data_completeness(self, patient_data: Dict[str, pd.DataFrame]) -> bool:
        """Check if patient has sufficient data for modeling"""
        
        # Check vital signs completeness
        if patient_data['vitals'].empty:
            return False
        
        # Check time span
        vitals_span = (
            patient_data['vitals']['timestamp'].max() - 
            patient_data['vitals']['timestamp'].min()
        ).days
        
        if vitals_span < self.config.lookback_days:
            return False
        
        # Check data density
        expected_records = vitals_span  # Assuming daily vitals
        actual_records = len(patient_data['vitals'])
        completeness = actual_records / expected_records
        
        return completeness >= self.config.min_data_completeness
    
    def _generate_features_at_timepoint(
        self, 
        patient_id: str, 
        patient_data: Dict[str, pd.DataFrame],
        feature_date: datetime,
        demographics: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """Generate features for a specific timepoint"""
        
        # Filter data to lookback window
        lookback_start = feature_date - timedelta(days=self.config.lookback_days)
        
        windowed_data = {}
        for data_type, df in patient_data.items():
            if not df.empty:
                timestamp_col = self._get_timestamp_column(df)
                if timestamp_col:
                    mask = (df[timestamp_col] >= lookback_start) & (df[timestamp_col] <= feature_date)
                    windowed_data[data_type] = df[mask]
                else:
                    windowed_data[data_type] = df
            else:
                windowed_data[data_type] = df
        
        # Generate features using preprocessor
        features = self.preprocessor.generate_features(
            patient_id, windowed_data, feature_date, demographics
        )
        
        return features
    
    def _get_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Get the timestamp column name for a dataframe"""
        timestamp_cols = ['timestamp', 'start_datetime', 'start_date', 'event_date']
        for col in timestamp_cols:
            if col in df.columns:
                return col
        return None
    
    def _is_near_positive_event(
        self, 
        sample_date: datetime, 
        patient_labels: pd.DataFrame,
        buffer_days: int = 30
    ) -> bool:
        """Check if sample date is too close to a positive event"""
        
        if patient_labels.empty:
            return False
        
        for _, label in patient_labels.iterrows():
            days_diff = abs((sample_date - label['label_date']).days)
            if days_diff < buffer_days:
                return True
        
        return False


def main():
    """Main pipeline execution"""
    
    # Configuration
    config = PipelineConfig(
        lookback_days=180,
        prediction_horizon_days=90,
        min_data_completeness=0.3
    )
    
    # Label definition
    label_definition = {
        'event_types': ['hospitalization', 'ed_visit', 'mortality'],
        'prediction_horizon_days': 90
    }
    
    # Initialize and run pipeline
    pipeline = DataPipeline(config)
    
    try:
        results = pipeline.process_patient_cohort(
            raw_data_path="./data/raw",
            output_path="./data/processed",
            label_definition=label_definition
        )
        
        print("Pipeline completed successfully!")
        print(f"Processed {results['processed_patients']} patients")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
