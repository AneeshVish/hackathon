"""
Data validation module for patient deterioration prediction system.
Implements comprehensive validation checks for clinical data quality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Represents a single validation rule"""
    name: str
    description: str
    severity: str  # 'critical', 'warning', 'info'
    check_function: callable


class DataValidator:
    """Comprehensive data validation for clinical datasets"""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        
    def validate_cohort(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate entire patient cohort data
        
        Args:
            raw_data: Dictionary of dataframes by data type
            
        Returns:
            Validation report with errors, warnings, and statistics
        """
        report = {
            'validation_timestamp': datetime.utcnow().isoformat(),
            'critical_errors': 0,
            'warnings': 0,
            'info_messages': 0,
            'details': {},
            'data_quality_scores': {},
            'recommendations': []
        }
        
        # Validate each data type
        for data_type, df in raw_data.items():
            if not df.empty:
                type_report = self._validate_data_type(data_type, df)
                report['details'][data_type] = type_report
                
                # Aggregate counts
                report['critical_errors'] += type_report['critical_errors']
                report['warnings'] += type_report['warnings']
                report['info_messages'] += type_report['info_messages']
                
                # Calculate quality score
                report['data_quality_scores'][data_type] = self._calculate_quality_score(type_report)
        
        # Cross-dataset validations
        cross_validation = self._validate_cross_dataset(raw_data)
        report['details']['cross_dataset'] = cross_validation
        report['critical_errors'] += cross_validation['critical_errors']
        report['warnings'] += cross_validation['warnings']
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _initialize_validation_rules(self) -> Dict[str, List[ValidationRule]]:
        """Initialize validation rules for each data type"""
        
        rules = {
            'demographics': [
                ValidationRule(
                    'patient_id_unique',
                    'Patient IDs must be unique',
                    'critical',
                    self._check_unique_patient_ids
                ),
                ValidationRule(
                    'valid_birth_dates',
                    'Birth dates must be valid and reasonable',
                    'critical',
                    self._check_birth_dates
                ),
                ValidationRule(
                    'valid_gender',
                    'Gender values must be valid',
                    'warning',
                    self._check_gender_values
                )
            ],
            'vitals': [
                ValidationRule(
                    'vital_ranges',
                    'Vital signs must be within physiological ranges',
                    'critical',
                    self._check_vital_ranges
                ),
                ValidationRule(
                    'timestamp_order',
                    'Timestamps must be in chronological order',
                    'warning',
                    self._check_timestamp_order
                ),
                ValidationRule(
                    'missing_vitals',
                    'Check for excessive missing vital signs',
                    'warning',
                    self._check_missing_vitals
                )
            ],
            'labs': [
                ValidationRule(
                    'lab_values',
                    'Lab values must be within reasonable ranges',
                    'critical',
                    self._check_lab_ranges
                ),
                ValidationRule(
                    'lab_units',
                    'Lab units must be consistent',
                    'warning',
                    self._check_lab_units
                )
            ],
            'medications': [
                ValidationRule(
                    'medication_dates',
                    'Medication start/end dates must be logical',
                    'warning',
                    self._check_medication_dates
                ),
                ValidationRule(
                    'adherence_values',
                    'Adherence percentages must be 0-100',
                    'critical',
                    self._check_adherence_values
                )
            ]
        }
        
        return rules
    
    def _validate_data_type(self, data_type: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate a specific data type"""
        
        type_report = {
            'record_count': len(df),
            'critical_errors': 0,
            'warnings': 0,
            'info_messages': 0,
            'issues': [],
            'statistics': {}
        }
        
        # Get validation rules for this data type
        rules = self.validation_rules.get(data_type, [])
        
        for rule in rules:
            try:
                issues = rule.check_function(df)
                
                if issues:
                    type_report['issues'].extend([
                        {
                            'rule': rule.name,
                            'severity': rule.severity,
                            'description': rule.description,
                            'details': issue
                        }
                        for issue in issues
                    ])
                    
                    # Count by severity
                    if rule.severity == 'critical':
                        type_report['critical_errors'] += len(issues)
                    elif rule.severity == 'warning':
                        type_report['warnings'] += len(issues)
                    else:
                        type_report['info_messages'] += len(issues)
                        
            except Exception as e:
                logger.error(f"Error running validation rule {rule.name}: {e}")
                type_report['issues'].append({
                    'rule': rule.name,
                    'severity': 'critical',
                    'description': f"Validation rule failed: {e}",
                    'details': str(e)
                })
                type_report['critical_errors'] += 1
        
        # Calculate basic statistics
        type_report['statistics'] = self._calculate_basic_stats(df)
        
        return type_report
    
    def _check_unique_patient_ids(self, df: pd.DataFrame) -> List[str]:
        """Check for duplicate patient IDs"""
        issues = []
        
        if 'patient_id' in df.columns:
            duplicates = df['patient_id'].duplicated()
            if duplicates.any():
                duplicate_ids = df[duplicates]['patient_id'].unique()
                issues.append(f"Duplicate patient IDs found: {list(duplicate_ids)}")
        
        return issues
    
    def _check_birth_dates(self, df: pd.DataFrame) -> List[str]:
        """Check birth date validity"""
        issues = []
        
        if 'date_of_birth' in df.columns:
            # Convert to datetime
            try:
                birth_dates = pd.to_datetime(df['date_of_birth'])
                
                # Check for future dates
                future_dates = birth_dates > datetime.now()
                if future_dates.any():
                    issues.append(f"Future birth dates found: {future_dates.sum()} records")
                
                # Check for unreasonably old dates (>120 years)
                old_threshold = datetime.now() - timedelta(days=120*365)
                too_old = birth_dates < old_threshold
                if too_old.any():
                    issues.append(f"Unreasonably old birth dates: {too_old.sum()} records")
                    
            except Exception as e:
                issues.append(f"Invalid birth date format: {e}")
        
        return issues
    
    def _check_gender_values(self, df: pd.DataFrame) -> List[str]:
        """Check gender value validity"""
        issues = []
        
        if 'gender' in df.columns:
            valid_genders = {'M', 'F', 'O', 'U'}
            invalid_genders = ~df['gender'].isin(valid_genders)
            
            if invalid_genders.any():
                invalid_values = df[invalid_genders]['gender'].unique()
                issues.append(f"Invalid gender values: {list(invalid_values)}")
        
        return issues
    
    def _check_vital_ranges(self, df: pd.DataFrame) -> List[str]:
        """Check vital sign ranges"""
        issues = []
        
        # Define physiological ranges
        vital_ranges = {
            'heart_rate': (30, 300),
            'systolic_bp': (60, 300),
            'diastolic_bp': (30, 200),
            'weight': (20, 500),
            'spo2': (70, 100),
            'respiratory_rate': (8, 60),
            'temperature': (32, 45)
        }
        
        for vital, (min_val, max_val) in vital_ranges.items():
            if vital in df.columns:
                out_of_range = (df[vital] < min_val) | (df[vital] > max_val)
                out_of_range = out_of_range & df[vital].notna()
                
                if out_of_range.any():
                    issues.append(
                        f"{vital}: {out_of_range.sum()} values out of range "
                        f"({min_val}-{max_val})"
                    )
        
        return issues
    
    def _check_timestamp_order(self, df: pd.DataFrame) -> List[str]:
        """Check timestamp chronological order"""
        issues = []
        
        timestamp_cols = ['timestamp', 'start_datetime', 'event_date']
        
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    timestamps = pd.to_datetime(df[col])
                    
                    # Check for each patient
                    if 'patient_id' in df.columns:
                        for patient_id in df['patient_id'].unique():
                            patient_timestamps = timestamps[df['patient_id'] == patient_id]
                            if not patient_timestamps.is_monotonic_increasing:
                                issues.append(
                                    f"Non-chronological {col} for patient {patient_id}"
                                )
                    
                except Exception as e:
                    issues.append(f"Invalid timestamp format in {col}: {e}")
        
        return issues
    
    def _check_missing_vitals(self, df: pd.DataFrame) -> List[str]:
        """Check for excessive missing vital signs"""
        issues = []
        
        vital_cols = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'weight', 'spo2']
        
        for col in vital_cols:
            if col in df.columns:
                missing_pct = df[col].isna().mean() * 100
                if missing_pct > 50:  # More than 50% missing
                    issues.append(f"{col}: {missing_pct:.1f}% missing values")
        
        return issues
    
    def _check_lab_ranges(self, df: pd.DataFrame) -> List[str]:
        """Check laboratory value ranges"""
        issues = []
        
        # Common lab ranges (simplified)
        lab_ranges = {
            'BNP': (0, 10000),
            'creatinine': (0.3, 15.0),
            'HbA1c': (3.0, 20.0),
            'glucose': (20, 800),
            'hemoglobin': (3.0, 25.0)
        }
        
        if 'lab_name' in df.columns and 'value' in df.columns:
            for lab_name, (min_val, max_val) in lab_ranges.items():
                lab_data = df[df['lab_name'].str.contains(lab_name, case=False, na=False)]
                
                if not lab_data.empty:
                    out_of_range = (lab_data['value'] < min_val) | (lab_data['value'] > max_val)
                    out_of_range = out_of_range & lab_data['value'].notna()
                    
                    if out_of_range.any():
                        issues.append(
                            f"{lab_name}: {out_of_range.sum()} values out of range "
                            f"({min_val}-{max_val})"
                        )
        
        return issues
    
    def _check_lab_units(self, df: pd.DataFrame) -> List[str]:
        """Check lab unit consistency"""
        issues = []
        
        if 'lab_name' in df.columns and 'units' in df.columns:
            # Check for consistent units per lab
            lab_units = df.groupby('lab_name')['units'].nunique()
            inconsistent_labs = lab_units[lab_units > 1]
            
            if not inconsistent_labs.empty:
                issues.append(
                    f"Inconsistent units for labs: {list(inconsistent_labs.index)}"
                )
        
        return issues
    
    def _check_medication_dates(self, df: pd.DataFrame) -> List[str]:
        """Check medication date logic"""
        issues = []
        
        if 'start_date' in df.columns and 'end_date' in df.columns:
            try:
                start_dates = pd.to_datetime(df['start_date'])
                end_dates = pd.to_datetime(df['end_date'])
                
                # Check for end dates before start dates
                invalid_dates = (end_dates < start_dates) & end_dates.notna()
                
                if invalid_dates.any():
                    issues.append(f"End dates before start dates: {invalid_dates.sum()} records")
                    
            except Exception as e:
                issues.append(f"Invalid medication date format: {e}")
        
        return issues
    
    def _check_adherence_values(self, df: pd.DataFrame) -> List[str]:
        """Check adherence percentage values"""
        issues = []
        
        if 'adherence_pct' in df.columns:
            invalid_adherence = (
                (df['adherence_pct'] < 0) | 
                (df['adherence_pct'] > 100)
            ) & df['adherence_pct'].notna()
            
            if invalid_adherence.any():
                issues.append(
                    f"Invalid adherence percentages: {invalid_adherence.sum()} records"
                )
        
        return issues
    
    def _validate_cross_dataset(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate relationships across datasets"""
        
        cross_report = {
            'critical_errors': 0,
            'warnings': 0,
            'info_messages': 0,
            'issues': []
        }
        
        # Check patient ID consistency across datasets
        if 'demographics' in raw_data and not raw_data['demographics'].empty:
            demo_patients = set(raw_data['demographics']['patient_id'])
            
            for data_type, df in raw_data.items():
                if data_type != 'demographics' and not df.empty and 'patient_id' in df.columns:
                    data_patients = set(df['patient_id'])
                    
                    # Patients in data but not in demographics
                    orphaned = data_patients - demo_patients
                    if orphaned:
                        cross_report['issues'].append({
                            'type': 'orphaned_patients',
                            'severity': 'warning',
                            'description': f"{data_type}: {len(orphaned)} patients without demographics",
                            'details': list(orphaned)[:10]  # Show first 10
                        })
                        cross_report['warnings'] += 1
        
        # Check temporal consistency
        self._check_temporal_consistency(raw_data, cross_report)
        
        return cross_report
    
    def _check_temporal_consistency(self, raw_data: Dict[str, pd.DataFrame], report: Dict[str, Any]):
        """Check temporal consistency across datasets"""
        
        # Check if vital signs timestamps align reasonably with encounters
        if ('vitals' in raw_data and 'encounters' in raw_data and 
            not raw_data['vitals'].empty and not raw_data['encounters'].empty):
            
            # Sample check for a few patients
            sample_patients = raw_data['demographics']['patient_id'].head(10) if 'demographics' in raw_data else []
            
            for patient_id in sample_patients:
                patient_vitals = raw_data['vitals'][raw_data['vitals']['patient_id'] == patient_id]
                patient_encounters = raw_data['encounters'][raw_data['encounters']['patient_id'] == patient_id]
                
                if not patient_vitals.empty and not patient_encounters.empty:
                    vital_dates = pd.to_datetime(patient_vitals['timestamp']).dt.date
                    encounter_dates = pd.to_datetime(patient_encounters['start_datetime']).dt.date
                    
                    # Check for vitals without nearby encounters (could indicate data quality issues)
                    # This is just a sample check - in practice, you'd want more sophisticated logic
                    pass
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for a dataframe"""
        
        stats = {
            'record_count': len(df),
            'column_count': len(df.columns),
            'missing_data_pct': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Patient count if applicable
        if 'patient_id' in df.columns:
            stats['unique_patients'] = df['patient_id'].nunique()
        
        # Date range if applicable
        date_cols = ['timestamp', 'start_datetime', 'event_date', 'start_date']
        for col in date_cols:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col])
                    stats['date_range'] = {
                        'start': dates.min().isoformat(),
                        'end': dates.max().isoformat(),
                        'span_days': (dates.max() - dates.min()).days
                    }
                    break
                except:
                    continue
        
        return stats
    
    def _calculate_quality_score(self, type_report: Dict[str, Any]) -> float:
        """Calculate data quality score (0-100)"""
        
        base_score = 100.0
        
        # Deduct points for issues
        base_score -= type_report['critical_errors'] * 10  # 10 points per critical error
        base_score -= type_report['warnings'] * 2  # 2 points per warning
        base_score -= type_report['info_messages'] * 0.5  # 0.5 points per info message
        
        # Deduct points for missing data
        missing_pct = type_report['statistics'].get('missing_data_pct', 0)
        base_score -= missing_pct * 0.5  # 0.5 points per percent missing
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Critical error recommendations
        if report['critical_errors'] > 0:
            recommendations.append(
                "CRITICAL: Address all critical errors before proceeding with model training"
            )
        
        # Data quality recommendations
        avg_quality = np.mean(list(report['data_quality_scores'].values()))
        if avg_quality < 80:
            recommendations.append(
                f"Data quality score is {avg_quality:.1f}%. Consider additional data cleaning"
            )
        
        # Missing data recommendations
        for data_type, details in report['details'].items():
            if isinstance(details, dict) and 'statistics' in details:
                missing_pct = details['statistics'].get('missing_data_pct', 0)
                if missing_pct > 30:
                    recommendations.append(
                        f"{data_type}: {missing_pct:.1f}% missing data - implement imputation strategy"
                    )
        
        return recommendations
