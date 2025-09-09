"""
Data preprocessing module for patient deterioration prediction.
Implements feature engineering, imputation, and clinical transformations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Comprehensive data preprocessing for clinical ML"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        
    def generate_features(
        self, 
        patient_id: str,
        patient_data: Dict[str, pd.DataFrame],
        feature_date: datetime,
        demographics: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """
        Generate comprehensive feature set for a patient at a specific timepoint
        
        Args:
            patient_id: Patient identifier
            patient_data: Patient's time series data
            feature_date: Date to generate features for
            demographics: Patient demographics
            
        Returns:
            Dictionary of features or None if insufficient data
        """
        
        features = {
            'patient_id': patient_id,
            'feature_timestamp': feature_date
        }
        
        try:
            # Demographic features
            demo_features = self._generate_demographic_features(demographics, feature_date)
            features.update(demo_features)
            
            # Vital sign features
            vital_features = self._generate_vital_features(patient_data['vitals'], feature_date)
            features.update(vital_features)
            
            # Laboratory features
            lab_features = self._generate_lab_features(patient_data['labs'], feature_date)
            features.update(lab_features)
            
            # Medication features
            med_features = self._generate_medication_features(patient_data['medications'], feature_date)
            features.update(med_features)
            
            # Utilization features
            util_features = self._generate_utilization_features(patient_data['encounters'], feature_date)
            features.update(util_features)
            
            # Lifestyle features
            lifestyle_features = self._generate_lifestyle_features(patient_data['lifestyle'], feature_date)
            features.update(lifestyle_features)
            
            # Temporal features
            temporal_features = self._generate_temporal_features(feature_date)
            features.update(temporal_features)
            
            # Composite risk scores
            composite_features = self._generate_composite_features(features, patient_data)
            features.update(composite_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating features for patient {patient_id}: {e}")
            return None
    
    def _generate_demographic_features(self, demographics: pd.Series, feature_date: datetime) -> Dict[str, Any]:
        """Generate demographic-based features"""
        
        features = {}
        
        # Age calculation
        if 'date_of_birth' in demographics and pd.notna(demographics['date_of_birth']):
            try:
                birth_date = pd.to_datetime(demographics['date_of_birth'])
                age = (feature_date - birth_date).days / 365.25
                features['age'] = age
                features['age_group'] = self._categorize_age(age)
            except:
                features['age'] = 65.0  # Default age
                features['age_group'] = 'elderly'
        else:
            features['age'] = 65.0  # Default age
            features['age_group'] = 'elderly'
        
        # Gender encoding
        if 'gender' in demographics and pd.notna(demographics['gender']):
            gender_map = {'M': 1, 'F': 0, 'O': 2, 'U': -1}
            features['gender_encoded'] = gender_map.get(demographics['gender'], -1)
        else:
            features['gender_encoded'] = -1
        
        # Comorbidity features
        if 'comorbidities' in demographics and demographics['comorbidities']:
            comorbidities = demographics['comorbidities']
            if isinstance(comorbidities, str):
                comorbidities = comorbidities.split(',')
            
            features['comorbidity_count'] = len(comorbidities)
            features['charlson_score'] = self._calculate_charlson_score(comorbidities)
            
            # Specific comorbidity flags
            features['has_diabetes'] = any('E11' in code or 'E10' in code for code in comorbidities)
            features['has_hypertension'] = any('I10' in code for code in comorbidities)
            features['has_heart_failure'] = any('I50' in code for code in comorbidities)
            features['has_copd'] = any('J44' in code for code in comorbidities)
        else:
            features['comorbidity_count'] = 0
            features['charlson_score'] = 0
            features['has_diabetes'] = False
            features['has_hypertension'] = False
            features['has_heart_failure'] = False
            features['has_copd'] = False
        
        return features
    
    def _generate_vital_features(self, vitals_df: pd.DataFrame, feature_date: datetime) -> Dict[str, Any]:
        """Generate vital sign-based features"""
        
        features = {}
        
        if vitals_df.empty:
            return self._get_default_vital_features()
        
        # Define aggregation windows
        windows = [7, 14, 30, 90]
        vital_columns = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'weight', 'spo2', 'respiratory_rate', 'temperature']
        
        for window in windows:
            window_start = feature_date - timedelta(days=window)
            window_data = vitals_df[
                (vitals_df['timestamp'] >= window_start) & 
                (vitals_df['timestamp'] <= feature_date)
            ]
            
            if not window_data.empty:
                for vital in vital_columns:
                    if vital in window_data.columns:
                        values = window_data[vital].dropna()
                        
                        if len(values) > 0:
                            prefix = f"{vital}_{window}d"
                            
                            # Basic statistics
                            features[f"{prefix}_mean"] = values.mean()
                            features[f"{prefix}_std"] = values.std()
                            features[f"{prefix}_min"] = values.min()
                            features[f"{prefix}_max"] = values.max()
                            features[f"{prefix}_last"] = values.iloc[-1]
                            features[f"{prefix}_count"] = len(values)
                            
                            # Trend analysis
                            if len(values) >= 3:
                                features[f"{prefix}_trend"] = self._calculate_trend(values)
                                features[f"{prefix}_variability"] = values.std() / values.mean() if values.mean() != 0 else 0
                            
                            # Change from baseline (90-day window as baseline)
                            if window < 90:
                                baseline_start = feature_date - timedelta(days=90)
                                baseline_end = feature_date - timedelta(days=60)
                                baseline_data = vitals_df[
                                    (vitals_df['timestamp'] >= baseline_start) & 
                                    (vitals_df['timestamp'] <= baseline_end)
                                ]
                                
                                if not baseline_data.empty and vital in baseline_data.columns:
                                    baseline_mean = baseline_data[vital].mean()
                                    if not pd.isna(baseline_mean) and baseline_mean != 0:
                                        current_mean = values.mean()
                                        features[f"{prefix}_change_pct"] = ((current_mean - baseline_mean) / baseline_mean) * 100
        
        # Special vital sign features
        features.update(self._generate_special_vital_features(vitals_df, feature_date))
        
        return features
    
    def _generate_lab_features(self, labs_df: pd.DataFrame, feature_date: datetime) -> Dict[str, Any]:
        """Generate laboratory-based features"""
        
        features = {}
        
        if labs_df.empty:
            return self._get_default_lab_features()
        
        # Key lab values to focus on
        key_labs = ['BNP', 'creatinine', 'HbA1c', 'glucose', 'hemoglobin', 'sodium', 'potassium']
        
        # Get labs within lookback period
        lookback_start = feature_date - timedelta(days=self.config.lookback_days)
        recent_labs = labs_df[
            (labs_df['timestamp'] >= lookback_start) & 
            (labs_df['timestamp'] <= feature_date)
        ]
        
        for lab_name in key_labs:
            lab_data = recent_labs[
                recent_labs['lab_name'].str.contains(lab_name, case=False, na=False)
            ]
            
            if not lab_data.empty:
                # Sort by timestamp
                lab_data = lab_data.sort_values('timestamp')
                values = lab_data['value']
                
                prefix = f"{lab_name.lower()}"
                
                # Most recent value
                features[f"{prefix}_last"] = values.iloc[-1]
                features[f"{prefix}_count"] = len(values)
                
                # Time since last measurement
                last_timestamp = lab_data['timestamp'].iloc[-1]
                features[f"{prefix}_days_since"] = (feature_date - last_timestamp).days
                
                # Trend if multiple values
                if len(values) >= 2:
                    features[f"{prefix}_trend"] = self._calculate_trend(values)
                    
                    # Change from first to last
                    first_value = values.iloc[0]
                    last_value = values.iloc[-1]
                    if first_value != 0:
                        features[f"{prefix}_change_pct"] = ((last_value - first_value) / first_value) * 100
                
                # Abnormal flags
                if 'abnormal_flag' in lab_data.columns:
                    abnormal_count = (lab_data['abnormal_flag'].isin(['H', 'L'])).sum()
                    features[f"{prefix}_abnormal_count"] = abnormal_count
            else:
                # Default values for missing labs
                features[f"{prefix}_last"] = np.nan
                features[f"{prefix}_count"] = 0
                features[f"{prefix}_days_since"] = 999
                features[f"{prefix}_trend"] = 0
                features[f"{prefix}_change_pct"] = 0
                features[f"{prefix}_abnormal_count"] = 0
        
        return features
    
    def _generate_medication_features(self, meds_df: pd.DataFrame, feature_date: datetime) -> Dict[str, Any]:
        """Generate medication-based features"""
        
        features = {}
        
        if meds_df.empty:
            return self._get_default_medication_features()
        
        # Active medications at feature date
        active_meds = meds_df[
            (meds_df['start_date'] <= feature_date) &
            ((meds_df['end_date'].isna()) | (meds_df['end_date'] >= feature_date))
        ]
        
        features['medication_count'] = len(active_meds)
        features['polypharmacy'] = 1 if len(active_meds) >= 5 else 0
        
        # Adherence features
        if 'adherence_pct' in meds_df.columns:
            recent_adherence = active_meds['adherence_pct'].dropna()
            if len(recent_adherence) > 0:
                features['adherence_mean'] = recent_adherence.mean()
                features['adherence_min'] = recent_adherence.min()
                features['poor_adherence_count'] = (recent_adherence < 80).sum()
            else:
                features['adherence_mean'] = np.nan
                features['adherence_min'] = np.nan
                features['poor_adherence_count'] = 0
        
        # Recent medication changes
        lookback_30d = feature_date - timedelta(days=30)
        recent_starts = meds_df[meds_df['start_date'] >= lookback_30d]
        recent_stops = meds_df[
            (meds_df['end_date'] >= lookback_30d) & 
            (meds_df['end_date'].notna())
        ]
        
        features['recent_med_starts'] = len(recent_starts)
        features['recent_med_stops'] = len(recent_stops)
        features['recent_med_changes'] = len(recent_starts) + len(recent_stops)
        
        # Medication class features
        features.update(self._generate_medication_class_features(active_meds))
        
        return features
    
    def _generate_utilization_features(self, encounters_df: pd.DataFrame, feature_date: datetime) -> Dict[str, Any]:
        """Generate healthcare utilization features"""
        
        features = {}
        
        if encounters_df.empty:
            return self._get_default_utilization_features()
        
        # Define time windows
        windows = [30, 90, 365]
        
        for window in windows:
            window_start = feature_date - timedelta(days=window)
            window_encounters = encounters_df[
                (encounters_df['start_datetime'] >= window_start) &
                (encounters_df['start_datetime'] <= feature_date)
            ]
            
            prefix = f"{window}d"
            
            # Total encounters
            features[f"encounters_{prefix}"] = len(window_encounters)
            
            # By encounter type
            for enc_type in ['emergency', 'inpatient', 'outpatient']:
                type_encounters = window_encounters[
                    window_encounters['encounter_type'].str.contains(enc_type, case=False, na=False)
                ]
                features[f"{enc_type}_visits_{prefix}"] = len(type_encounters)
        
        # Time since last visit
        if not encounters_df.empty:
            last_visit = encounters_df['start_datetime'].max()
            features['days_since_last_visit'] = (feature_date - last_visit).days
        else:
            features['days_since_last_visit'] = 999
        
        # Readmission patterns
        inpatient_encounters = encounters_df[
            encounters_df['encounter_type'].str.contains('inpatient', case=False, na=False)
        ].sort_values('start_datetime')
        
        if len(inpatient_encounters) >= 2:
            # 30-day readmissions
            readmissions_30d = 0
            for i in range(1, len(inpatient_encounters)):
                days_between = (inpatient_encounters.iloc[i]['start_datetime'] - 
                              inpatient_encounters.iloc[i-1]['start_datetime']).days
                if days_between <= 30:
                    readmissions_30d += 1
            
            features['readmissions_30d'] = readmissions_30d
        else:
            features['readmissions_30d'] = 0
        
        return features
    
    def _generate_lifestyle_features(self, lifestyle_df: pd.DataFrame, feature_date: datetime) -> Dict[str, Any]:
        """Generate lifestyle and patient-reported outcome features"""
        
        features = {}
        
        if lifestyle_df.empty:
            return self._get_default_lifestyle_features()
        
        # Recent lifestyle data (last 30 days)
        lookback_30d = feature_date - timedelta(days=30)
        recent_lifestyle = lifestyle_df[
            (lifestyle_df['timestamp'] >= lookback_30d) &
            (lifestyle_df['timestamp'] <= feature_date)
        ]
        
        if not recent_lifestyle.empty:
            lifestyle_metrics = ['activity_minutes', 'steps', 'sleep_hours', 'sleep_quality', 'diet_quality', 'stress_level']
            
            for metric in lifestyle_metrics:
                if metric in recent_lifestyle.columns:
                    values = recent_lifestyle[metric].dropna()
                    if len(values) > 0:
                        features[f"{metric}_mean_30d"] = values.mean()
                        features[f"{metric}_std_30d"] = values.std()
                        features[f"{metric}_trend_30d"] = self._calculate_trend(values)
                    else:
                        features[f"{metric}_mean_30d"] = np.nan
                        features[f"{metric}_std_30d"] = np.nan
                        features[f"{metric}_trend_30d"] = 0
            
            # Smoking status (most recent)
            if 'smoking_status' in recent_lifestyle.columns:
                smoking_status = recent_lifestyle['smoking_status'].iloc[-1]
                smoking_map = {'never': 0, 'former': 1, 'current': 2}
                features['smoking_encoded'] = smoking_map.get(smoking_status, -1)
        
        return features
    
    def _generate_temporal_features(self, feature_date: datetime) -> Dict[str, Any]:
        """Generate temporal features"""
        
        features = {}
        
        # Date components
        features['month'] = feature_date.month
        features['day_of_week'] = feature_date.weekday()
        features['quarter'] = (feature_date.month - 1) // 3 + 1
        
        # Season
        season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                     3: 'spring', 4: 'spring', 5: 'spring',
                     6: 'summer', 7: 'summer', 8: 'summer',
                     9: 'fall', 10: 'fall', 11: 'fall'}
        features['season'] = season_map[feature_date.month]
        
        # Holiday proximity (simplified)
        features['is_holiday_season'] = 1 if feature_date.month in [11, 12, 1] else 0
        
        return features
    
    def _generate_composite_features(self, features: Dict[str, Any], patient_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate composite risk scores and derived features"""
        
        composite = {}
        
        # Frailty indicators
        frailty_score = 0
        if features.get('age', 0) > 75:
            frailty_score += 1
        if features.get('medication_count', 0) >= 5:
            frailty_score += 1
        if features.get('weight_30d_change_pct', 0) < -5:  # Weight loss
            frailty_score += 1
        if features.get('activity_minutes_mean_30d', 0) < 30:  # Low activity
            frailty_score += 1
        
        composite['frailty_score'] = frailty_score
        
        # Clinical instability score
        instability_score = 0
        if features.get('heart_rate_7d_std', 0) > 15:  # High HR variability
            instability_score += 1
        if features.get('systolic_bp_7d_std', 0) > 20:  # High BP variability
            instability_score += 1
        if features.get('recent_med_changes', 0) > 2:  # Multiple med changes
            instability_score += 1
        if features.get('emergency_visits_30d', 0) > 0:  # Recent ED visits
            instability_score += 1
        
        composite['instability_score'] = instability_score
        
        # Heart failure specific features (if applicable)
        if features.get('has_heart_failure', False):
            hf_score = 0
            if features.get('bnp_last', 0) > 400:  # Elevated BNP
                hf_score += 2
            if features.get('weight_7d_change_pct', 0) > 3:  # Rapid weight gain
                hf_score += 2
            if features.get('adherence_mean', 100) < 80:  # Poor adherence
                hf_score += 1
            
            composite['hf_risk_score'] = hf_score
        
        return composite
    
    def _calculate_trend(self, values: pd.Series) -> float:
        """Calculate trend (slope) of values over time"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        try:
            slope, _, _, _, _ = stats.linregress(x, values)
            return slope
        except:
            return 0.0
    
    def _categorize_age(self, age: float) -> str:
        """Categorize age into groups"""
        if age < 18:
            return 'pediatric'
        elif age < 65:
            return 'adult'
        elif age < 80:
            return 'elderly'
        else:
            return 'very_elderly'
    
    def _calculate_charlson_score(self, comorbidities: List[str]) -> float:
        """Calculate Charlson Comorbidity Index (simplified)"""
        
        charlson_weights = {
            'I21': 1,  # MI
            'I50': 1,  # CHF
            'I63': 1,  # Stroke
            'J44': 1,  # COPD
            'E10': 1,  # Diabetes
            'E11': 1,  # Diabetes
            'N18': 2,  # CKD
            'C78': 6,  # Metastatic cancer
        }
        
        score = 0
        for code in comorbidities:
            for icd_code, weight in charlson_weights.items():
                if icd_code in code:
                    score += weight
                    break
        
        return score
    
    def _generate_special_vital_features(self, vitals_df: pd.DataFrame, feature_date: datetime) -> Dict[str, Any]:
        """Generate special vital sign features"""
        
        features = {}
        
        # Pulse pressure (if both systolic and diastolic BP available)
        recent_vitals = vitals_df[vitals_df['timestamp'] <= feature_date].tail(10)
        
        if not recent_vitals.empty:
            if 'systolic_bp' in recent_vitals.columns and 'diastolic_bp' in recent_vitals.columns:
                systolic = recent_vitals['systolic_bp'].dropna()
                diastolic = recent_vitals['diastolic_bp'].dropna()
                
                if len(systolic) > 0 and len(diastolic) > 0:
                    # Use most recent values where both are available
                    recent_complete = recent_vitals.dropna(subset=['systolic_bp', 'diastolic_bp'])
                    if not recent_complete.empty:
                        last_systolic = recent_complete['systolic_bp'].iloc[-1]
                        last_diastolic = recent_complete['diastolic_bp'].iloc[-1]
                        features['pulse_pressure'] = last_systolic - last_diastolic
        
        return features
    
    def _generate_medication_class_features(self, active_meds: pd.DataFrame) -> Dict[str, Any]:
        """Generate medication class-specific features"""
        
        features = {}
        
        # Common medication classes (simplified mapping)
        med_classes = {
            'ace_inhibitor': ['lisinopril', 'enalapril', 'captopril'],
            'beta_blocker': ['metoprolol', 'atenolol', 'propranolol'],
            'diuretic': ['furosemide', 'hydrochlorothiazide', 'spironolactone'],
            'statin': ['atorvastatin', 'simvastatin', 'rosuvastatin'],
            'diabetes_med': ['metformin', 'insulin', 'glipizide']
        }
        
        for class_name, med_names in med_classes.items():
            count = 0
            for med_name in med_names:
                count += active_meds['medication_name'].str.contains(med_name, case=False, na=False).sum()
            features[f"{class_name}_count"] = count
            features[f"on_{class_name}"] = 1 if count > 0 else 0
        
        return features
    
    # Default feature methods for missing data
    def _get_default_vital_features(self) -> Dict[str, Any]:
        """Return default vital features when no data available"""
        features = {}
        windows = [7, 14, 30, 90]
        vitals = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'weight', 'spo2', 'respiratory_rate', 'temperature']
        
        for window in windows:
            for vital in vitals:
                prefix = f"{vital}_{window}d"
                features[f"{prefix}_mean"] = np.nan
                features[f"{prefix}_std"] = np.nan
                features[f"{prefix}_trend"] = 0
                features[f"{prefix}_count"] = 0
        
        return features
    
    def _get_default_lab_features(self) -> Dict[str, Any]:
        """Return default lab features when no data available"""
        features = {}
        labs = ['bnp', 'creatinine', 'hba1c', 'glucose', 'hemoglobin']
        
        for lab in labs:
            features[f"{lab}_last"] = np.nan
            features[f"{lab}_count"] = 0
            features[f"{lab}_days_since"] = 999
            features[f"{lab}_trend"] = 0
            features[f"{lab}_change_pct"] = 0
        
        return features
    
    def _get_default_medication_features(self) -> Dict[str, Any]:
        """Return default medication features when no data available"""
        return {
            'medication_count': 0,
            'polypharmacy': 0,
            'adherence_mean': np.nan,
            'recent_med_changes': 0,
            'on_ace_inhibitor': 0,
            'on_beta_blocker': 0,
            'on_diuretic': 0
        }
    
    def _get_default_utilization_features(self) -> Dict[str, Any]:
        """Return default utilization features when no data available"""
        features = {}
        windows = [30, 90, 365]
        
        for window in windows:
            features[f"encounters_{window}d"] = 0
            features[f"emergency_visits_{window}d"] = 0
            features[f"inpatient_visits_{window}d"] = 0
            features[f"outpatient_visits_{window}d"] = 0
        
        features['days_since_last_visit'] = 999
        features['readmissions_30d'] = 0
        
        return features
    
    def _get_default_lifestyle_features(self) -> Dict[str, Any]:
        """Return default lifestyle features when no data available"""
        return {
            'activity_minutes_mean_30d': np.nan,
            'sleep_quality_mean_30d': np.nan,
            'smoking_encoded': -1
        }
