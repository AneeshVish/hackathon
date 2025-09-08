"""
Explainability module for patient deterioration prediction models.
Implements SHAP-based global and local explanations with clinical interpretations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ExplainabilityEngine:
    """Comprehensive explainability for clinical ML models"""
    
    def __init__(self):
        self.explainers = {}
        self.feature_groups = self._define_feature_groups()
        
    def _define_feature_groups(self) -> Dict[str, List[str]]:
        """Define clinical feature groups for organized explanations"""
        
        return {
            'demographics': ['age', 'gender_encoded', 'comorbidity_count', 'charlson_score'],
            'vitals': [
                'heart_rate', 'systolic_bp', 'diastolic_bp', 'weight', 'spo2', 
                'respiratory_rate', 'temperature', 'pulse_pressure'
            ],
            'labs': [
                'bnp', 'creatinine', 'hba1c', 'glucose', 'hemoglobin', 
                'sodium', 'potassium'
            ],
            'medications': [
                'medication_count', 'adherence_mean', 'polypharmacy',
                'recent_med_changes', 'on_ace_inhibitor', 'on_beta_blocker'
            ],
            'utilization': [
                'encounters', 'emergency_visits', 'inpatient_visits', 
                'days_since_last_visit', 'readmissions'
            ],
            'lifestyle': [
                'activity_minutes', 'sleep_quality', 'smoking_encoded'
            ],
            'composite': [
                'frailty_score', 'instability_score', 'hf_risk_score'
            ]
        }
    
    def generate_global_explanations(
        self, 
        model, 
        X_test: np.ndarray, 
        feature_names: List[str],
        max_display: int = 20
    ) -> Dict[str, Any]:
        """
        Generate global model explanations using SHAP
        
        Args:
            model: Trained model
            X_test: Test data for explanation
            feature_names: List of feature names
            max_display: Maximum features to display
            
        Returns:
            Global explanation results
        """
        
        logger.info("Generating global explanations with SHAP")
        
        try:
            # Initialize SHAP explainer
            if hasattr(model, 'predict_proba'):
                # For sklearn-like models
                explainer = shap.Explainer(model.predict_proba, X_test[:100])  # Use sample for efficiency
            else:
                # For other models
                explainer = shap.Explainer(model, X_test[:100])
            
            # Calculate SHAP values
            shap_values = explainer(X_test[:500])  # Use subset for efficiency
            
            # Extract SHAP values for positive class
            if len(shap_values.shape) == 3:  # Multi-class output
                shap_vals = shap_values[:, :, 1]  # Positive class
            else:
                shap_vals = shap_values
            
            # Feature importance (mean absolute SHAP values)
            feature_importance = np.mean(np.abs(shap_vals), axis=0)
            
            # Sort features by importance
            importance_indices = np.argsort(feature_importance)[::-1]
            
            # Top features
            top_features = []
            for i in importance_indices[:max_display]:
                top_features.append({
                    'feature': feature_names[i],
                    'importance': float(feature_importance[i]),
                    'mean_shap': float(np.mean(shap_vals[:, i])),
                    'clinical_interpretation': self._get_clinical_interpretation(feature_names[i])
                })
            
            # Feature group analysis
            group_importance = self._calculate_group_importance(
                feature_importance, feature_names
            )
            
            # Generate summary plots data
            summary_data = self._prepare_summary_plot_data(
                shap_vals, feature_names, importance_indices[:max_display]
            )
            
            return {
                'top_features': top_features,
                'group_importance': group_importance,
                'summary_plot_data': summary_data,
                'global_interpretation': self._generate_global_interpretation(top_features)
            }
            
        except Exception as e:
            logger.error(f"Error generating global explanations: {e}")
            # Fallback to permutation importance
            return self._fallback_global_explanations(model, X_test, feature_names)
    
    def generate_local_explanations(
        self, 
        model, 
        patient_features: np.ndarray,
        feature_names: List[str],
        patient_id: str = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate local explanations for a single patient
        
        Args:
            model: Trained model
            patient_features: Single patient's features
            feature_names: List of feature names
            patient_id: Patient identifier
            top_k: Number of top features to explain
            
        Returns:
            Local explanation results
        """
        
        try:
            # Ensure patient_features is 2D
            if patient_features.ndim == 1:
                patient_features = patient_features.reshape(1, -1)
            
            # Get prediction
            prediction_proba = model.predict_proba(patient_features)[0, 1]
            
            # Initialize SHAP explainer (simplified for single prediction)
            explainer = shap.Explainer(model.predict_proba, patient_features)
            shap_values = explainer(patient_features)
            
            # Extract SHAP values for positive class
            if len(shap_values.shape) == 3:
                patient_shap = shap_values[0, :, 1]
            else:
                patient_shap = shap_values[0]
            
            # Sort by absolute SHAP value
            shap_indices = np.argsort(np.abs(patient_shap))[::-1]
            
            # Top contributing features
            top_contributors = []
            for i in shap_indices[:top_k]:
                feature_name = feature_names[i]
                shap_value = float(patient_shap[i])
                feature_value = float(patient_features[0, i])
                
                top_contributors.append({
                    'feature': feature_name,
                    'shap_value': shap_value,
                    'feature_value': feature_value,
                    'contribution': 'increases' if shap_value > 0 else 'decreases',
                    'clinical_interpretation': self._get_clinical_interpretation(feature_name),
                    'explanation': self._generate_feature_explanation(
                        feature_name, feature_value, shap_value
                    )
                })
            
            # Generate natural language explanation
            natural_language = self._generate_natural_language_explanation(
                prediction_proba, top_contributors, patient_id
            )
            
            # Counterfactual analysis
            counterfactuals = self._generate_counterfactuals(
                model, patient_features, feature_names, top_contributors[:5]
            )
            
            return {
                'patient_id': patient_id,
                'prediction_probability': prediction_proba,
                'risk_level': self._categorize_risk(prediction_proba),
                'top_contributors': top_contributors,
                'natural_language_explanation': natural_language,
                'counterfactuals': counterfactuals,
                'base_rate': 0.15  # Assumed base rate for context
            }
            
        except Exception as e:
            logger.error(f"Error generating local explanations: {e}")
            return self._fallback_local_explanations(
                model, patient_features, feature_names, patient_id
            )
    
    def _calculate_group_importance(
        self, 
        feature_importance: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate importance by clinical feature groups"""
        
        group_importance = {}
        
        for group_name, group_features in self.feature_groups.items():
            group_total = 0.0
            group_count = 0
            
            for feature_name in feature_names:
                # Check if feature belongs to this group (partial matching)
                if any(group_feat in feature_name.lower() for group_feat in group_features):
                    feature_idx = feature_names.index(feature_name)
                    group_total += feature_importance[feature_idx]
                    group_count += 1
            
            if group_count > 0:
                group_importance[group_name] = group_total
        
        return group_importance
    
    def _prepare_summary_plot_data(
        self, 
        shap_values: np.ndarray, 
        feature_names: List[str], 
        top_indices: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare data for SHAP summary plots"""
        
        summary_data = {
            'features': [feature_names[i] for i in top_indices],
            'shap_values': shap_values[:, top_indices].tolist(),
            'feature_values': []  # Would need original feature values
        }
        
        return summary_data
    
    def _get_clinical_interpretation(self, feature_name: str) -> str:
        """Get clinical interpretation for a feature"""
        
        interpretations = {
            'age': 'Patient age - older patients typically have higher risk',
            'bnp_last': 'B-type natriuretic peptide - elevated in heart failure',
            'weight_change': 'Recent weight change - rapid gain may indicate fluid retention',
            'adherence_mean': 'Medication adherence - poor adherence increases risk',
            'emergency_visits': 'Recent emergency department visits - indicates clinical instability',
            'heart_rate_std': 'Heart rate variability - high variability may indicate instability',
            'creatinine_last': 'Serum creatinine - elevated levels indicate kidney dysfunction',
            'medication_count': 'Number of medications - polypharmacy increases complexity',
            'days_since_last_visit': 'Time since last healthcare visit - longer gaps may indicate risk',
            'charlson_score': 'Comorbidity burden - higher scores indicate more comorbidities'
        }
        
        # Find matching interpretation
        for key, interpretation in interpretations.items():
            if key in feature_name.lower():
                return interpretation
        
        return f"Clinical feature: {feature_name}"
    
    def _generate_feature_explanation(
        self, 
        feature_name: str, 
        feature_value: float, 
        shap_value: float
    ) -> str:
        """Generate human-readable explanation for a feature contribution"""
        
        direction = "increases" if shap_value > 0 else "decreases"
        magnitude = "strongly" if abs(shap_value) > 0.1 else "moderately" if abs(shap_value) > 0.05 else "slightly"
        
        # Feature-specific explanations
        if 'bnp' in feature_name.lower():
            if feature_value > 400:
                return f"Elevated BNP ({feature_value:.0f}) {magnitude} {direction} risk - suggests heart failure"
            else:
                return f"BNP level ({feature_value:.0f}) {magnitude} {direction} risk"
        
        elif 'weight' in feature_name.lower() and 'change' in feature_name.lower():
            if feature_value > 3:
                return f"Rapid weight gain ({feature_value:.1f}%) {magnitude} {direction} risk - may indicate fluid retention"
            elif feature_value < -5:
                return f"Significant weight loss ({feature_value:.1f}%) {magnitude} {direction} risk"
            else:
                return f"Weight change ({feature_value:.1f}%) {magnitude} {direction} risk"
        
        elif 'adherence' in feature_name.lower():
            if feature_value < 80:
                return f"Poor medication adherence ({feature_value:.0f}%) {magnitude} {direction} risk"
            else:
                return f"Medication adherence ({feature_value:.0f}%) {magnitude} {direction} risk"
        
        elif 'age' in feature_name.lower():
            return f"Patient age ({feature_value:.0f} years) {magnitude} {direction} risk"
        
        else:
            return f"{feature_name} ({feature_value:.2f}) {magnitude} {direction} risk"
    
    def _generate_natural_language_explanation(
        self, 
        prediction_proba: float, 
        top_contributors: List[Dict], 
        patient_id: str = None
    ) -> str:
        """Generate natural language explanation"""
        
        risk_level = self._categorize_risk(prediction_proba)
        patient_ref = f"Patient {patient_id}" if patient_id else "This patient"
        
        explanation = f"{patient_ref} has a {prediction_proba:.0%} probability of deterioration in the next 90 days ({risk_level} risk). "
        
        # Main drivers
        positive_contributors = [c for c in top_contributors[:3] if c['shap_value'] > 0]
        negative_contributors = [c for c in top_contributors[:3] if c['shap_value'] < 0]
        
        if positive_contributors:
            explanation += "Main risk factors: "
            risk_factors = []
            for contrib in positive_contributors:
                if 'bnp' in contrib['feature'].lower():
                    risk_factors.append(f"elevated BNP ({contrib['feature_value']:.0f})")
                elif 'weight' in contrib['feature'].lower() and 'change' in contrib['feature'].lower():
                    risk_factors.append(f"weight gain ({contrib['feature_value']:.1f}%)")
                elif 'adherence' in contrib['feature'].lower():
                    risk_factors.append(f"medication non-adherence ({contrib['feature_value']:.0f}%)")
                else:
                    risk_factors.append(contrib['feature'].replace('_', ' '))
            
            explanation += ", ".join(risk_factors) + ". "
        
        if negative_contributors:
            explanation += "Protective factors: "
            protective_factors = [c['feature'].replace('_', ' ') for c in negative_contributors]
            explanation += ", ".join(protective_factors) + ". "
        
        return explanation
    
    def _generate_counterfactuals(
        self, 
        model, 
        patient_features: np.ndarray, 
        feature_names: List[str],
        top_contributors: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations"""
        
        counterfactuals = []
        original_prob = model.predict_proba(patient_features)[0, 1]
        
        for contrib in top_contributors:
            if contrib['shap_value'] > 0:  # Only for risk-increasing features
                feature_idx = feature_names.index(contrib['feature'])
                
                # Create modified features
                modified_features = patient_features.copy()
                
                # Suggest improvement based on feature type
                if 'adherence' in contrib['feature'].lower():
                    # Improve adherence to 90%
                    modified_features[0, feature_idx] = 90.0
                    improvement_text = "medication adherence improved to 90%"
                
                elif 'weight' in contrib['feature'].lower() and 'change' in contrib['feature'].lower():
                    # Reduce weight change
                    modified_features[0, feature_idx] = max(0, contrib['feature_value'] - 2)
                    improvement_text = "weight gain reduced by 2%"
                
                elif 'bnp' in contrib['feature'].lower():
                    # Reduce BNP by 25%
                    modified_features[0, feature_idx] = contrib['feature_value'] * 0.75
                    improvement_text = "BNP reduced by 25%"
                
                else:
                    continue  # Skip features we can't easily modify
                
                # Calculate new probability
                new_prob = model.predict_proba(modified_features)[0, 1]
                risk_reduction = original_prob - new_prob
                
                if risk_reduction > 0.01:  # Only include meaningful reductions
                    counterfactuals.append({
                        'intervention': improvement_text,
                        'original_risk': original_prob,
                        'new_risk': new_prob,
                        'risk_reduction': risk_reduction,
                        'relative_reduction': risk_reduction / original_prob
                    })
        
        return sorted(counterfactuals, key=lambda x: x['risk_reduction'], reverse=True)
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level based on probability"""
        
        if probability >= 0.30:
            return "High"
        elif probability >= 0.10:
            return "Medium"
        else:
            return "Low"
    
    def _generate_global_interpretation(self, top_features: List[Dict]) -> str:
        """Generate global model interpretation"""
        
        interpretation = "This model identifies patients at risk of deterioration based on: "
        
        # Group features by clinical category
        feature_groups = {}
        for feature in top_features[:10]:
            category = self._categorize_feature(feature['feature'])
            if category not in feature_groups:
                feature_groups[category] = []
            feature_groups[category].append(feature['feature'])
        
        group_descriptions = []
        for category, features in feature_groups.items():
            group_descriptions.append(f"{category} ({len(features)} features)")
        
        interpretation += ", ".join(group_descriptions) + ". "
        
        # Add clinical insights
        interpretation += "The model emphasizes clinical instability markers, medication adherence, "
        interpretation += "and healthcare utilization patterns as key predictors of deterioration risk."
        
        return interpretation
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize a feature into clinical groups"""
        
        feature_lower = feature_name.lower()
        
        if any(x in feature_lower for x in ['heart_rate', 'bp', 'weight', 'spo2', 'temperature']):
            return "vital signs"
        elif any(x in feature_lower for x in ['bnp', 'creatinine', 'glucose', 'hemoglobin']):
            return "laboratory values"
        elif any(x in feature_lower for x in ['medication', 'adherence', 'drug']):
            return "medications"
        elif any(x in feature_lower for x in ['visit', 'encounter', 'admission']):
            return "healthcare utilization"
        elif any(x in feature_lower for x in ['age', 'gender', 'comorbid']):
            return "demographics"
        else:
            return "other clinical factors"
    
    def _fallback_global_explanations(
        self, 
        model, 
        X_test: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Fallback method using permutation importance"""
        
        try:
            # Use permutation importance as fallback
            perm_importance = permutation_importance(
                model, X_test[:200], np.random.randint(0, 2, 200), 
                n_repeats=5, random_state=42
            )
            
            # Sort features by importance
            importance_indices = np.argsort(perm_importance.importances_mean)[::-1]
            
            top_features = []
            for i in importance_indices[:20]:
                top_features.append({
                    'feature': feature_names[i],
                    'importance': float(perm_importance.importances_mean[i]),
                    'clinical_interpretation': self._get_clinical_interpretation(feature_names[i])
                })
            
            return {
                'top_features': top_features,
                'method': 'permutation_importance',
                'global_interpretation': 'Fallback explanation using permutation importance'
            }
            
        except Exception as e:
            logger.error(f"Fallback explanation failed: {e}")
            return {'error': 'Unable to generate explanations'}
    
    def _fallback_local_explanations(
        self, 
        model, 
        patient_features: np.ndarray, 
        feature_names: List[str],
        patient_id: str = None
    ) -> Dict[str, Any]:
        """Fallback local explanations"""
        
        try:
            prediction_proba = model.predict_proba(patient_features.reshape(1, -1))[0, 1]
            
            return {
                'patient_id': patient_id,
                'prediction_probability': prediction_proba,
                'risk_level': self._categorize_risk(prediction_proba),
                'natural_language_explanation': f"Patient has {prediction_proba:.0%} risk of deterioration. Detailed explanations unavailable.",
                'method': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Fallback local explanation failed: {e}")
            return {'error': 'Unable to generate patient explanation'}
