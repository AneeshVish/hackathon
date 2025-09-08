"""
Comprehensive model evaluation module for patient deterioration prediction.
Implements clinical validation metrics, calibration analysis, and subgroup evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation for clinical ML"""
    
    def __init__(self):
        self.evaluation_history = []
        
    def evaluate_model(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with clinical metrics
        
        Args:
            model: Trained model with predict_proba method
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Basic classification metrics
        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'auprc': average_precision_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_test, y_pred)
        }
        
        # Confidence intervals for AUC
        auc_ci = self._bootstrap_auc_ci(y_test, y_pred_proba)
        metrics['auc_ci_lower'] = auc_ci[0]
        metrics['auc_ci_upper'] = auc_ci[1]
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
        # PPV and NPV
        metrics['ppv'] = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        metrics['npv'] = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
        
        # Multiple threshold analysis
        threshold_metrics = self._analyze_multiple_thresholds(y_test, y_pred_proba)
        metrics['threshold_analysis'] = threshold_metrics
        
        # Calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(y_test, y_pred_proba)
        metrics.update(calibration_metrics)
        
        return metrics
    
    def analyze_calibration(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Detailed calibration analysis
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            n_bins: Number of bins for calibration curve
            
        Returns:
            Calibration analysis results
        """
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=n_bins, strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_test, y_pred_proba, n_bins)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(y_test, y_pred_proba, n_bins)
        
        # Reliability diagram data
        reliability_data = []
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_test[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                
                reliability_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'proportion': prop_in_bin,
                    'count': in_bin.sum()
                })
        
        return {
            'ece': ece,
            'mce': mce,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            },
            'reliability_data': reliability_data,
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'brier_skill_score': self._calculate_brier_skill_score(y_test, y_pred_proba)
        }
    
    def subgroup_analysis(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        subgroup_features: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze model performance across different subgroups
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            subgroup_features: Dictionary mapping subgroup names to feature arrays
            
        Returns:
            Subgroup analysis results
        """
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        subgroup_results = {}
        
        for subgroup_name, subgroup_mask in subgroup_features.items():
            if subgroup_mask.sum() > 10:  # Minimum samples for meaningful analysis
                subgroup_metrics = self.evaluate_model(
                    model, X_test[subgroup_mask], y_test[subgroup_mask]
                )
                subgroup_results[subgroup_name] = subgroup_metrics
        
        # Fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(
            y_test, y_pred_proba, subgroup_features
        )
        
        return {
            'subgroup_performance': subgroup_results,
            'fairness_metrics': fairness_metrics
        }
    
    def decision_curve_analysis(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        threshold_range: Tuple[float, float] = (0.0, 1.0),
        n_thresholds: int = 100
    ) -> Dict[str, Any]:
        """
        Decision curve analysis for clinical utility
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            threshold_range: Range of thresholds to analyze
            n_thresholds: Number of thresholds to evaluate
            
        Returns:
            Decision curve analysis results
        """
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        
        net_benefits = []
        treat_all_benefits = []
        treat_none_benefits = []
        
        prevalence = y_test.mean()
        
        for threshold in thresholds:
            # Model-based strategy
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            tp = ((y_pred == 1) & (y_test == 1)).sum()
            fp = ((y_pred == 1) & (y_test == 0)).sum()
            
            net_benefit = (tp / len(y_test)) - (fp / len(y_test)) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)
            
            # Treat all strategy
            treat_all_benefit = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
            treat_all_benefits.append(max(0, treat_all_benefit))
            
            # Treat none strategy
            treat_none_benefits.append(0)
        
        return {
            'thresholds': thresholds.tolist(),
            'net_benefits': net_benefits,
            'treat_all_benefits': treat_all_benefits,
            'treat_none_benefits': treat_none_benefits,
            'prevalence': prevalence
        }
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp = cm[0, 0], cm[0, 1]
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _bootstrap_auc_ci(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for AUC"""
        
        bootstrap_aucs = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_scores_boot = y_scores[indices]
            
            # Skip if all labels are the same
            if len(np.unique(y_true_boot)) > 1:
                auc_boot = roc_auc_score(y_true_boot, y_scores_boot)
                bootstrap_aucs.append(auc_boot)
        
        if bootstrap_aucs:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_aucs, lower_percentile)
            ci_upper = np.percentile(bootstrap_aucs, upper_percentile)
            
            return ci_lower, ci_upper
        else:
            return 0.0, 1.0
    
    def _analyze_multiple_thresholds(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze performance at multiple thresholds"""
        
        # Common clinical thresholds
        thresholds = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
        
        threshold_results = {}
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            threshold_results[f"threshold_{threshold}"] = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }
        
        return threshold_results
    
    def _calculate_calibration_metrics(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray
    ) -> Dict[str, float]:
        """Calculate calibration-related metrics"""
        
        # Hosmer-Lemeshow test (simplified)
        hl_statistic, hl_p_value = self._hosmer_lemeshow_test(y_true, y_scores)
        
        return {
            'hosmer_lemeshow_statistic': hl_statistic,
            'hosmer_lemeshow_p_value': hl_p_value,
            'calibration_slope': self._calibration_slope(y_true, y_scores),
            'calibration_intercept': self._calibration_intercept(y_true, y_scores)
        }
    
    def _calculate_ece(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_scores[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """Calculate Maximum Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_scores[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    def _calculate_brier_skill_score(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray
    ) -> float:
        """Calculate Brier Skill Score"""
        
        brier_score = brier_score_loss(y_true, y_scores)
        
        # Reference forecast (climatological probability)
        climatological_prob = y_true.mean()
        reference_brier = brier_score_loss(y_true, np.full_like(y_scores, climatological_prob))
        
        if reference_brier == 0:
            return 0.0
        
        brier_skill_score = 1 - (brier_score / reference_brier)
        return brier_skill_score
    
    def _hosmer_lemeshow_test(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        n_bins: int = 10
    ) -> Tuple[float, float]:
        """Simplified Hosmer-Lemeshow goodness-of-fit test"""
        
        # Create bins based on predicted probabilities
        bin_edges = np.percentile(y_scores, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-8  # Ensure all values are included
        
        observed = []
        expected = []
        
        for i in range(n_bins):
            mask = (y_scores >= bin_edges[i]) & (y_scores < bin_edges[i + 1])
            
            if mask.sum() > 0:
                obs_pos = y_true[mask].sum()
                exp_pos = y_scores[mask].sum()
                
                observed.extend([obs_pos, mask.sum() - obs_pos])
                expected.extend([exp_pos, mask.sum() - exp_pos])
        
        if len(observed) > 0 and all(e > 0 for e in expected):
            statistic = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
            df = len(observed) - 2
            p_value = 1 - stats.chi2.cdf(statistic, df) if df > 0 else 1.0
            return statistic, p_value
        else:
            return 0.0, 1.0
    
    def _calibration_slope(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate calibration slope"""
        
        # Logistic regression of outcomes on logit of predicted probabilities
        logit_scores = np.log(y_scores / (1 - y_scores + 1e-15))
        
        try:
            slope, _, _, _, _ = stats.linregress(logit_scores, y_true)
            return slope
        except:
            return 1.0
    
    def _calibration_intercept(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate calibration intercept"""
        
        logit_scores = np.log(y_scores / (1 - y_scores + 1e-15))
        
        try:
            _, intercept, _, _, _ = stats.linregress(logit_scores, y_true)
            return intercept
        except:
            return 0.0
    
    def _calculate_fairness_metrics(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        subgroup_features: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate fairness metrics across subgroups"""
        
        fairness_results = {}
        
        # Calculate AUC for each subgroup
        subgroup_aucs = {}
        for subgroup_name, mask in subgroup_features.items():
            if mask.sum() > 10 and len(np.unique(y_true[mask])) > 1:
                auc = roc_auc_score(y_true[mask], y_scores[mask])
                subgroup_aucs[subgroup_name] = auc
        
        if len(subgroup_aucs) > 1:
            # Equalized odds difference
            fairness_results['auc_differences'] = {
                f"{group1}_vs_{group2}": abs(auc1 - auc2)
                for group1, auc1 in subgroup_aucs.items()
                for group2, auc2 in subgroup_aucs.items()
                if group1 != group2
            }
            
            # Overall fairness score (lower is more fair)
            auc_values = list(subgroup_aucs.values())
            fairness_results['auc_range'] = max(auc_values) - min(auc_values)
            fairness_results['auc_std'] = np.std(auc_values)
        
        return fairness_results
    
    def generate_evaluation_report(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_name: str = "Model"
    ) -> str:
        """Generate a comprehensive evaluation report"""
        
        # Basic evaluation
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Calibration analysis
        calibration = self.analyze_calibration(model, X_test, y_test)
        
        report = f"""
# {model_name} Evaluation Report

## Performance Metrics
- **AUROC**: {metrics['auc']:.3f} (95% CI: {metrics['auc_ci_lower']:.3f}-{metrics['auc_ci_upper']:.3f})
- **AUPRC**: {metrics['auprc']:.3f}
- **Brier Score**: {metrics['brier_score']:.3f}

## Classification Performance (threshold = 0.5)
- **Sensitivity**: {metrics['recall']:.3f}
- **Specificity**: {metrics['specificity']:.3f}
- **PPV**: {metrics['ppv']:.3f}
- **NPV**: {metrics['npv']:.3f}
- **F1 Score**: {metrics['f1_score']:.3f}

## Confusion Matrix
|           | Predicted |         |
|-----------|-----------|---------|
| **Actual**| Negative  | Positive|
| Negative  | {metrics['confusion_matrix']['tn']:>8} | {metrics['confusion_matrix']['fp']:>8}|
| Positive  | {metrics['confusion_matrix']['fn']:>8} | {metrics['confusion_matrix']['tp']:>8}|

## Calibration Analysis
- **Expected Calibration Error**: {calibration['ece']:.3f}
- **Brier Skill Score**: {calibration['brier_skill_score']:.3f}
- **Hosmer-Lemeshow p-value**: {calibration['hosmer_lemeshow_p_value']:.3f}

## Clinical Interpretation
- At 25% threshold: Sensitivity = {metrics['threshold_analysis']['threshold_0.25']['sensitivity']:.3f}, Specificity = {metrics['threshold_analysis']['threshold_0.25']['specificity']:.3f}
- At 30% threshold: Sensitivity = {metrics['threshold_analysis']['threshold_0.3']['sensitivity']:.3f}, Specificity = {metrics['threshold_analysis']['threshold_0.3']['specificity']:.3f}

## Recommendations
"""
        
        # Add recommendations based on performance
        if metrics['auc'] >= 0.8:
            report += "- Model shows good discrimination ability (AUC ≥ 0.8)\n"
        elif metrics['auc'] >= 0.7:
            report += "- Model shows acceptable discrimination ability (AUC ≥ 0.7)\n"
        else:
            report += "- Model discrimination may need improvement (AUC < 0.7)\n"
        
        if calibration['ece'] <= 0.1:
            report += "- Model is well-calibrated (ECE ≤ 0.1)\n"
        else:
            report += "- Model calibration may need improvement (ECE > 0.1)\n"
        
        return report
