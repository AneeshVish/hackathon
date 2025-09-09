"""
Script to generate realistic patient data for the dashboard.
"""

import json
import random
from datetime import datetime, timedelta
import numpy as np

# Configuration
TOTAL_PATIENTS = 55
RISK_DISTRIBUTION = {'high': 0.2, 'medium': 0.3, 'low': 0.5}

# Generate patient data
def generate_patient(patient_id):
    """Generate a single patient's data."""
    # Determine risk bucket based on distribution
    rand = random.random()
    if rand < RISK_DISTRIBUTION['high']:
        risk_bucket = 'high'
        risk_score = round(random.uniform(0.7, 0.95), 4)
        vital_signs = {
            'heart_rate': random.randint(90, 120),
            'blood_pressure': f"{random.randint(140, 170)}/{random.randint(85, 110)}",
            'temperature': round(random.uniform(37.5, 39.0), 1),
            'respiratory_rate': random.randint(20, 30),
            'spo2': random.randint(85, 92),
            'weight': round(random.uniform(70, 120), 1),
            'weight_trend': round(random.uniform(1.5, 4.0), 1)
        }
        interventions = [
            'Schedule urgent follow-up',
            'Consider diuretics adjustment',
            'Daily weight monitoring',
            'Supplemental oxygen as needed'
        ]
    elif rand < RISK_DISTRIBUTION['high'] + RISK_DISTRIBUTION['medium']:
        risk_bucket = 'medium'
        risk_score = round(random.uniform(0.4, 0.7), 4)
        vital_signs = {
            'heart_rate': random.randint(80, 100),
            'blood_pressure': f"{random.randint(130, 150)}/{random.randint(80, 95)}",
            'temperature': round(random.uniform(36.8, 37.5), 1),
            'respiratory_rate': random.randint(16, 22),
            'spo2': random.randint(92, 96),
            'weight': round(random.uniform(65, 110), 1),
            'weight_trend': round(random.uniform(0.5, 2.0), 1)
        }
        interventions = [
            'Increase diuretic dose',
            'Low-sodium diet education',
            'Bi-weekly weight checks'
        ]
    else:
        risk_bucket = 'low'
        risk_score = round(random.uniform(0.1, 0.4), 4)
        vital_signs = {
            'heart_rate': random.randint(60, 85),
            'blood_pressure': f"{random.randint(110, 135)}/{random.randint(70, 85)}",
            'temperature': round(random.uniform(36.5, 37.2), 1),
            'respiratory_rate': random.randint(12, 18),
            'spo2': random.randint(96, 100),
            'weight': round(random.uniform(60, 100), 1),
            'weight_trend': round(random.uniform(-0.5, 1.0), 1)
        }
        interventions = [
            'Continue current regimen',
            'Routine follow-up in 1 month'
        ]
    
    # Generate key risk factors based on risk bucket
    if risk_bucket == 'high':
        key_risk_factors = [
            f'Rising weight trend ({vital_signs["weight_trend"]}kg in 7 days)',
            f'Decreasing SpO2 levels ({vital_signs["spo2"]}% current)',
            f'Elevated respiratory rate ({vital_signs["respiratory_rate"]} bpm)'
        ]
    elif risk_bucket == 'medium':
        key_risk_factors = [
            f'Mild weight gain ({vital_signs["weight_trend"]}kg in 7 days)',
            'Stable but elevated blood pressure',
            'Mild peripheral edema'
        ]
    else:
        key_risk_factors = [
            'Stable weight',
            'Good medication adherence',
            'No recent hospitalizations'
        ]
    
    # Generate dates
    last_visit = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
    next_appointment = (datetime.now() + timedelta(days=random.randint(1, 14))).strftime('%Y-%m-%d')
    
    return {
        'patient_id': f'PT_{1000 + patient_id}',
        'name': f'Patient {1000 + patient_id}',
        'age': random.randint(55, 90),
        'gender': random.choice(['M', 'F']),
        'risk_score': risk_score,
        'risk_bucket': risk_bucket,
        'prediction_confidence': random.choice(['High', 'Medium', 'Low']),
        'last_visit': last_visit,
        'next_appointment': next_appointment,
        'key_risk_factors': key_risk_factors,
        'vital_signs': vital_signs,
        'interventions': interventions
    }

def generate_dashboard_data():
    """Generate complete dashboard data."""
    # Generate patient data
    patients = [generate_patient(i) for i in range(TOTAL_PATIENTS)]
    
    # Calculate risk distribution
    risk_counts = {
        'high': len([p for p in patients if p['risk_bucket'] == 'high']),
        'medium': len([p for p in patients if p['risk_bucket'] == 'medium']),
        'low': len([p for p in patients if p['risk_bucket'] == 'low']),
        'total': TOTAL_PATIENTS
    }
    
    # Model metrics (example values)
    model_metrics = {
        'accuracy': 0.7231,
        'precision': 0.8214,
        'recall': 0.8519,
        'f1': 0.8364,
        'auc': 0.8199,
        'avg_precision': 0.9657,
        'brier_score': 0.2478,
        'specificity': 0.5,
        'npv': 0.9,
        'ppv': 0.8214
    }
    
    # Confusion matrix (example values)
    confusion_matrix = {
        'true_positive': 23,
        'false_positive': 5,
        'true_negative': 22,
        'false_negative': 4
    }
    
    # Feature importance (example values)
    feature_importance = [
        {'feature': 'Weight 7d Trend', 'importance': 0.0517, 'description': '7-day weight change trend'},
        {'feature': 'Heart Rate 14d Trend', 'importance': 0.0517, 'description': '14-day heart rate trend'},
        {'feature': 'Weight 14d Trend', 'importance': 0.0517, 'description': '14-day weight change trend'},
        {'feature': 'Respiratory Rate 30d Trend', 'importance': 0.0517, 'description': '30-day respiratory rate trend'},
        {'feature': 'Temperature 30d Trend', 'importance': 0.0517, 'description': '30-day temperature trend'},
        {'feature': 'Age Group', 'importance': 0.0345, 'description': 'Patient age category'},
        {'feature': 'Diastolic BP 14d Trend', 'importance': 0.0345, 'description': '14-day diastolic BP trend'},
        {'feature': 'SpO2 14d Count', 'importance': 0.0345, 'description': '14-day oxygen saturation measurements'},
        {'feature': 'Systolic BP 14d Trend', 'importance': 0.0345, 'description': '14-day systolic BP trend'},
        {'feature': 'Weight 14d Count', 'importance': 0.0345, 'description': '14-day weight measurements'}
    ]
    
    # Patient stats
    patient_stats = {
        'total': TOTAL_PATIENTS,
        'high': risk_counts['high'],
        'medium': risk_counts['medium'],
        'low': risk_counts['low'],
        'avgRiskScore': round(sum(p['risk_score'] for p in patients) / TOTAL_PATIENTS, 4),
        'highRiskPercentage': round((risk_counts['high'] / TOTAL_PATIENTS) * 100, 2)
    }
    
    return {
        'model_metrics': model_metrics,
        'confusion_matrix': confusion_matrix,
        'feature_importance': feature_importance,
        'patients': sorted(patients, key=lambda x: x['risk_score'], reverse=True),
        'risk_distribution': risk_counts,
        'patient_stats': patient_stats
    }

def save_to_js_file(data, output_file):
    """Save data as a JavaScript module."""
    js_content = f"""// Auto-generated patient data for dashboard
// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

export const trainedModelMetrics = {model_metrics};

export const trainedConfusionMatrix = {confusion_matrix};

export const trainedFeatureImportance = {feature_importance};

export const trainedPatientPredictions = {patient_predictions};

export const trainedRiskDistribution = {risk_distribution};

export const trainedPatientStats = {patient_stats};
""".format(
        model_metrics=json.dumps(data['model_metrics'], indent=2),
        confusion_matrix=json.dumps(data['confusion_matrix'], indent=2),
        feature_importance=json.dumps(data['feature_importance'], indent=2),
        patient_predictions=json.dumps(data['patients'], indent=2),
        risk_distribution=json.dumps(data['risk_distribution'], indent=2),
        patient_stats=json.dumps(data['patient_stats'], indent=2)
    )
    
    with open(output_file, 'w') as f:
        f.write(js_content)
    
    print(f"Dashboard data saved to {output_file}")

if __name__ == "__main__":
    # Generate the data
    dashboard_data = generate_dashboard_data()
    
    # Save to dashboard file
    output_path = r"c:\Users\ShyamVenkatraman\Desktop\Hackwell\hackathon\dashboard\src\data\trainedModelData.js"
    save_to_js_file(dashboard_data, output_path)
