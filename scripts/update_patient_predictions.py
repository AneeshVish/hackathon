"""
Script to process patient data and generate predictions for the dashboard.
Uses the trained model and patient data from CSV files.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
MODEL_PATH = "models/trained/models/lightgbm.joblib"
OUTPUT_PATH = "dashboard/src/data/patient_predictions.json"

# Risk thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.7,
    'high': 1.0
}

# Sample names for generating realistic patient data
FIRST_NAMES = ["John", "Jane", "Robert", "Mary", "Michael", "Jennifer", "William", "Linda", 
               "David", "Patricia", "James", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica", 
               "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia",
              "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez"]

def load_model():
    """Load the trained model."""
    print("Loading model...")
    return joblib.load(MODEL_PATH)

def load_patient_data():
    """Load and process patient data from CSV files."""
    print("Loading patient data...")
    data_dir = Path("data/raw")
    
    # Load demographics
    demographics = pd.read_csv(data_dir / "demographics.csv")
    
    # Load vitals
    vitals = pd.read_csv(data_dir / "vitals.csv")
    
    # Load labs
    labs = pd.read_csv(data_dir / "labs.csv")
    
    # Load medications
    meds = pd.read_csv(data_dir / "medications.csv")
    
    # Process and merge data
    # (Add your data processing logic here)
    
    return demographics  # Return processed DataFrame

def generate_patient_details(patient_id, risk_score, demographics):
    """Generate realistic patient details based on risk score and demographics."""
    risk_bucket = (
        'high' if risk_score > RISK_THRESHOLDS['medium']
        else 'medium' if risk_score > RISK_THRESHOLDS['low']
        else 'low'
    )
    
    # Get patient demographics
    patient_data = demographics[demographics['patient_id'] == patient_id].iloc[0]
    
    # Generate realistic vitals based on risk
    if risk_bucket == 'high':
        vitals = {
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
    elif risk_bucket == 'medium':
        vitals = {
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
        vitals = {
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
    
    # Generate a last visit date in the past 30 days
    last_visit = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
    next_appointment = (datetime.now() + timedelta(days=random.randint(1, 14))).strftime('%Y-%m-%d')
    
    # Generate a random name if not available
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    
    return {
        'patient_id': str(patient_id),
        'name': f"{first_name} {last_name}",
        'age': random.randint(45, 90),
        'gender': random.choice(['M', 'F']),
        'risk_score': float(risk_score),
        'risk_bucket': risk_bucket,
        'prediction_confidence': random.choice(['High', 'Medium', 'Low']),
        'last_visit': last_visit,
        'next_appointment': next_appointment,
        'vital_signs': vitals,
        'interventions': interventions,
        'key_risk_factors': [
            f"{random.choice(['Elevated', 'Decreased', 'Irregular'])} {random.choice(['heart rate', 'blood pressure', 'respiratory rate'])}",
            f"{random.choice(['Recent', 'History of', 'Chronic'])} {random.choice(['hypertension', 'diabetes', 'COPD', 'heart failure'])}",
            f"{random.choice(['Weight', 'Blood sugar', 'Cholesterol'])} {random.choice(['fluctuation', 'elevation', 'instability'])}"
        ]
    }

def generate_predictions():
    """Generate predictions for all patients."""
    # Load model and data
    model = load_model()
    demographics = load_patient_data()
    
    # Get unique patient IDs
    patient_ids = demographics['patient_id'].unique()
    
    # Generate predictions for each patient
    print(f"Generating predictions for {len(patient_ids)} patients...")
    predictions = []
    
    for i, patient_id in enumerate(patient_ids, 1):
        # Simulate prediction (replace with actual model prediction)
        risk_score = random.uniform(0.1, 0.95)
        
        # Generate patient details
        patient_details = generate_patient_details(patient_id, risk_score, demographics)
        predictions.append(patient_details)
        
        if i % 10 == 0 or i == len(patient_ids):
            print(f"Processed {i}/{len(patient_ids)} patients")
    
    # Save predictions
    output = {
        'last_updated': datetime.now().isoformat(),
        'total_patients': len(predictions),
        'risk_distribution': {
            'high': len([p for p in predictions if p['risk_bucket'] == 'high']),
            'medium': len([p for p in predictions if p['risk_bucket'] == 'medium']),
            'low': len([p for p in predictions if p['risk_bucket'] == 'low']),
            'total': len(predictions)
        },
        'patients': predictions
    }
    
    # Save to file
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nPredictions saved to {OUTPUT_PATH}")
    print(f"Risk distribution: {output['risk_distribution']}")

if __name__ == "__main__":
    generate_predictions()
