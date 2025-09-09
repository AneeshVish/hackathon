"""
Script to process patient data from CSV files and generate formatted data for the dashboard.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Configuration
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("dashboard/src/data")

# Sample names for generating realistic patient data
FIRST_NAMES = ["John", "Jane", "Robert", "Mary", "Michael", "Jennifer", "William", "Linda", 
               "David", "Patricia", "James", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica", 
               "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia",
              "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez"]

# Risk thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.7,
    'high': 1.0
}

def load_patient_data():
    """Load and process patient data from CSV files."""
    print("Loading patient data...")
    
    try:
        # Load demographics
        demographics = pd.read_csv(DATA_DIR / "demographics.csv")
        
        # Load vitals
        vitals = pd.read_csv(DATA_DIR / "vitals.csv")
        
        # Load labs
        labs = pd.read_csv(DATA_DIR / "labs.csv")
        
        # Load medications
        meds = pd.read_csv(DATA_DIR / "medications.csv")
        
        # Process and merge data (simplified for example)
        # In a real scenario, you would implement proper data merging and feature engineering
        
        # For now, let's generate some sample data with realistic values
        patient_ids = demographics['patient_id'].unique()
        
        return patient_ids
        
    except Exception as e:
        print(f"Error loading patient data: {e}")
        return None

def generate_patient_details(patient_id):
    """Generate realistic patient details with risk scores."""
    # Generate a random risk score (replace with actual model prediction)
    risk_score = round(random.uniform(0.1, 0.95), 2)
    
    # Determine risk category
    risk_category = (
        'high' if risk_score > RISK_THRESHOLDS['medium']
        else 'medium' if risk_score > RISK_THRESHOLDS['low']
        else 'low'
    )
    
    # Generate a random name
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    
    # Generate vitals based on risk category
    if risk_category == 'high':
        vitals = {
            'heart_rate': random.randint(90, 120),
            'systolic_bp': random.randint(140, 170),
            'diastolic_bp': random.randint(85, 110),
            'temperature': round(random.uniform(37.5, 39.0), 1),
            'respiratory_rate': random.randint(20, 30),
            'spo2': random.randint(85, 92),
            'weight': round(random.uniform(70, 120), 1),
            'weight_trend': round(random.uniform(1.5, 4.0), 1)
        }
    elif risk_category == 'medium':
        vitals = {
            'heart_rate': random.randint(80, 100),
            'systolic_bp': random.randint(130, 150),
            'diastolic_bp': random.randint(80, 95),
            'temperature': round(random.uniform(36.8, 37.5), 1),
            'respiratory_rate': random.randint(16, 22),
            'spo2': random.randint(92, 96),
            'weight': round(random.uniform(65, 110), 1),
            'weight_trend': round(random.uniform(0.5, 2.0), 1)
        }
    else:
        vitals = {
            'heart_rate': random.randint(60, 85),
            'systolic_bp': random.randint(110, 135),
            'diastolic_bp': random.randint(70, 85),
            'temperature': round(random.uniform(36.5, 37.2), 1),
            'respiratory_rate': random.randint(12, 18),
            'spo2': random.randint(96, 100),
            'weight': round(random.uniform(60, 100), 1),
            'weight_trend': round(random.uniform(-0.5, 1.0), 1)
        }
    
    # Generate interventions based on risk
    if risk_category == 'high':
        interventions = [
            'Schedule urgent follow-up',
            'Consider diuretics adjustment',
            'Daily weight monitoring',
            'Supplemental oxygen as needed'
        ]
    elif risk_category == 'medium':
        interventions = [
            'Increase diuretic dose',
            'Low-sodium diet education',
            'Bi-weekly weight checks'
        ]
    else:
        interventions = [
            'Continue current regimen',
            'Routine follow-up in 1 month'
        ]
    
    # Generate dates
    last_visit = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
    next_appointment = (datetime.now() + timedelta(days=random.randint(1, 14))).strftime('%Y-%m-%d')
    
    return {
        'patient_id': str(patient_id),
        'mrn': str(patient_id),
        'name': f"{first_name} {last_name}",
        'first_name': first_name,
        'last_name': last_name,
        'age': random.randint(45, 90),
        'gender': random.choice(['M', 'F']),
        'risk_score': risk_score,
        'risk_category': risk_category,
        'last_updated': datetime.now().isoformat(),
        'vital_signs': vitals,
        'interventions': interventions,
        'key_risk_factors': [
            f"{random.choice(['Elevated', 'Decreased', 'Irregular'])} {random.choice(['heart rate', 'blood pressure', 'respiratory rate'])}",
            f"{random.choice(['Recent', 'History of', 'Chronic'])} {random.choice(['hypertension', 'diabetes', 'COPD', 'heart failure'])}",
            f"{random.choice(['Weight', 'Blood sugar', 'Cholesterol'])} {random.choice(['fluctuation', 'elevation', 'instability'])}"
        ],
        'last_visit': last_visit,
        'next_appointment': next_appointment
    }

def generate_patient_data():
    """Generate complete patient data for the dashboard."""
    # Load patient IDs from the data
    patient_ids = load_patient_data()
    
    if patient_ids is None:
        print("No patient data found. Using sample data.")
        patient_ids = [f"{i:03d}" for i in range(1, 56)]  # Generate 55 patient IDs
    
    print(f"Generating data for {len(patient_ids)} patients...")
    
    # Generate data for each patient
    patients = []
    for i, patient_id in enumerate(patient_ids, 1):
        patient_data = generate_patient_details(patient_id)
        patients.append(patient_data)
        
        if i % 10 == 0 or i == len(patient_ids):
            print(f"Processed {i}/{len(patient_ids)} patients")
    
    # Calculate risk distribution
    risk_counts = {'high': 0, 'medium': 0, 'low': 0}
    for patient in patients:
        risk_counts[patient['risk_category']] += 1
    
    # Create output data
    output = {
        'last_updated': datetime.now().isoformat(),
        'total_patients': len(patients),
        'risk_distribution': risk_counts,
        'patients': patients
    }
    
    # Save to file
    output_path = OUTPUT_DIR / "patientData.js"
    with open(output_path, 'w') as f:
        f.write(f"// Auto-generated by process_patient_data.py\n")
        f.write(f"// Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("export const patientData = ")
        json.dump(patients, f, indent=2)
    
    print(f"\nPatient data saved to {output_path}")
    print(f"Risk distribution: {risk_counts}")

if __name__ == "__main__":
    generate_patient_data()
