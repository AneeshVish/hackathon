"""
Script to process raw patient data and generate formatted data for the dashboard.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
DATA_DIR = Path("data/raw")
OUTPUT_FILE = "dashboard/src/data/patientData.js"

# Risk thresholds for categorizing patients
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.7,
    'high': 1.0
}

def load_and_merge_data():
    """Load and merge all raw data files."""
    print("Loading and merging raw data files...")
    
    # Load all CSV files
    data_files = {
        'demographics': 'demographics.csv',
        'vitals': 'vitals.csv',
        'labs': 'labs.csv',
        'medications': 'medications.csv',
        'encounters': 'encounters.csv',
        'lifestyle': 'lifestyle.csv',
        'devices': 'devices.csv',
        'deterioration_events': 'deterioration_events.csv'
    }
    
    # Load each file into a dictionary of DataFrames
    data = {}
    for name, filename in data_files.items():
        try:
            filepath = DATA_DIR / filename
            if filepath.exists():
                data[name] = pd.read_csv(filepath, low_memory=False)
                print(f"Loaded {len(data[name])} records from {filename}")
                print(f"Columns in {filename}: {data[name].columns.tolist()}")
            else:
                print(f"Warning: {filename} not found")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    return data

def calculate_risk_scores(data):
    """Calculate risk scores based on the available data."""
    print("Calculating risk scores...")
    
    if 'demographics' not in data:
        raise ValueError("Demographics data not found. Cannot proceed with risk calculation.")
    
    # Create a base DataFrame with all patient IDs
    patients = data['demographics'].copy()
    
    # Initialize risk components
    patients['vitals_risk'] = 0.0
    patients['labs_risk'] = 0.0
    patients['meds_risk'] = 0.0
    patients['hosp_risk'] = 0.0
    
    # 1. Calculate vitals-based risk if available
    if 'vitals' in data and not data['vitals'].empty:
        # Get the most recent vitals for each patient
        latest_vitals = data['vitals'].sort_values('patient_id').groupby('patient_id').last().reset_index()
        
        # Merge with patients
        patients = patients.merge(latest_vitals, on='patient_id', how='left')
        
        # Process blood pressure (systolic/diastolic)
        if 'blood_pressure' in patients.columns:
            # Split blood pressure into systolic and diastolic
            patients[['systolic_bp', 'diastolic_bp']] = patients['blood_pressure'].str.extract('(\d+)/(\d+)').astype(float)
        
        # Add risk for abnormal vitals (checking for column existence first)
        vital_columns = {
            'heart_rate': (60, 100),
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'temperature': (36, 38),
            'respiratory_rate': (12, 20),
            'spo2': (95, 100)
        }
        
        for col, (low, high) in vital_columns.items():
            if col in patients.columns:
                patients['vitals_risk'] += np.where(
                    (patients[col] < low) | (patients[col] > high),
                    0.15, 0.0
                )
    
    # 2. Add risk from lab results if available
    if 'labs' in data and not data['labs'].empty:
        # Get most recent lab results for each patient
        latest_labs = data['labs'].sort_values('date').groupby('patient_id').last().reset_index()
        
        # Pivot lab results to have one row per patient with columns for each lab test
        lab_pivot = data['labs'].pivot_table(
            index='patient_id',
            columns='lab_test',
            values='result',
            aggfunc='last'  # Get the most recent result for each test
        ).reset_index()
        
        # Merge with patients
        patients = patients.merge(lab_pivot, on='patient_id', how='left')
        
        # Add lab-based risk (checking for column existence first)
        lab_columns = {
            'creatinine': 1.2,
            'glucose': 126,
            'hemoglobin': 12,
            'wbc_count': (4, 11)
        }
        
        for col, threshold in lab_columns.items():
            if col in patients.columns:
                if isinstance(threshold, tuple):  # Range check
                    low, high = threshold
                    patients['labs_risk'] += np.where(
                        (patients[col] < low) | (patients[col] > high),
                        0.1, 0.0
                    )
                else:  # Single threshold check
                    patients['labs_risk'] += np.where(
                        patients[col] > threshold,
                        0.1, 0.0
                    )
    
    # 3. Add risk from medications if available
    if 'medications' in data and not data['medications'].empty:
        # Count number of medications per patient
        meds_count = data['medications'].groupby('patient_id').size().reset_index(name='meds_count')
        patients = patients.merge(meds_count, on='patient_id', how='left')
        patients['meds_risk'] = np.minimum(patients['meds_count'].fillna(0) * 0.03, 0.2)
    
    # 4. Add risk from recent hospitalizations if available
    if 'encounters' in data and not data['encounters'].empty:
        try:
            # Convert date columns to datetime if they exist
            if 'start_datetime' in data['encounters'].columns:
                data['encounters']['date'] = pd.to_datetime(data['encounters']['start_datetime'].str.split().str[0], errors='coerce')
                recent_hospitalizations = data['encounters'][
                    (data['encounters']['encounter_type'] == 'inpatient') &
                    (data['encounters']['date'] > (pd.Timestamp.now() - pd.Timedelta(days=30)))
                ]
                recent_hosp_count = recent_hospitalizations.groupby('patient_id').size().reset_index(name='recent_hosp_count')
                patients = patients.merge(recent_hosp_count, on='patient_id', how='left')
                patients['hosp_risk'] = np.minimum(patients['recent_hosp_count'].fillna(0) * 0.15, 0.2)
        except Exception as e:
            print(f"Warning: Could not process hospitalization data: {str(e)}")
    
    # Calculate total risk score (0-1 scale)
    risk_components = ['vitals_risk', 'labs_risk', 'meds_risk', 'hosp_risk']
    patients['risk_score'] = patients[risk_components].sum(axis=1).clip(0, 1)
    
    # Assign risk categories
    patients['risk_category'] = pd.cut(
        patients['risk_score'],
        bins=[0, RISK_THRESHOLDS['low'], RISK_THRESHOLDS['medium'], 1],
        labels=['low', 'medium', 'high'],
        right=False
    )
    
    return patients

def format_for_dashboard(patients):
    """Format patient data for the dashboard."""
    print("Formatting data for dashboard...")
    
    formatted_patients = []
    
    # Get the list of all possible columns for reference
    all_columns = patients.columns.tolist()
    
    for _, patient in patients.iterrows():
        # Get first and last name, handling potential missing values
        first_name = patient.get('first_name', patient.get('first_name', 'Unknown'))
        last_name = patient.get('last_name', patient.get('last_name', 'Patient'))
        
        # Format basic info
        formatted = {
            'patient_id': str(patient.get('patient_id', '')),
            'mrn': str(patient.get('mrn', patient.get('patient_id', ''))),
            'name': f"{first_name} {last_name}".strip(),
            'first_name': first_name,
            'last_name': last_name,
            'age': int(patient.get('age', 0)) if pd.notna(patient.get('age')) else None,
            'gender': patient.get('gender', 'Unknown'),
            'risk_score': float(patient.get('risk_score', 0)),
            'risk_category': patient.get('risk_category', 'low'),
            'last_updated': datetime.now().isoformat(),
            'vital_signs': {},
            'medical_history': [],
            'medications': [],
            'key_risk_factors': generate_risk_factors(patient),
            'interventions': generate_interventions(patient)
        }
        
        # Add contact information if available
        contact_info = {}
        if 'phone' in patient and pd.notna(patient['phone']):
            contact_info['phone'] = str(patient['phone'])
        if 'email' in patient and pd.notna(patient['email']):
            contact_info['email'] = patient['email']
        if contact_info:
            formatted['contact'] = contact_info
        
        # Add vital signs if available
        vital_fields = [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature',
            'respiratory_rate', 'spo2', 'weight', 'height', 'bmi'
        ]
        
        for field in vital_fields:
            if field in patient and pd.notna(patient[field]):
                try:
                    formatted['vital_signs'][field] = float(patient[field])
                except (ValueError, TypeError):
                    pass  # Skip if conversion fails
        
        # Add medical history (simplified)
        medical_history = []
        condition_fields = ['conditions', 'medical_history', 'diagnoses']
        for field in condition_fields:
            if field in patient and pd.notna(patient[field]) and isinstance(patient[field], str):
                medical_history.extend([c.strip() for c in patient[field].split(';') if c.strip()])
        
        if medical_history:
            formatted['medical_history'] = list(set(medical_history))  # Remove duplicates
        
        formatted_patients.append(formatted)
    
    return formatted_patients

def generate_risk_factors(patient):
    """Generate risk factors based on patient data."""
    risk_factors = []
    
    # Add risk factors based on available data
    risk_score = patient.get('risk_score', 0)
    if risk_score > 0.7:
        risk_factors.append('High risk score')
    elif risk_score > 0.3:
        risk_factors.append('Moderate risk score')
    
    # Age-based risk
    age = patient.get('age')
    if age is not None:
        if age >= 65:
            risk_factors.append('Elderly (65+)')
        elif age >= 50:
            risk_factors.append('Middle-aged (50-64)')
    
    # Check for common conditions
    condition_fields = ['conditions', 'medical_history', 'diagnoses']
    conditions = []
    for field in condition_fields:
        if field in patient and pd.notna(patient[field]) and isinstance(patient[field], str):
            conditions.extend([c.strip().lower() for c in patient[field].split(';')])
    
    # Add condition-based risk factors
    condition_risks = {
        'diabetes': 'Diabetes',
        'hypertension': 'Hypertension',
        'heart failure': 'Heart Failure',
        'copd': 'COPD',
        'asthma': 'Asthma',
        'obesity': 'Obesity',
        'ckd': 'Chronic Kidney Disease',
        'depression': 'Depression',
        'anxiety': 'Anxiety',
        'cad': 'Coronary Artery Disease',
        'afib': 'Atrial Fibrillation'
    }
    
    for cond, display in condition_risks.items():
        if any(cond in c for c in conditions):
            risk_factors.append(display)
    
    # Lifestyle factors
    if 'smoking_status' in patient and pd.notna(patient['smoking_status']):
        if patient['smoking_status'].lower() == 'current':
            risk_factors.append('Current smoker')
        elif patient['smoking_status'].lower() == 'former':
            risk_factors.append('Former smoker')
    
    if 'bmi' in patient and pd.notna(patient['bmi']):
        bmi = float(patient['bmi'])
        if bmi >= 30:
            risk_factors.append('Obesity (BMI ≥ 30)')
        elif bmi >= 25:
            risk_factors.append('Overweight (BMI 25-30)')
    
    return risk_factors if risk_factors else ['No significant risk factors identified']

def generate_interventions(patient):
    """Generate interventions based on patient risk and conditions."""
    interventions = []
    risk_category = patient.get('risk_category', 'low')
    
    # Base interventions on risk category
    if risk_category == 'high':
        interventions.extend([
            'Schedule urgent follow-up',
            'Consider medication adjustment',
            'Daily symptom monitoring',
            'Consider home health services',
            'Review care plan with specialist'
        ])
    elif risk_category == 'medium':
        interventions.extend([
            'Schedule follow-up in 1-2 weeks',
            'Review medications',
            'Bi-weekly check-ins',
            'Monitor symptoms closely',
            'Provide patient education'
        ])
    else:
        interventions.extend([
            'Routine follow-up in 1-3 months',
            'Continue current regimen',
            'Maintain healthy lifestyle',
            'Annual wellness check',
            'Preventive care recommendations'
        ])
    
    # Add specific interventions based on conditions
    condition_fields = ['conditions', 'medical_history', 'diagnoses']
    conditions = []
    for field in condition_fields:
        if field in patient and pd.notna(patient[field]) and isinstance(patient[field], str):
            conditions.extend([c.strip().lower() for c in patient[field].split(';')])
    
    condition_interventions = {
        'diabetes': 'Monitor blood glucose levels',
        'heart failure': 'Monitor daily weights and symptoms',
        'copd': 'Pulmonary rehabilitation',
        'hypertension': 'Monitor blood pressure',
        'asthma': 'Review inhaler technique',
        'obesity': 'Nutrition counseling',
        'ckd': 'Monitor renal function',
        'depression': 'Mental health assessment',
        'anxiety': 'Stress management techniques',
        'cad': 'Cardiac rehabilitation'
    }
    
    for cond, intervention in condition_interventions.items():
        if any(cond in c for c in conditions):
            interventions.append(intervention)
    
    # Ensure we don't have too many interventions
    return list(dict.fromkeys(interventions))[:8]  # Remove duplicates and limit to 8

def save_to_js(data):
    """Save the processed data to a JavaScript file for the dashboard."""
    output_file = os.path.join('dashboard', 'src', 'data', 'patientData.js')
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert the data to JSON with proper formatting
    json_str = json.dumps(data, indent=2, default=str)
    
    # Write to the JavaScript file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('// Auto-generated by process_patient_data.py\n')
        f.write('// Last updated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n')
        f.write('export const patientData = ')
        f.write(json_str)
        f.write(';')
    
    print(f"Successfully saved data for {len(data)} patients to {output_file}")

def main():
    """Main function to process data and generate output."""
    try:
        # Load and merge data
        data = load_and_merge_data()
        
        if not data:
            raise ValueError("No data loaded. Please check your data files.")
        
        # Calculate risk scores
        patients = calculate_risk_scores(data)
        
        # Format for dashboard
        formatted_data = format_for_dashboard(patients)
        
        # Save to JavaScript file
        save_to_js(formatted_data)
        
        print("\nData processing completed successfully!")
        print(f"Processed {len(formatted_data)} patients.")
        
        # Print summary statistics
        if formatted_data:
            risk_counts = {}
            for patient in formatted_data:
                risk = patient.get('risk_category', 'unknown')
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            print("\nRisk Category Distribution:")
            for risk, count in sorted(risk_counts.items()):
                print(f"- {risk.capitalize()}: {count} patients")
        
    except Exception as e:
        print(f"\nError processing data: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
