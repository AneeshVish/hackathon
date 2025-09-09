// Enhanced patient data with more realistic details and risk factors
export const patientDataEnhanced = [
  {
    "patient_id": "101",
    "mrn": "M1001",
    "name": "John Smith",
    "first_name": "John",
    "last_name": "Smith",
    "age": 68,
    "gender": "M",
    "risk_score": 0.82,
    "risk_category": "high",
    "last_updated": new Date().toISOString(),
    "vital_signs": {
      "heart_rate": 92,
      "systolic_bp": 148,
      "diastolic_bp": 92,
      "weight": 94,
      "height": 175,
      "bmi": 30.7,
      "oxygen_saturation": 95,
      "temperature": 36.8
    },
    "medical_history": [
      "Hypertension",
      "Type 2 Diabetes",
      "Hyperlipidemia",
      "Coronary Artery Disease"
    ],
    "medications": [
      "Metformin 1000mg BID",
      "Lisinopril 20mg daily",
      "Atorvastatin 40mg nightly",
      "Aspirin 81mg daily"
    ],
    "key_risk_factors": [
      "Uncontrolled hypertension",
      "HbA1c > 9%",
      "LDL > 130 mg/dL",
      "History of CAD"
    ],
    "interventions": [
      "Cardiology referral",
      "Diabetes education",
      "Strict BP monitoring",
      "Lifestyle modification counseling"
    ],
    "lab_results": {
      "hba1c": 9.2,
      "ldl": 145,
      "hdl": 38,
      "triglycerides": 210,
      "creatinine": 1.2,
      "egfr": 68
    },
    "recent_encounters": [
      {
        "date": new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        "type": "Primary Care",
        "provider": "Dr. Sarah Johnson",
        "summary": "Follow-up for hypertension and diabetes management. BP elevated. Medication adherence concerns noted."
      }
    ],
    "contact": {
      "phone": "(555) 123-4567",
      "email": "john.smith@example.com",
      "address": "123 Maple Street, Boston, MA 02115",
      "emergency_contact": {
        "name": "Mary Smith",
        "relationship": "Spouse",
        "phone": "(555) 123-4568"
      }
    },
    "insurance": {
      "provider": "Blue Cross Blue Shield",
      "policy_number": "BCBS12345678",
      "group_number": "GRP98765"
    },
    "social_determinants": {
      "housing_status": "Stable",
      "food_security": "Food secure",
      "transportation": "Owns car",
      "social_support": "Lives with spouse"
    },
    "risk_prediction": {
      "readmission_risk_30d": 0.45,
      "ed_visit_risk_30d": 0.32,
      "mortality_risk_1y": 0.28,
      "key_drivers": [
        "Uncontrolled diabetes",
        "Multiple comorbidities",
        "Recent medication non-adherence"
      ]
    }
  },
  // More patients with similar structure...
];

export default patientDataEnhanced;
