// Auto-generated patient data with risk predictions
// Last updated: 2025-09-10 01:55:00

// Risk categories
const RISK_CATEGORIES = {
  HIGH: 'high',
  MEDIUM: 'medium',
  LOW: 'low'
};

// Generate random vitals based on risk category
const generateVitals = (riskCategory) => {
  switch (riskCategory) {
    case RISK_CATEGORIES.HIGH:
      return {
        heart_rate: Math.floor(Math.random() * 30) + 90, // 90-120
        systolic_bp: Math.floor(Math.random() * 40) + 140, // 140-180
        diastolic_bp: Math.floor(Math.random() * 20) + 85, // 85-105
        temperature: (Math.random() * 2) + 37.2, // 37.2-39.2
        respiratory_rate: Math.floor(Math.random() * 10) + 20, // 20-30
        spo2: Math.floor(Math.random() * 8) + 85, // 85-93
        weight: (Math.random() * 50) + 70, // 70-120 kg
        weight_trend: (Math.random() * 2.5) + 1.5 // 1.5-4.0 kg
      };
    case RISK_CATEGORIES.MEDIUM:
      return {
        heart_rate: Math.floor(Math.random() * 20) + 80, // 80-100
        systolic_bp: Math.floor(Math.random() * 30) + 125, // 125-155
        diastolic_bp: Math.floor(Math.random() * 15) + 75, // 75-90
        temperature: (Math.random() * 1.5) + 36.8, // 36.8-38.3
        respiratory_rate: Math.floor(Math.random() * 10) + 15, // 15-25
        spo2: Math.floor(Math.random() * 10) + 88, // 88-98
        weight: (Math.random() * 45) + 60, // 60-105 kg
        weight_trend: (Math.random() * 1.5) + 0.5 // 0.5-2.0 kg
      };
    default: // LOW
      return {
        heart_rate: Math.floor(Math.random() * 25) + 60, // 60-85
        systolic_bp: Math.floor(Math.random() * 25) + 110, // 110-135
        diastolic_bp: Math.floor(Math.random() * 15) + 70, // 70-85
        temperature: (Math.random() * 1.2) + 36.5, // 36.5-37.7
        respiratory_rate: Math.floor(Math.random() * 8) + 12, // 12-20
        spo2: Math.floor(Math.random() * 5) + 95, // 95-100
        weight: (Math.random() * 40) + 55, // 55-95 kg
        weight_trend: (Math.random() * 1.5) - 0.5 // -0.5 to 1.0 kg
      };
  }
};

// Generate interventions based on risk
const getInterventions = (riskCategory) => {
  const baseInterventions = [
    'Regular monitoring',
    'Medication adherence check',
    'Lifestyle counseling'
  ];

  if (riskCategory === RISK_CATEGORIES.HIGH) {
    return [
      'Immediate clinical assessment',
      'Consider hospitalization',
      'Intensify monitoring',
      'Review all medications',
      ...baseInterventions
    ];
  } else if (riskCategory === RISK_CATEGORIES.MEDIUM) {
    return [
      'Schedule follow-up within 1 week',
      'Consider medication adjustment',
      'Increase monitoring frequency',
      ...baseInterventions
    ];
  } else {
    return [
      'Routine follow-up in 1 month',
      'Continue current treatment',
      ...baseInterventions
    ];
  }
};

// Generate risk factors based on vitals
const getRiskFactors = (vitals) => {
  const factors = [];
  
  if (vitals.heart_rate > 100) factors.push('Tachycardia');
  if (vitals.systolic_bp > 140) factors.push('Hypertension');
  if (vitals.temperature > 38) factors.push('Fever');
  if (vitals.respiratory_rate > 20) factors.push('Tachypnea');
  if (vitals.spo2 < 92) factors.push('Hypoxemia');
  if (vitals.weight_trend > 2) factors.push('Significant weight gain');
  else if (vitals.weight_trend > 0.5) factors.push('Weight gain');
  
  // Add some random common factors if none found
  if (factors.length === 0) {
    const commonFactors = [
      'Mild hypertension',
      'Elevated heart rate',
      'Recent weight fluctuation',
      'Mild respiratory distress',
      'Decreased activity level'
    ];
    factors.push(...commonFactors.slice(0, 2));
  }
  
  return factors;
};

// Generate patient data
export const patientData = [
  {
    "patient_id": "101",
    "mrn": "101",
    "name": "Michael Davis",
    "first_name": "Michael",
    "last_name": "Davis",
    "age": 46,
    "gender": "M",
    "risk_score": 0.7,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 99,
      "systolic_bp": 132,
      "diastolic_bp": 83,
      "temperature": 37.5,
      "respiratory_rate": 21,
      "spo2": 93,
      "weight": 97.7,
      "weight_trend": 1.7
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "Chronic hypertension",
      "Blood sugar instability"
    ],
    "last_visit": "2025-08-12",
    "next_appointment": "2025-09-16"
  },
  {
    "patient_id": "102",
    "mrn": "102",
    "name": "Jane Miller",
    "first_name": "Jane",
    "last_name": "Miller",
    "age": 73,
    "gender": "M",
    "risk_score": 0.74,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 110,
      "systolic_bp": 163,
      "diastolic_bp": 104,
      "temperature": 37.7,
      "respiratory_rate": 23,
      "spo2": 90,
      "weight": 78.8,
      "weight_trend": 3.3
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Decreased respiratory rate",
      "History of heart failure",
      "Weight instability"
    ],
    "last_visit": "2025-08-15",
    "next_appointment": "2025-09-17"
  },
  {
    "patient_id": "103",
    "mrn": "103",
    "name": "Thomas Hernandez",
    "first_name": "Thomas",
    "last_name": "Hernandez",
    "age": 60,
    "gender": "F",
    "risk_score": 0.21,
    "risk_category": "low",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 78,
      "systolic_bp": 132,
      "diastolic_bp": 84,
      "temperature": 36.5,
      "respiratory_rate": 12,
      "spo2": 97,
      "weight": 74.9,
      "weight_trend": 0.4
    },
    "interventions": [
      "Continue current regimen",
      "Routine follow-up in 1 month"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "Chronic COPD",
      "Weight elevation"
    ],
    "last_visit": "2025-08-27",
    "next_appointment": "2025-09-10"
  },
  {
    "patient_id": "104",
    "mrn": "104",
    "name": "Susan Williams",
    "first_name": "Susan",
    "last_name": "Williams",
    "age": 51,
    "gender": "F",
    "risk_score": 0.44,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 100,
      "systolic_bp": 137,
      "diastolic_bp": 90,
      "temperature": 37.2,
      "respiratory_rate": 18,
      "spo2": 92,
      "weight": 94.3,
      "weight_trend": 1.8
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "Recent diabetes",
      "Weight fluctuation"
    ],
    "last_visit": "2025-09-06",
    "next_appointment": "2025-09-18"
  },
  {
    "patient_id": "105",
    "mrn": "105",
    "name": "Sarah Taylor",
    "first_name": "Sarah",
    "last_name": "Taylor",
    "age": 56,
    "gender": "M",
    "risk_score": 0.9,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 118,
      "systolic_bp": 156,
      "diastolic_bp": 108,
      "temperature": 38.8,
      "respiratory_rate": 22,
      "spo2": 90,
      "weight": 114.0,
      "weight_trend": 2.2
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "History of COPD",
      "Weight fluctuation"
    ],
    "last_visit": "2025-08-23",
    "next_appointment": "2025-09-20"
  },
  {
    "patient_id": "106",
    "mrn": "106",
    "name": "John Rodriguez",
    "first_name": "John",
    "last_name": "Rodriguez",
    "age": 54,
    "gender": "F",
    "risk_score": 0.19,
    "risk_category": "low",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 83,
      "systolic_bp": 125,
      "diastolic_bp": 75,
      "temperature": 36.8,
      "respiratory_rate": 16,
      "spo2": 100,
      "weight": 82.4,
      "weight_trend": 0.5
    },
    "interventions": [
      "Continue current regimen",
      "Routine follow-up in 1 month"
    ],
    "key_risk_factors": [
      "Decreased blood pressure",
      "History of COPD",
      "Weight instability"
    ],
    "last_visit": "2025-08-11",
    "next_appointment": "2025-09-21"
  },
  {
    "patient_id": "107",
    "mrn": "107",
    "name": "Charles Garcia",
    "first_name": "Charles",
    "last_name": "Garcia",
    "age": 69,
    "gender": "M",
    "risk_score": 0.5,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 99,
      "systolic_bp": 149,
      "diastolic_bp": 80,
      "temperature": 37.1,
      "respiratory_rate": 17,
      "spo2": 93,
      "weight": 70.6,
      "weight_trend": 0.7
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "Recent diabetes",
      "Blood sugar instability"
    ],
    "last_visit": "2025-08-10",
    "next_appointment": "2025-09-21"
  },
  {
    "patient_id": "108",
    "mrn": "108",
    "name": "Charles Miller",
    "first_name": "Charles",
    "last_name": "Miller",
    "age": 65,
    "gender": "F",
    "risk_score": 0.49,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 95,
      "systolic_bp": 147,
      "diastolic_bp": 90,
      "temperature": 37.3,
      "respiratory_rate": 18,
      "spo2": 96,
      "weight": 99.4,
      "weight_trend": 0.7
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "Recent diabetes",
      "Cholesterol instability"
    ],
    "last_visit": "2025-08-15",
    "next_appointment": "2025-09-16"
  },
  {
    "patient_id": "109",
    "mrn": "109",
    "name": "William Taylor",
    "first_name": "William",
    "last_name": "Taylor",
    "age": 57,
    "gender": "M",
    "risk_score": 0.74,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 94,
      "systolic_bp": 150,
      "diastolic_bp": 106,
      "temperature": 37.7,
      "respiratory_rate": 23,
      "spo2": 88,
      "weight": 80.5,
      "weight_trend": 1.6
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Elevated blood pressure",
      "History of COPD",
      "Blood sugar fluctuation"
    ],
    "last_visit": "2025-08-27",
    "next_appointment": "2025-09-15"
  },
  {
    "patient_id": "110",
    "mrn": "110",
    "name": "Elizabeth Davis",
    "first_name": "Elizabeth",
    "last_name": "Davis",
    "age": 53,
    "gender": "F",
    "risk_score": 0.94,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 92,
      "systolic_bp": 156,
      "diastolic_bp": 97,
      "temperature": 38.4,
      "respiratory_rate": 24,
      "spo2": 85,
      "weight": 85.7,
      "weight_trend": 3.7
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Decreased respiratory rate",
      "History of hypertension",
      "Weight elevation"
    ],
    "last_visit": "2025-08-16",
    "next_appointment": "2025-09-12"
  },
  {
    "patient_id": "111",
    "mrn": "111",
    "name": "Christopher Williams",
    "first_name": "Christopher",
    "last_name": "Williams",
    "age": 79,
    "gender": "M",
    "risk_score": 0.34,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 94,
      "systolic_bp": 143,
      "diastolic_bp": 84,
      "temperature": 37.5,
      "respiratory_rate": 22,
      "spo2": 96,
      "weight": 105.0,
      "weight_trend": 1.1
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Decreased heart rate",
      "History of heart failure",
      "Cholesterol elevation"
    ],
    "last_visit": "2025-09-06",
    "next_appointment": "2025-09-13"
  },
  {
    "patient_id": "112",
    "mrn": "112",
    "name": "Charles Anderson",
    "first_name": "Charles",
    "last_name": "Anderson",
    "age": 58,
    "gender": "M",
    "risk_score": 0.82,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 93,
      "systolic_bp": 168,
      "diastolic_bp": 94,
      "temperature": 37.8,
      "respiratory_rate": 30,
      "spo2": 88,
      "weight": 89.8,
      "weight_trend": 3.2
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Decreased heart rate",
      "Chronic diabetes",
      "Blood sugar instability"
    ],
    "last_visit": "2025-08-21",
    "next_appointment": "2025-09-12"
  },
  {
    "patient_id": "113",
    "mrn": "113",
    "name": "Joseph Wilson",
    "first_name": "Joseph",
    "last_name": "Wilson",
    "age": 64,
    "gender": "M",
    "risk_score": 0.93,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 94,
      "systolic_bp": 168,
      "diastolic_bp": 89,
      "temperature": 38.7,
      "respiratory_rate": 23,
      "spo2": 91,
      "weight": 83.3,
      "weight_trend": 3.9
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Elevated respiratory rate",
      "History of heart failure",
      "Weight instability"
    ],
    "last_visit": "2025-08-14",
    "next_appointment": "2025-09-23"
  },
  {
    "patient_id": "114",
    "mrn": "114",
    "name": "John Johnson",
    "first_name": "John",
    "last_name": "Johnson",
    "age": 67,
    "gender": "M",
    "risk_score": 0.72,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 103,
      "systolic_bp": 170,
      "diastolic_bp": 109,
      "temperature": 38.9,
      "respiratory_rate": 30,
      "spo2": 89,
      "weight": 89.0,
      "weight_trend": 2.3
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Decreased blood pressure",
      "Recent hypertension",
      "Weight instability"
    ],
    "last_visit": "2025-08-18",
    "next_appointment": "2025-09-15"
  },
  {
    "patient_id": "115",
    "mrn": "115",
    "name": "Susan Martinez",
    "first_name": "Susan",
    "last_name": "Martinez",
    "age": 88,
    "gender": "M",
    "risk_score": 0.7,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 99,
      "systolic_bp": 137,
      "diastolic_bp": 85,
      "temperature": 37.4,
      "respiratory_rate": 21,
      "spo2": 95,
      "weight": 83.8,
      "weight_trend": 1.4
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "History of heart failure",
      "Blood sugar elevation"
    ],
    "last_visit": "2025-08-18",
    "next_appointment": "2025-09-10"
  },
  {
    "patient_id": "116",
    "mrn": "116",
    "name": "Linda Davis",
    "first_name": "Linda",
    "last_name": "Davis",
    "age": 79,
    "gender": "F",
    "risk_score": 0.9,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 110,
      "systolic_bp": 150,
      "diastolic_bp": 93,
      "temperature": 38.2,
      "respiratory_rate": 24,
      "spo2": 86,
      "weight": 94.4,
      "weight_trend": 4.0
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Irregular blood pressure",
      "History of COPD",
      "Weight elevation"
    ],
    "last_visit": "2025-08-13",
    "next_appointment": "2025-09-17"
  },
  {
    "patient_id": "117",
    "mrn": "117",
    "name": "Thomas Miller",
    "first_name": "Thomas",
    "last_name": "Miller",
    "age": 56,
    "gender": "M",
    "risk_score": 0.7,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 98,
      "systolic_bp": 140,
      "diastolic_bp": 84,
      "temperature": 37.4,
      "respiratory_rate": 21,
      "spo2": 96,
      "weight": 90.5,
      "weight_trend": 1.3
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular blood pressure",
      "Recent heart failure",
      "Cholesterol instability"
    ],
    "last_visit": "2025-08-13",
    "next_appointment": "2025-09-12"
  },
  {
    "patient_id": "118",
    "mrn": "118",
    "name": "Robert Taylor",
    "first_name": "Robert",
    "last_name": "Taylor",
    "age": 45,
    "gender": "F",
    "risk_score": 0.14,
    "risk_category": "low",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 69,
      "systolic_bp": 134,
      "diastolic_bp": 77,
      "temperature": 37.0,
      "respiratory_rate": 14,
      "spo2": 100,
      "weight": 99.1,
      "weight_trend": 0.7
    },
    "interventions": [
      "Continue current regimen",
      "Routine follow-up in 1 month"
    ],
    "key_risk_factors": [
      "Elevated heart rate",
      "History of hypertension",
      "Cholesterol fluctuation"
    ],
    "last_visit": "2025-08-12",
    "next_appointment": "2025-09-11"
  },
  {
    "patient_id": "119",
    "mrn": "119",
    "name": "Susan Garcia",
    "first_name": "Susan",
    "last_name": "Garcia",
    "age": 48,
    "gender": "M",
    "risk_score": 0.12,
    "risk_category": "low",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 81,
      "systolic_bp": 118,
      "diastolic_bp": 74,
      "temperature": 36.9,
      "respiratory_rate": 14,
      "spo2": 100,
      "weight": 62.8,
      "weight_trend": -0.1
    },
    "interventions": [
      "Continue current regimen",
      "Routine follow-up in 1 month"
    ],
    "key_risk_factors": [
      "Irregular blood pressure",
      "Chronic diabetes",
      "Blood sugar elevation"
    ],
    "last_visit": "2025-08-20",
    "next_appointment": "2025-09-11"
  },
  {
    "patient_id": "120",
    "mrn": "120",
    "name": "Susan Anderson",
    "first_name": "Susan",
    "last_name": "Anderson",
    "age": 85,
    "gender": "F",
    "risk_score": 0.33,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 88,
      "systolic_bp": 131,
      "diastolic_bp": 93,
      "temperature": 37.3,
      "respiratory_rate": 19,
      "spo2": 92,
      "weight": 77.6,
      "weight_trend": 1.5
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Decreased blood pressure",
      "Chronic heart failure",
      "Blood sugar fluctuation"
    ],
    "last_visit": "2025-08-17",
    "next_appointment": "2025-09-12"
  },
  {
    "patient_id": "121",
    "mrn": "121",
    "name": "Lisa Wilson",
    "first_name": "Lisa",
    "last_name": "Wilson",
    "age": 61,
    "gender": "F",
    "risk_score": 0.64,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.043310",
    "vital_signs": {
      "heart_rate": 88,
      "systolic_bp": 137,
      "diastolic_bp": 85,
      "temperature": 36.8,
      "respiratory_rate": 19,
      "spo2": 93,
      "weight": 109.5,
      "weight_trend": 2.0
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "Chronic heart failure",
      "Cholesterol elevation"
    ],
    "last_visit": "2025-08-24",
    "next_appointment": "2025-09-22"
  },
  {
    "patient_id": "122",
    "mrn": "122",
    "name": "Karen Williams",
    "first_name": "Karen",
    "last_name": "Williams",
    "age": 72,
    "gender": "M",
    "risk_score": 0.51,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 91,
      "systolic_bp": 135,
      "diastolic_bp": 89,
      "temperature": 37.3,
      "respiratory_rate": 18,
      "spo2": 94,
      "weight": 74.5,
      "weight_trend": 1.7
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "Recent hypertension",
      "Weight instability"
    ],
    "last_visit": "2025-08-21",
    "next_appointment": "2025-09-12"
  },
  {
    "patient_id": "123",
    "mrn": "123",
    "name": "Nancy Davis",
    "first_name": "Nancy",
    "last_name": "Davis",
    "age": 81,
    "gender": "F",
    "risk_score": 0.5,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 91,
      "systolic_bp": 143,
      "diastolic_bp": 91,
      "temperature": 37.4,
      "respiratory_rate": 19,
      "spo2": 95,
      "weight": 107.1,
      "weight_trend": 1.6
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Decreased heart rate",
      "Recent heart failure",
      "Weight instability"
    ],
    "last_visit": "2025-09-05",
    "next_appointment": "2025-09-18"
  },
  {
    "patient_id": "124",
    "mrn": "124",
    "name": "Jessica Hernandez",
    "first_name": "Jessica",
    "last_name": "Hernandez",
    "age": 50,
    "gender": "F",
    "risk_score": 0.93,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 93,
      "systolic_bp": 164,
      "diastolic_bp": 108,
      "temperature": 37.5,
      "respiratory_rate": 24,
      "spo2": 89,
      "weight": 77.3,
      "weight_trend": 3.1
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Elevated heart rate",
      "Chronic hypertension",
      "Weight instability"
    ],
    "last_visit": "2025-08-31",
    "next_appointment": "2025-09-23"
  },
  {
    "patient_id": "125",
    "mrn": "125",
    "name": "Patricia Wilson",
    "first_name": "Patricia",
    "last_name": "Wilson",
    "age": 83,
    "gender": "F",
    "risk_score": 0.82,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 93,
      "systolic_bp": 159,
      "diastolic_bp": 98,
      "temperature": 38.3,
      "respiratory_rate": 25,
      "spo2": 85,
      "weight": 95.7,
      "weight_trend": 2.3
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Decreased heart rate",
      "History of diabetes",
      "Blood sugar fluctuation"
    ],
    "last_visit": "2025-08-27",
    "next_appointment": "2025-09-15"
  },
  {
    "patient_id": "126",
    "mrn": "126",
    "name": "Jessica Taylor",
    "first_name": "Jessica",
    "last_name": "Taylor",
    "age": 70,
    "gender": "M",
    "risk_score": 0.65,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 81,
      "systolic_bp": 133,
      "diastolic_bp": 87,
      "temperature": 36.8,
      "respiratory_rate": 17,
      "spo2": 95,
      "weight": 95.9,
      "weight_trend": 1.4
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Elevated heart rate",
      "History of COPD",
      "Weight fluctuation"
    ],
    "last_visit": "2025-09-01",
    "next_appointment": "2025-09-21"
  },
  {
    "patient_id": "127",
    "mrn": "127",
    "name": "Sarah Rodriguez",
    "first_name": "Sarah",
    "last_name": "Rodriguez",
    "age": 72,
    "gender": "M",
    "risk_score": 0.93,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 111,
      "systolic_bp": 140,
      "diastolic_bp": 94,
      "temperature": 38.0,
      "respiratory_rate": 21,
      "spo2": 88,
      "weight": 112.6,
      "weight_trend": 1.8
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Elevated heart rate",
      "Recent COPD",
      "Weight fluctuation"
    ],
    "last_visit": "2025-08-17",
    "next_appointment": "2025-09-19"
  },
  {
    "patient_id": "128",
    "mrn": "128",
    "name": "Thomas Davis",
    "first_name": "Thomas",
    "last_name": "Davis",
    "age": 72,
    "gender": "F",
    "risk_score": 0.73,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 97,
      "systolic_bp": 146,
      "diastolic_bp": 90,
      "temperature": 38.8,
      "respiratory_rate": 27,
      "spo2": 87,
      "weight": 105.6,
      "weight_trend": 3.4
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Irregular blood pressure",
      "Recent heart failure",
      "Cholesterol instability"
    ],
    "last_visit": "2025-08-24",
    "next_appointment": "2025-09-18"
  },
  {
    "patient_id": "129",
    "mrn": "129",
    "name": "Richard Taylor",
    "first_name": "Richard",
    "last_name": "Taylor",
    "age": 47,
    "gender": "M",
    "risk_score": 0.44,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 93,
      "systolic_bp": 149,
      "diastolic_bp": 93,
      "temperature": 37.3,
      "respiratory_rate": 18,
      "spo2": 93,
      "weight": 100.7,
      "weight_trend": 1.7
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Decreased respiratory rate",
      "History of hypertension",
      "Cholesterol instability"
    ],
    "last_visit": "2025-08-31",
    "next_appointment": "2025-09-13"
  },
  {
    "patient_id": "130",
    "mrn": "130",
    "name": "James Garcia",
    "first_name": "James",
    "last_name": "Garcia",
    "age": 54,
    "gender": "F",
    "risk_score": 0.87,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 95,
      "systolic_bp": 164,
      "diastolic_bp": 91,
      "temperature": 37.6,
      "respiratory_rate": 30,
      "spo2": 92,
      "weight": 108.0,
      "weight_trend": 3.4
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Irregular heart rate",
      "Recent COPD",
      "Blood sugar instability"
    ],
    "last_visit": "2025-08-30",
    "next_appointment": "2025-09-21"
  },
  {
    "patient_id": "131",
    "mrn": "131",
    "name": "Joseph Williams",
    "first_name": "Joseph",
    "last_name": "Williams",
    "age": 85,
    "gender": "F",
    "risk_score": 0.95,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 103,
      "systolic_bp": 167,
      "diastolic_bp": 103,
      "temperature": 37.8,
      "respiratory_rate": 27,
      "spo2": 87,
      "weight": 110.2,
      "weight_trend": 3.5
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Elevated respiratory rate",
      "Chronic diabetes",
      "Weight elevation"
    ],
    "last_visit": "2025-08-28",
    "next_appointment": "2025-09-14"
  },
  {
    "patient_id": "132",
    "mrn": "132",
    "name": "Mary Miller",
    "first_name": "Mary",
    "last_name": "Miller",
    "age": 89,
    "gender": "F",
    "risk_score": 0.88,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 93,
      "systolic_bp": 167,
      "diastolic_bp": 107,
      "temperature": 38.3,
      "respiratory_rate": 22,
      "spo2": 85,
      "weight": 84.7,
      "weight_trend": 3.1
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Elevated blood pressure",
      "Chronic heart failure",
      "Blood sugar fluctuation"
    ],
    "last_visit": "2025-08-19",
    "next_appointment": "2025-09-23"
  },
  {
    "patient_id": "133",
    "mrn": "133",
    "name": "Patricia Jones",
    "first_name": "Patricia",
    "last_name": "Jones",
    "age": 75,
    "gender": "M",
    "risk_score": 0.11,
    "risk_category": "low",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 67,
      "systolic_bp": 130,
      "diastolic_bp": 77,
      "temperature": 36.9,
      "respiratory_rate": 17,
      "spo2": 96,
      "weight": 93.0,
      "weight_trend": 0.8
    },
    "interventions": [
      "Continue current regimen",
      "Routine follow-up in 1 month"
    ],
    "key_risk_factors": [
      "Elevated heart rate",
      "Chronic diabetes",
      "Blood sugar elevation"
    ],
    "last_visit": "2025-08-12",
    "next_appointment": "2025-09-19"
  },
  {
    "patient_id": "134",
    "mrn": "134",
    "name": "Thomas Smith",
    "first_name": "Thomas",
    "last_name": "Smith",
    "age": 62,
    "gender": "M",
    "risk_score": 0.16,
    "risk_category": "low",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 67,
      "systolic_bp": 132,
      "diastolic_bp": 83,
      "temperature": 37.2,
      "respiratory_rate": 15,
      "spo2": 98,
      "weight": 61.4,
      "weight_trend": 1.0
    },
    "interventions": [
      "Continue current regimen",
      "Routine follow-up in 1 month"
    ],
    "key_risk_factors": [
      "Decreased heart rate",
      "History of COPD",
      "Weight fluctuation"
    ],
    "last_visit": "2025-08-11",
    "next_appointment": "2025-09-11"
  },
  {
    "patient_id": "135",
    "mrn": "135",
    "name": "Richard Thomas",
    "first_name": "Richard",
    "last_name": "Thomas",
    "age": 67,
    "gender": "F",
    "risk_score": 0.31,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 95,
      "systolic_bp": 139,
      "diastolic_bp": 91,
      "temperature": 37.2,
      "respiratory_rate": 19,
      "spo2": 92,
      "weight": 81.8,
      "weight_trend": 1.5
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Elevated heart rate",
      "Chronic COPD",
      "Weight instability"
    ],
    "last_visit": "2025-08-18",
    "next_appointment": "2025-09-18"
  },
  {
    "patient_id": "136",
    "mrn": "136",
    "name": "Jennifer Davis",
    "first_name": "Jennifer",
    "last_name": "Davis",
    "age": 69,
    "gender": "M",
    "risk_score": 0.51,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 96,
      "systolic_bp": 141,
      "diastolic_bp": 86,
      "temperature": 37.0,
      "respiratory_rate": 21,
      "spo2": 92,
      "weight": 67.4,
      "weight_trend": 0.5
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "Recent hypertension",
      "Cholesterol fluctuation"
    ],
    "last_visit": "2025-09-07",
    "next_appointment": "2025-09-12"
  },
  {
    "patient_id": "137",
    "mrn": "137",
    "name": "Susan Jones",
    "first_name": "Susan",
    "last_name": "Jones",
    "age": 65,
    "gender": "M",
    "risk_score": 0.85,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 91,
      "systolic_bp": 152,
      "diastolic_bp": 95,
      "temperature": 38.9,
      "respiratory_rate": 24,
      "spo2": 90,
      "weight": 75.2,
      "weight_trend": 1.6
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Elevated blood pressure",
      "Recent heart failure",
      "Blood sugar elevation"
    ],
    "last_visit": "2025-08-25",
    "next_appointment": "2025-09-10"
  },
  {
    "patient_id": "138",
    "mrn": "138",
    "name": "Robert Rodriguez",
    "first_name": "Robert",
    "last_name": "Rodriguez",
    "age": 55,
    "gender": "M",
    "risk_score": 0.66,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 82,
      "systolic_bp": 143,
      "diastolic_bp": 94,
      "temperature": 36.9,
      "respiratory_rate": 22,
      "spo2": 94,
      "weight": 104.9,
      "weight_trend": 0.7
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Elevated blood pressure",
      "History of hypertension",
      "Blood sugar instability"
    ],
    "last_visit": "2025-08-26",
    "next_appointment": "2025-09-11"
  },
  {
    "patient_id": "139",
    "mrn": "139",
    "name": "Susan Johnson",
    "first_name": "Susan",
    "last_name": "Johnson",
    "age": 47,
    "gender": "F",
    "risk_score": 0.15,
    "risk_category": "low",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 84,
      "systolic_bp": 112,
      "diastolic_bp": 74,
      "temperature": 36.7,
      "respiratory_rate": 14,
      "spo2": 98,
      "weight": 94.3,
      "weight_trend": -0.2
    },
    "interventions": [
      "Continue current regimen",
      "Routine follow-up in 1 month"
    ],
    "key_risk_factors": [
      "Elevated blood pressure",
      "History of diabetes",
      "Weight elevation"
    ],
    "last_visit": "2025-08-11",
    "next_appointment": "2025-09-19"
  },
  {
    "patient_id": "140",
    "mrn": "140",
    "name": "Daniel Taylor",
    "first_name": "Daniel",
    "last_name": "Taylor",
    "age": 66,
    "gender": "F",
    "risk_score": 0.88,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 117,
      "systolic_bp": 155,
      "diastolic_bp": 96,
      "temperature": 38.9,
      "respiratory_rate": 22,
      "spo2": 92,
      "weight": 103.3,
      "weight_trend": 3.7
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Decreased heart rate",
      "Recent COPD",
      "Cholesterol instability"
    ],
    "last_visit": "2025-08-17",
    "next_appointment": "2025-09-12"
  },
  {
    "patient_id": "141",
    "mrn": "141",
    "name": "Charles Johnson",
    "first_name": "Charles",
    "last_name": "Johnson",
    "age": 76,
    "gender": "M",
    "risk_score": 0.57,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 90,
      "systolic_bp": 132,
      "diastolic_bp": 93,
      "temperature": 37.3,
      "respiratory_rate": 21,
      "spo2": 96,
      "weight": 84.9,
      "weight_trend": 0.6
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Elevated heart rate",
      "Chronic COPD",
      "Weight instability"
    ],
    "last_visit": "2025-08-20",
    "next_appointment": "2025-09-20"
  },
  {
    "patient_id": "142",
    "mrn": "142",
    "name": "Lisa Martinez",
    "first_name": "Lisa",
    "last_name": "Martinez",
    "age": 67,
    "gender": "M",
    "risk_score": 0.63,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 88,
      "systolic_bp": 150,
      "diastolic_bp": 91,
      "temperature": 36.8,
      "respiratory_rate": 22,
      "spo2": 94,
      "weight": 77.9,
      "weight_trend": 1.0
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "History of COPD",
      "Cholesterol fluctuation"
    ],
    "last_visit": "2025-08-27",
    "next_appointment": "2025-09-17"
  },
  {
    "patient_id": "143",
    "mrn": "143",
    "name": "Lisa Jones",
    "first_name": "Lisa",
    "last_name": "Jones",
    "age": 74,
    "gender": "F",
    "risk_score": 0.43,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 81,
      "systolic_bp": 148,
      "diastolic_bp": 91,
      "temperature": 36.8,
      "respiratory_rate": 19,
      "spo2": 92,
      "weight": 92.2,
      "weight_trend": 1.1
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Elevated respiratory rate",
      "Recent hypertension",
      "Cholesterol elevation"
    ],
    "last_visit": "2025-08-20",
    "next_appointment": "2025-09-11"
  },
  {
    "patient_id": "144",
    "mrn": "144",
    "name": "Jane Miller",
    "first_name": "Jane",
    "last_name": "Miller",
    "age": 89,
    "gender": "F",
    "risk_score": 0.93,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 100,
      "systolic_bp": 166,
      "diastolic_bp": 104,
      "temperature": 37.7,
      "respiratory_rate": 27,
      "spo2": 89,
      "weight": 100.3,
      "weight_trend": 2.0
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "History of COPD",
      "Weight elevation"
    ],
    "last_visit": "2025-08-21",
    "next_appointment": "2025-09-11"
  },
  {
    "patient_id": "145",
    "mrn": "145",
    "name": "Robert Miller",
    "first_name": "Robert",
    "last_name": "Miller",
    "age": 76,
    "gender": "F",
    "risk_score": 0.68,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 89,
      "systolic_bp": 149,
      "diastolic_bp": 90,
      "temperature": 37.2,
      "respiratory_rate": 17,
      "spo2": 94,
      "weight": 92.1,
      "weight_trend": 0.7
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular blood pressure",
      "Chronic diabetes",
      "Cholesterol instability"
    ],
    "last_visit": "2025-08-20",
    "next_appointment": "2025-09-20"
  },
  {
    "patient_id": "146",
    "mrn": "146",
    "name": "Robert Brown",
    "first_name": "Robert",
    "last_name": "Brown",
    "age": 84,
    "gender": "M",
    "risk_score": 0.67,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 96,
      "systolic_bp": 147,
      "diastolic_bp": 81,
      "temperature": 37.1,
      "respiratory_rate": 18,
      "spo2": 96,
      "weight": 86.9,
      "weight_trend": 1.3
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular heart rate",
      "Recent heart failure",
      "Cholesterol instability"
    ],
    "last_visit": "2025-09-02",
    "next_appointment": "2025-09-23"
  },
  {
    "patient_id": "147",
    "mrn": "147",
    "name": "Patricia Miller",
    "first_name": "Patricia",
    "last_name": "Miller",
    "age": 89,
    "gender": "M",
    "risk_score": 0.58,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 88,
      "systolic_bp": 132,
      "diastolic_bp": 80,
      "temperature": 37.5,
      "respiratory_rate": 22,
      "spo2": 93,
      "weight": 75.0,
      "weight_trend": 0.9
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular heart rate",
      "Recent COPD",
      "Cholesterol fluctuation"
    ],
    "last_visit": "2025-08-23",
    "next_appointment": "2025-09-22"
  },
  {
    "patient_id": "148",
    "mrn": "148",
    "name": "Susan Garcia",
    "first_name": "Susan",
    "last_name": "Garcia",
    "age": 53,
    "gender": "F",
    "risk_score": 0.3,
    "risk_category": "low",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 68,
      "systolic_bp": 130,
      "diastolic_bp": 70,
      "temperature": 36.9,
      "respiratory_rate": 17,
      "spo2": 98,
      "weight": 95.0,
      "weight_trend": 0.9
    },
    "interventions": [
      "Continue current regimen",
      "Routine follow-up in 1 month"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "History of diabetes",
      "Blood sugar instability"
    ],
    "last_visit": "2025-08-23",
    "next_appointment": "2025-09-12"
  },
  {
    "patient_id": "149",
    "mrn": "149",
    "name": "Karen Miller",
    "first_name": "Karen",
    "last_name": "Miller",
    "age": 83,
    "gender": "M",
    "risk_score": 0.32,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 82,
      "systolic_bp": 136,
      "diastolic_bp": 90,
      "temperature": 37.1,
      "respiratory_rate": 17,
      "spo2": 94,
      "weight": 82.9,
      "weight_trend": 1.4
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular heart rate",
      "History of COPD",
      "Blood sugar instability"
    ],
    "last_visit": "2025-08-10",
    "next_appointment": "2025-09-11"
  },
  {
    "patient_id": "150",
    "mrn": "150",
    "name": "Linda Davis",
    "first_name": "Linda",
    "last_name": "Davis",
    "age": 50,
    "gender": "M",
    "risk_score": 0.59,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.044319",
    "vital_signs": {
      "heart_rate": 89,
      "systolic_bp": 136,
      "diastolic_bp": 88,
      "temperature": 37.4,
      "respiratory_rate": 20,
      "spo2": 95,
      "weight": 83.3,
      "weight_trend": 1.2
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Decreased respiratory rate",
      "History of COPD",
      "Blood sugar instability"
    ],
    "last_visit": "2025-09-05",
    "next_appointment": "2025-09-10"
  },
  {
    "patient_id": "151",
    "mrn": "151",
    "name": "Charles Wilson",
    "first_name": "Charles",
    "last_name": "Wilson",
    "age": 88,
    "gender": "F",
    "risk_score": 0.41,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.045326",
    "vital_signs": {
      "heart_rate": 81,
      "systolic_bp": 135,
      "diastolic_bp": 95,
      "temperature": 36.9,
      "respiratory_rate": 22,
      "spo2": 92,
      "weight": 68.0,
      "weight_trend": 1.3
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Decreased heart rate",
      "Recent heart failure",
      "Blood sugar instability"
    ],
    "last_visit": "2025-08-17",
    "next_appointment": "2025-09-20"
  },
  {
    "patient_id": "152",
    "mrn": "152",
    "name": "Richard Johnson",
    "first_name": "Richard",
    "last_name": "Johnson",
    "age": 72,
    "gender": "M",
    "risk_score": 0.35,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.045326",
    "vital_signs": {
      "heart_rate": 100,
      "systolic_bp": 131,
      "diastolic_bp": 87,
      "temperature": 37.4,
      "respiratory_rate": 20,
      "spo2": 92,
      "weight": 93.4,
      "weight_trend": 0.8
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Elevated blood pressure",
      "Recent COPD",
      "Blood sugar fluctuation"
    ],
    "last_visit": "2025-08-15",
    "next_appointment": "2025-09-17"
  },
  {
    "patient_id": "153",
    "mrn": "153",
    "name": "David Johnson",
    "first_name": "David",
    "last_name": "Johnson",
    "age": 69,
    "gender": "M",
    "risk_score": 0.27,
    "risk_category": "low",
    "last_updated": "2025-09-09T20:27:49.045326",
    "vital_signs": {
      "heart_rate": 68,
      "systolic_bp": 117,
      "diastolic_bp": 79,
      "temperature": 36.7,
      "respiratory_rate": 17,
      "spo2": 100,
      "weight": 78.2,
      "weight_trend": 0.0
    },
    "interventions": [
      "Continue current regimen",
      "Routine follow-up in 1 month"
    ],
    "key_risk_factors": [
      "Irregular respiratory rate",
      "Chronic hypertension",
      "Weight fluctuation"
    ],
    "last_visit": "2025-08-13",
    "next_appointment": "2025-09-15"
  },
  {
    "patient_id": "154",
    "mrn": "154",
    "name": "Robert Thomas",
    "first_name": "Robert",
    "last_name": "Thomas",
    "age": 87,
    "gender": "M",
    "risk_score": 0.73,
    "risk_category": "high",
    "last_updated": "2025-09-09T20:27:49.045326",
    "vital_signs": {
      "heart_rate": 105,
      "systolic_bp": 158,
      "diastolic_bp": 103,
      "temperature": 37.5,
      "respiratory_rate": 29,
      "spo2": 85,
      "weight": 97.3,
      "weight_trend": 3.5
    },
    "interventions": [
      "Schedule urgent follow-up",
      "Consider diuretics adjustment",
      "Daily weight monitoring",
      "Supplemental oxygen as needed"
    ],
    "key_risk_factors": [
      "Decreased blood pressure",
      "History of hypertension",
      "Cholesterol fluctuation"
    ],
    "last_visit": "2025-09-07",
    "next_appointment": "2025-09-20"
  },
  {
    "patient_id": "155",
    "mrn": "155",
    "name": "Sarah Taylor",
    "first_name": "Sarah",
    "last_name": "Taylor",
    "age": 51,
    "gender": "F",
    "risk_score": 0.52,
    "risk_category": "medium",
    "last_updated": "2025-09-09T20:27:49.045326",
    "vital_signs": {
      "heart_rate": 83,
      "systolic_bp": 131,
      "diastolic_bp": 87,
      "temperature": 36.9,
      "respiratory_rate": 18,
      "spo2": 93,
      "weight": 79.4,
      "weight_trend": 1.2
    },
    "interventions": [
      "Increase diuretic dose",
      "Low-sodium diet education",
      "Bi-weekly weight checks"
    ],
    "key_risk_factors": [
      "Irregular heart rate",
      "History of diabetes",
      "Weight instability"
    ],
    "last_visit": "2025-08-23",
    "next_appointment": "2025-09-17"
  }
]