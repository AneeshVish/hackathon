import { patientData } from './patientData';

// Convert the real patient data to match the expected format of trainedModelData.js
export const getProcessedPatientData = () => {
  // Process patient data to match the expected format
  const processedData = patientData.map(patient => ({
    patient_id: patient.patient_id,
    mrn: patient.mrn,
    name: patient.name,
    age: patient.age,
    gender: patient.gender,
    risk_score: patient.risk_score,
    risk_bucket: patient.risk_category,
    prediction_confidence: 0.85, // Default confidence
    last_visit: new Date().toISOString().split('T')[0],
    next_appointment: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    last_prediction: 'Just now',
    contact: {
      phone: '555-0100',
      email: `${patient.first_name.toLowerCase()}.${patient.last_name.toLowerCase()}@example.com`,
      address: '123 Main St, Anytown, USA'
    },
    insurance: {
      provider: 'MediCare',
      policy_number: `POL-${patient.mrn}`,
      group_number: 'GRP-12345'
    },
    key_risk_factors: patient.key_risk_factors || [],
    vital_signs: patient.vital_signs || {},
    interventions: patient.interventions || [],
    medical_history: patient.medical_history || [],
    medications: patient.medications || []
  }));

  return processedData;
};

// Generate risk distribution based on the processed data
export const getProcessedRiskDistribution = (patients) => {
  const riskCounts = patients.reduce((acc, patient) => {
    const riskLevel = patient.risk_bucket?.toLowerCase() || 'low';
    acc[riskLevel] = (acc[riskLevel] || 0) + 1;
    return acc;
  }, { high: 0, medium: 0, low: 0 });

  return [
    { name: 'High', value: riskCounts.high || 0 },
    { name: 'Medium', value: riskCounts.medium || 0 },
    { name: 'Low', value: riskCounts.low || 0 }
  ];
};

// Generate patient statistics
export const getProcessedPatientStats = (patients) => {
  const totalPatients = patients.length;
  const riskDistribution = getProcessedRiskDistribution(patients);
  
  return {
    totalPatients,
    highRisk: riskDistribution[0].value,
    mediumRisk: riskDistribution[1].value,
    lowRisk: riskDistribution[2].value,
    avgRiskScore: (patients.reduce((sum, p) => sum + (p.risk_score || 0), 0) / totalPatients).toFixed(2)
  };
};
