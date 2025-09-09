import { faker } from '@faker-js/faker';

// Generate realistic patient data with risk factors and trends
const generatePatientData = (count = 50) => {
  const patients = [];
  const conditions = [
    'Hypertension', 'Type 2 Diabetes', 'Coronary Artery Disease', 'COPD',
    'Chronic Kidney Disease', 'Heart Failure', 'Atrial Fibrillation', 'Asthma'
  ];
  
  const medications = [
    'Metformin', 'Lisinopril', 'Atorvastatin', 'Metoprolol',
    'Sitagliptin', 'Losartan', 'Amlodipine', 'Sertraline'
  ];

  for (let i = 0; i < count; i++) {
    const gender = faker.person.sexType();
    const firstName = faker.person.firstName(gender);
    const lastName = faker.person.lastName();
    const age = faker.number.int({ min: 30, max: 95 });
    const hasDiabetes = Math.random() > 0.7;
    const hasHypertension = Math.random() > 0.6;
    
    // Generate risk score based on age and conditions
    let riskScore = 0.1;
    if (age > 65) riskScore += 0.3;
    if (age > 75) riskScore += 0.2;
    if (hasDiabetes) riskScore += 0.3;
    if (hasHypertension) riskScore += 0.2;
    
    // Add some randomness
    riskScore += (Math.random() * 0.2) - 0.1;
    riskScore = Math.max(0.1, Math.min(0.99, riskScore));
    
    // Generate risk trend (last 6 months)
    const riskTrend = Array.from({ length: 6 }, (_, i) => ({
      date: new Date(Date.now() - (6 - i) * 30 * 24 * 60 * 60 * 1000).toISOString(),
      value: Math.max(0.1, Math.min(0.99, riskScore + (Math.random() * 0.2 - 0.1)))
    }));
    
    // Generate key risk factors
    const riskFactors = [];
    if (age > 65) riskFactors.push('Age > 65');
    if (hasDiabetes) riskFactors.push('Uncontrolled Diabetes');
    if (hasHypertension) riskFactors.push('Hypertension');
    if (Math.random() > 0.7) riskFactors.push('Recent Hospitalization');
    if (Math.random() > 0.8) riskFactors.push('Multiple Comorbidities');
    
    // Generate recommended actions
    const recommendedActions = [];
    if (riskScore > 0.7) {
      recommendedActions.push({
        id: faker.string.uuid(),
        priority: 'high',
        action: 'Schedule High-Risk Care Coordination',
        dueDate: faker.date.soon({ days: 7 }).toISOString()
      });
    }
    if (hasDiabetes && Math.random() > 0.5) {
      recommendedActions.push({
        id: faker.string.uuid(),
        priority: 'medium',
        action: 'Diabetes Management Review',
        dueDate: faker.date.soon({ days: 14 }).toISOString()
      });
    }
    
    patients.push({
      id: faker.string.uuid(),
      mrn: `MRN${1000 + i}`,
      name: `${firstName} ${lastName}`,
      firstName,
      lastName,
      age,
      gender: gender.charAt(0).toUpperCase() + gender.slice(1),
      riskScore,
      riskCategory: riskScore > 0.7 ? 'High' : riskScore > 0.4 ? 'Medium' : 'Low',
      lastVisit: faker.date.recent({ days: 90 }).toISOString(),
      nextAppointment: faker.date.soon({ days: 30 }).toISOString(),
      conditions: faker.helpers.arrayElements(conditions, { min: 1, max: 4 }),
      medications: faker.helpers.arrayElements(medications, { min: 1, max: 5 }),
      riskFactors,
      riskTrend,
      recommendedActions,
      contact: {
        phone: faker.phone.number(),
        email: faker.internet.email({ firstName, lastName }),
        address: faker.location.streetAddress()
      },
      vitalSigns: {
        bloodPressure: `${faker.number.int({ min: 90, max: 160 })}/${faker.number.int({ min: 60, max: 100 })}`,
        heartRate: faker.number.int({ min: 50, max: 100 }),
        weight: faker.number.int({ min: 50, max: 120 }),
        height: faker.number.int({ min: 150, max: 195 })
      }
    });
  }
  
  return patients;
};

// Generate and export the patient data
const patientData = generatePatientData(55);

export const getPatients = () => patientData;

export const getPatientById = (id) => 
  patientData.find(patient => patient.id === id);

export const getRiskDistribution = () => {
  const distribution = {
    high: 0,
    medium: 0,
    low: 0
  };
  
  patientData.forEach(patient => {
    if (patient.riskScore > 0.7) distribution.high++;
    else if (patient.riskScore > 0.4) distribution.medium++;
    else distribution.low++;
  });
  
  return distribution;
};

export const getRiskTrend = (patientId) => {
  const patient = patientData.find(p => p.id === patientId);
  return patient ? patient.riskTrend : [];
};

export const getRecommendedActions = (patientId) => {
  const patient = patientData.find(p => p.id === patientId);
  return patient ? patient.recommendedActions : [];
};

export default {
  getPatients,
  getPatientById,
  getRiskDistribution,
  getRiskTrend,
  getRecommendedActions
};
