import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from 'react-query';
import {
  ArrowLeftIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ChartBarIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const PatientDetail = () => {
  const { patientId } = useParams();
  const [activeTab, setActiveTab] = useState('overview');

  // Mock patient data
  const mockPatientData = {
    patient_id: patientId,
    name: 'John Smith',
    age: 67,
    gender: 'Male',
    mrn: 'MRN123456',
    last_visit: '2025-01-05T10:30:00Z',
    risk_score: 0.72,
    risk_bucket: 'High',
    prediction_timestamp: '2025-01-08T08:00:00Z',
    demographics: {
      date_of_birth: '1957-03-15',
      insurance: 'Medicare',
      primary_care_physician: 'Dr. Emily Chen',
      emergency_contact: 'Jane Smith (Daughter)',
    },
    clinical_summary: {
      primary_diagnosis: 'Heart Failure with Reduced Ejection Fraction',
      comorbidities: ['Diabetes Type 2', 'Hypertension', 'Chronic Kidney Disease'],
      current_medications: [
        'Lisinopril 10mg daily',
        'Metformin 1000mg twice daily',
        'Furosemide 40mg daily',
        'Carvedilol 6.25mg twice daily'
      ],
      allergies: ['Penicillin', 'Sulfa drugs'],
    },
    risk_factors: [
      {
        factor: 'Elevated BNP',
        importance: 0.25,
        value: '850 pg/mL',
        normal_range: '<100 pg/mL',
        trend: 'increasing',
      },
      {
        factor: 'Recent Weight Gain',
        importance: 0.18,
        value: '+5 lbs in 7 days',
        normal_range: 'Stable',
        trend: 'concerning',
      },
      {
        factor: 'Medication Non-adherence',
        importance: 0.15,
        value: '65% adherence',
        normal_range: '>80%',
        trend: 'poor',
      },
      {
        factor: 'Decreased Exercise Tolerance',
        importance: 0.12,
        value: 'NYHA Class III',
        normal_range: 'Class I-II',
        trend: 'worsening',
      },
    ],
    vital_trends: [
      { date: '2025-01-01', weight: 185, bp_systolic: 140, bp_diastolic: 85, heart_rate: 78 },
      { date: '2025-01-02', weight: 186, bp_systolic: 145, bp_diastolic: 88, heart_rate: 82 },
      { date: '2025-01-03', weight: 188, bp_systolic: 142, bp_diastolic: 86, heart_rate: 80 },
      { date: '2025-01-04', weight: 189, bp_systolic: 148, bp_diastolic: 90, heart_rate: 85 },
      { date: '2025-01-05', weight: 190, bp_systolic: 150, bp_diastolic: 92, heart_rate: 88 },
    ],
    lab_trends: [
      { date: '2024-12-15', bnp: 650, creatinine: 1.4, sodium: 138, potassium: 4.2 },
      { date: '2025-01-01', bnp: 750, creatinine: 1.5, sodium: 136, potassium: 4.0 },
      { date: '2025-01-05', bnp: 850, creatinine: 1.6, sodium: 135, potassium: 3.9 },
    ],
    recommendations: [
      {
        category: 'Immediate',
        action: 'Contact patient within 24 hours to assess symptoms',
        priority: 'high',
      },
      {
        category: 'Clinical',
        action: 'Consider increasing diuretic dose or adding thiazide',
        priority: 'high',
      },
      {
        category: 'Monitoring',
        action: 'Schedule follow-up appointment within 1 week',
        priority: 'medium',
      },
      {
        category: 'Education',
        action: 'Reinforce medication adherence and daily weight monitoring',
        priority: 'medium',
      },
    ],
    explanation: {
      summary: 'This patient has a 72% probability of clinical deterioration within 90 days based on recent clinical indicators.',
      key_drivers: [
        'Elevated BNP levels (850 pg/mL) indicate worsening heart failure',
        'Recent weight gain of 5 pounds suggests fluid retention',
        'Poor medication adherence (65%) increases risk of decompensation',
        'Decreased exercise tolerance indicates functional decline'
      ],
      clinical_context: 'The combination of biomarker elevation, physical signs of fluid retention, and poor self-care behaviors creates a high-risk profile for this heart failure patient.',
    }
  };

  const { data: patientData, isLoading, error } = useQuery(
    ['patient', patientId],
    () => {
      // In a real app, this would make an API call
      // return axios.get(`/patient/${patientId}`);
      return Promise.resolve({ data: mockPatientData });
    }
  );

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  const getRiskBadgeColor = (riskBucket) => {
    switch (riskBucket) {
      case 'High':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'Medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'Low':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high':
        return 'text-red-600 bg-red-50';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50';
      case 'low':
        return 'text-green-600 bg-green-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="spinner"></div>
      </div>
    );
  }

  if (error || !patientData?.data) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">Error loading patient data</p>
      </div>
    );
  }

  const patient = patientData.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link
            to="/cohort"
            className="inline-flex items-center text-sm font-medium text-gray-500 hover:text-gray-700"
          >
            <ArrowLeftIcon className="h-4 w-4 mr-1" />
            Back to Cohort
          </Link>
        </div>
      </div>

      {/* Patient Header */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:px-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="h-16 w-16 bg-gray-300 rounded-full flex items-center justify-center">
                <span className="text-gray-600 font-medium text-xl">
                  {patient.name.charAt(0)}
                </span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">{patient.name}</h1>
                <div className="flex items-center space-x-4 mt-1">
                  <span className="text-sm text-gray-500">
                    Age {patient.age} • {patient.gender} • MRN: {patient.mrn}
                  </span>
                  <span
                    className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${getRiskBadgeColor(
                      patient.risk_bucket
                    )}`}
                  >
                    <ExclamationTriangleIcon className="h-4 w-4 mr-1" />
                    {patient.risk_bucket} Risk ({(patient.risk_score * 100).toFixed(0)}%)
                  </span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500">Last Visit</p>
              <p className="text-sm font-medium text-gray-900">
                {formatDate(patient.last_visit)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', name: 'Overview', icon: InformationCircleIcon },
            { id: 'trends', name: 'Trends', icon: ChartBarIcon },
            { id: 'timeline', name: 'Timeline', icon: ClockIcon },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`group inline-flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="h-5 w-5 mr-2" />
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Risk Explanation */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Risk Assessment Explanation
                </h3>
                <div className="space-y-4">
                  <div className="bg-red-50 border border-red-200 rounded-md p-4">
                    <p className="text-sm text-red-800">{patient.explanation.summary}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 mb-2">Key Risk Drivers:</h4>
                    <ul className="space-y-2">
                      {patient.explanation.key_drivers.map((driver, index) => (
                        <li key={index} className="flex items-start">
                          <span className="flex-shrink-0 h-1.5 w-1.5 bg-red-500 rounded-full mt-2 mr-3" />
                          <span className="text-sm text-gray-700">{driver}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                    <p className="text-sm text-blue-800">{patient.explanation.clinical_context}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Risk Factors */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Contributing Risk Factors
                </h3>
                <div className="space-y-4">
                  {patient.risk_factors.map((factor, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-sm font-medium text-gray-900">{factor.factor}</h4>
                        <span className="text-xs text-gray-500">
                          {(factor.importance * 100).toFixed(0)}% importance
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                        <div
                          className="bg-red-500 h-2 rounded-full"
                          style={{ width: `${factor.importance * 100}%` }}
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Current: </span>
                          <span className="font-medium">{factor.value}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Normal: </span>
                          <span className="font-medium">{factor.normal_range}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Recommendations */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Recommended Actions
                </h3>
                <div className="space-y-3">
                  {patient.recommendations.map((rec, index) => (
                    <div
                      key={index}
                      className={`p-3 rounded-md border ${getPriorityColor(rec.priority)}`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-medium uppercase tracking-wide">
                          {rec.category}
                        </span>
                        <span className="text-xs font-medium capitalize">
                          {rec.priority} Priority
                        </span>
                      </div>
                      <p className="text-sm">{rec.action}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Clinical Summary */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Clinical Summary
                </h3>
                <div className="space-y-4 text-sm">
                  <div>
                    <h4 className="font-medium text-gray-900">Primary Diagnosis</h4>
                    <p className="text-gray-700">{patient.clinical_summary.primary_diagnosis}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">Comorbidities</h4>
                    <ul className="text-gray-700 list-disc list-inside">
                      {patient.clinical_summary.comorbidities.map((condition, index) => (
                        <li key={index}>{condition}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">Current Medications</h4>
                    <ul className="text-gray-700 list-disc list-inside">
                      {patient.clinical_summary.current_medications.map((med, index) => (
                        <li key={index}>{med}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">Allergies</h4>
                    <p className="text-gray-700">{patient.clinical_summary.allergies.join(', ')}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'trends' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Vital Signs Trends */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Vital Signs Trends
              </h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={patient.vital_trends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="weight" stroke="#ef4444" name="Weight (lbs)" />
                    <Line type="monotone" dataKey="bp_systolic" stroke="#3b82f6" name="Systolic BP" />
                    <Line type="monotone" dataKey="heart_rate" stroke="#10b981" name="Heart Rate" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Lab Trends */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Laboratory Trends
              </h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={patient.lab_trends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="bnp" stroke="#ef4444" name="BNP (pg/mL)" />
                    <Line type="monotone" dataKey="creatinine" stroke="#f59e0b" name="Creatinine" />
                    <Line type="monotone" dataKey="sodium" stroke="#3b82f6" name="Sodium" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'timeline' && (
        <div className="bg-white shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Clinical Timeline
            </h3>
            <div className="flow-root">
              <ul className="-mb-8">
                <li>
                  <div className="relative pb-8">
                    <span className="absolute top-4 left-4 -ml-px h-full w-0.5 bg-gray-200" />
                    <div className="relative flex space-x-3">
                      <div>
                        <span className="h-8 w-8 rounded-full bg-red-500 flex items-center justify-center ring-8 ring-white">
                          <ExclamationTriangleIcon className="h-5 w-5 text-white" />
                        </span>
                      </div>
                      <div className="min-w-0 flex-1 pt-1.5 flex justify-between space-x-4">
                        <div>
                          <p className="text-sm text-gray-500">
                            High risk prediction generated{' '}
                            <span className="font-medium text-gray-900">72% risk score</span>
                          </p>
                        </div>
                        <div className="text-right text-sm whitespace-nowrap text-gray-500">
                          <time dateTime="2025-01-08">Jan 8, 2025</time>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
                <li>
                  <div className="relative pb-8">
                    <span className="absolute top-4 left-4 -ml-px h-full w-0.5 bg-gray-200" />
                    <div className="relative flex space-x-3">
                      <div>
                        <span className="h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center ring-8 ring-white">
                          <ChartBarIcon className="h-5 w-5 text-white" />
                        </span>
                      </div>
                      <div className="min-w-0 flex-1 pt-1.5 flex justify-between space-x-4">
                        <div>
                          <p className="text-sm text-gray-500">
                            Lab results received{' '}
                            <span className="font-medium text-gray-900">BNP: 850 pg/mL</span>
                          </p>
                        </div>
                        <div className="text-right text-sm whitespace-nowrap text-gray-500">
                          <time dateTime="2025-01-05">Jan 5, 2025</time>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
                <li>
                  <div className="relative">
                    <div className="relative flex space-x-3">
                      <div>
                        <span className="h-8 w-8 rounded-full bg-green-500 flex items-center justify-center ring-8 ring-white">
                          <ClockIcon className="h-5 w-5 text-white" />
                        </span>
                      </div>
                      <div className="min-w-0 flex-1 pt-1.5 flex justify-between space-x-4">
                        <div>
                          <p className="text-sm text-gray-500">
                            Office visit completed{' '}
                            <span className="font-medium text-gray-900">Routine follow-up</span>
                          </p>
                        </div>
                        <div className="text-right text-sm whitespace-nowrap text-gray-500">
                          <time dateTime="2025-01-05">Jan 5, 2025</time>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PatientDetail;
