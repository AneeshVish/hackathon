import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line,
  Scatter, ComposedChart, Area, RadialBarChart, RadialBar,
  ReferenceLine, LabelList
} from 'recharts';
import {
  ChartBarIcon, CpuChipIcon, ClockIcon, CheckCircleIcon,
  ExclamationTriangleIcon, ArrowTrendingUpIcon, UserGroupIcon,
  ScaleIcon, HeartIcon, BeakerIcon
} from '@heroicons/react/24/solid';
import {
  trainedModelMetrics, trainedConfusionMatrix, trainedFeatureImportance,
  trainedCalibrationData, trainedModelHistory, trainedPatientPredictions,
  trainedRiskDistribution
} from '../data/trainedModelData';
import ConfusionMatrix from '../components/ConfusionMatrix';


// Format numbers with commas
const formatNumber = (num) => {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
};

// Calculate model statistics
const calculateModelStats = (predictions) => {
  if (!predictions || predictions.length === 0) {
    return {
      totalPatients: 0,
      highRisk: 0,
      mediumRisk: 0,
      lowRisk: 0,
      highRiskPct: 0,
      mediumRiskPct: 0,
      lowRiskPct: 0
    };
  }
  
  const total = predictions.length;
  const highRisk = predictions.filter(p => p.risk_bucket === 'High').length;
  const mediumRisk = predictions.filter(p => p.risk_bucket === 'Medium').length;
  const lowRisk = predictions.filter(p => p.risk_bucket === 'Low').length;
  
  return {
    totalPatients: total,
    highRisk,
    mediumRisk,
    lowRisk,
    highRiskPct: Math.round((highRisk / total) * 100) || 0,
    mediumRiskPct: Math.round((mediumRisk / total) * 100) || 0,
    lowRiskPct: Math.round((lowRisk / total) * 100) || 0
  };
};

const Analytics = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('30d');
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Initialize data with default values
  const [data, setData] = useState({
    modelMetrics: {},
    featureImportance: [],
    confusionMatrix: {},
    calibrationData: [],
    modelHistory: [],
    patientPredictions: [],
    riskDistribution: {},
    riskCategories: []
  });

  // Load data on component mount
  useEffect(() => {
    try {
      // In a real app, you would fetch this data from an API
      // For now, we'll use the imported data directly
      const modelStats = calculateModelStats(trainedPatientPredictions || []);
      
      setData({
        modelMetrics: trainedModelMetrics || {},
        featureImportance: (trainedFeatureImportance || []).map(item => ({
          ...item,
          category: 'Clinical'
        })),
        calibrationData: trainedCalibrationData || [],
        modelHistory: trainedModelHistory || [],
        patientPredictions: trainedPatientPredictions || [],
        riskDistribution: trainedRiskDistribution || {},
        riskCategories: [
          { name: 'Low Risk', value: modelStats.lowRisk, color: '#10b981' },
          { name: 'Medium Risk', value: modelStats.mediumRisk, color: '#f59e0b' },
          { name: 'High Risk', value: modelStats.highRisk, color: '#ef4444' }
        ]
      });
      setIsLoading(false);
    } catch (err) {
      console.error('Error loading analytics data:', err);
      setError('Failed to load analytics data. Please try again later.');
      setIsLoading(false);
    }
  }, []);

  // Destructure data for easier access
  const {
    modelMetrics,
    featureImportance,
    confusionMatrix,
    calibrationData,
    modelHistory,
    patientPredictions,
    riskDistribution
  } = data;

  // Performance over time data
  const performanceOverTime = [
    { date: '2024-01', accuracy: 0.87, auc: 0.92, precision: 0.85, recall: 0.89 },
    { date: '2024-02', accuracy: 0.88, auc: 0.93, precision: 0.86, recall: 0.90 },
    { date: '2024-03', accuracy: 0.89, auc: 0.94, precision: 0.87, recall: 0.91 },
    { date: '2024-04', accuracy: 0.90, auc: 0.95, precision: 0.88, recall: 0.92 },
    { date: '2024-05', accuracy: 0.91, auc: 0.96, precision: 0.89, recall: 0.93 },
    { date: '2024-06', accuracy: 0.92, auc: 0.97, precision: 0.90, recall: 0.94 }
  ];

  // Calculate prediction distribution from patient data
  const predictionDistribution = [
    { 
      range: '0-20%', 
      count: patientPredictions.filter(p => p.risk_score <= 0.2).length,
      color: '#10b981' 
    },
    { 
      range: '20-40%', 
      count: patientPredictions.filter(p => p.risk_score > 0.2 && p.risk_score <= 0.4).length,
      color: '#3b82f6' 
    },
    { 
      range: '40-60%', 
      count: patientPredictions.filter(p => p.risk_score > 0.4 && p.risk_score <= 0.6).length,
      color: '#f59e0b' 
    },
    { 
      range: '60-80%', 
      count: patientPredictions.filter(p => p.risk_score > 0.6 && p.risk_score <= 0.8).length,
      color: '#ef4444' 
    },
    { 
      range: '80-100%', 
      count: patientPredictions.filter(p => p.risk_score > 0.8).length,
      color: '#dc2626' 
    }
  ];
  
  // Calculate risk categories from patient predictions
  const riskCategories = [
    { name: 'High Risk', value: patientPredictions.filter(p => p.risk_bucket === 'High').length, color: '#ef4444' },
    { name: 'Medium Risk', value: patientPredictions.filter(p => p.risk_bucket === 'Medium').length, color: '#f59e0b' },
    { name: 'Low Risk', value: patientPredictions.filter(p => p.risk_bucket === 'Low').length, color: '#10b981' }
  ];

  // Use modelHistory as trainingHistory for compatibility
  const trainingHistory = modelHistory;

  // Tab navigation
  const tabs = [
    { id: 'overview', name: 'Overview' },
    { id: 'performance', name: 'Model Performance' },
    { id: 'predictions', name: 'Predictions' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="sm:flex sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h1>
          <p className="mt-1 text-sm text-gray-500">
            Comprehensive model performance and prediction analytics
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex space-x-3">
          <button
            type="button"
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          >
            Export Report
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`${activeTab === tab.id
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {activeTab === 'overview' && (
          <>
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">AUC-ROC</p>
                    <p className="text-3xl font-bold text-blue-600">{(modelMetrics.auc * 100).toFixed(1)}%</p>
                  </div>
                  <div className="p-3 bg-blue-100 rounded-lg">
                    <ChartBarIcon className="h-6 w-6 text-blue-600" />
                  </div>
                </div>
                <div className="mt-4">
                  <div className="flex items-center text-sm text-green-600">
                    <CheckCircleIcon className="h-4 w-4 mr-1" />
                    Excellent discrimination
                  </div>
                </div>
              </div>

              <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Accuracy</p>
                    <p className="text-3xl font-bold text-green-600">{(modelMetrics.accuracy * 100).toFixed(1)}%</p>
                  </div>
                  <div className="p-3 bg-green-100 rounded-lg">
                    <CheckCircleIcon className="h-6 w-6 text-green-600" />
                  </div>
                </div>
                <div className="mt-4">
                  <div className="flex items-center text-sm text-green-600">
                    <CheckCircleIcon className="h-4 w-4 mr-1" />
                    High accuracy
                  </div>
                </div>
              </div>

              <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Precision</p>
                    <p className="text-3xl font-bold text-purple-600">{(modelMetrics.precision * 100).toFixed(1)}%</p>
                  </div>
                  <div className="p-3 bg-purple-100 rounded-lg">
                    <BeakerIcon className="h-6 w-6 text-purple-600" />
                  </div>
                </div>
                <div className="mt-4">
                  <div className="flex items-center text-sm text-green-600">
                    <CheckCircleIcon className="h-4 w-4 mr-1" />
                    Low false positives
                  </div>
                </div>
              </div>

              <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Recall</p>
                    <p className="text-3xl font-bold text-orange-600">{(modelMetrics.recall * 100).toFixed(1)}%</p>
                  </div>
                  <div className="p-3 bg-orange-100 rounded-lg">
                    <CpuChipIcon className="h-6 w-6 text-orange-600" />
                  </div>
                </div>
                <div className="mt-4">
                  <div className="flex items-center text-sm text-green-600">
                    <CheckCircleIcon className="h-4 w-4 mr-1" />
                    Good sensitivity
                  </div>
                </div>
              </div>
            </div>

      {/* Additional Training History */}
      <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Model Training History</h3>
        <p className="text-sm text-gray-600 mb-4">Training progression over epochs</p>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="epoch" tick={{ fontSize: 12 }} stroke="#64748b" />
              <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} stroke="#64748b" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #e2e8f0', 
                  borderRadius: '12px',
                  boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
                }}
              />
                    <Line 
                      type="monotone" 
                      dataKey="train_auc" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      name="Training AUC" 
                      dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="val_auc" 
                      stroke="#ef4444" 
                      strokeWidth={2}
                      name="Validation AUC" 
                      dot={{ fill: '#ef4444', strokeWidth: 2, r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Second Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Prediction Distribution */}
              <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Risk Score Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={predictionDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #e2e8f0', 
                  borderRadius: '12px',
                  boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Bar dataKey="count" fill="#8884d8">
                {predictionDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Categories Pie Chart */}
        <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Patient Risk Categories</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskCategories}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {riskCategories.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

            {/* Performance Over Time */}
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Model Performance Over Time</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={performanceOverTime}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={[0.8, 1]} />
            <Tooltip 
              formatter={(value, name) => [`${(value * 100).toFixed(1)}%`, name]}
              contentStyle={{ 
                backgroundColor: 'white', 
                border: '1px solid #e2e8f0', 
                borderRadius: '12px',
                boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
              }}
            />
            <Line type="monotone" dataKey="accuracy" stroke="#3b82f6" strokeWidth={2} name="Accuracy" />
            <Line type="monotone" dataKey="auc" stroke="#10b981" strokeWidth={2} name="AUC-ROC" />
            <Line type="monotone" dataKey="precision" stroke="#f59e0b" strokeWidth={2} name="Precision" />
            <Line type="monotone" dataKey="recall" stroke="#ef4444" strokeWidth={2} name="Recall" />
          </LineChart>
        </ResponsiveContainer>
      </div>

            {/* Calibration Plot */}
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Model Calibration</h3>
              <p className="text-sm text-gray-600 mb-4">Comparison between predicted probabilities and observed outcomes</p>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={calibrationData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="predicted" name="Predicted Probability" />
                  <YAxis dataKey="observed" name="Observed Frequency" />
            <Tooltip 
              formatter={(value, name) => [value.toFixed(3), name]}
              contentStyle={{ 
                backgroundColor: 'white', 
                border: '1px solid #e2e8f0', 
                borderRadius: '12px',
                boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
              }}
            />
            <Scatter dataKey="observed" fill="#3b82f6" name="Observed" />
            <Line type="monotone" dataKey="predicted" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" name="Perfect Calibration" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

            {/* Model Information */}
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Model Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-1 gap-6">
                <div className="space-y-4">
                  <h4 className="font-semibold text-gray-900">Model Type</h4>
                  <p className="text-gray-600">LightGBM</p>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'performance' && (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
          <h3 className="text-xl font-bold text-gray-900 mb-6">Model Performance</h3>
          <ConfusionMatrix data={{
            truePositive: trainedConfusionMatrix.truePositive || 0,
            falsePositive: trainedConfusionMatrix.falsePositive || 0,
            falseNegative: trainedConfusionMatrix.falseNegative || 0,
            trueNegative: trainedConfusionMatrix.trueNegative || 0
          }} />
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Feature Importance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={[...featureImportance].sort((a, b) => b.importance - a.importance).slice(0, 10)} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 0.2]} />
              <YAxis dataKey="feature" type="category" width={120} fontSize={12} />
              <Tooltip 
                formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Importance']}
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #e2e8f0', 
                  borderRadius: '12px',
                  boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Model Calibration</h3>
          <p className="text-sm text-gray-600 mb-4">Comparison between predicted probabilities and observed outcomes</p>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={calibrationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="predicted" name="Predicted Probability" domain={[0, 1]} />
              <YAxis dataKey="observed" name="Observed Frequency" domain={[0, 1]} />
              <Tooltip 
                formatter={(value, name) => [value.toFixed(3), name === 'observed' ? 'Observed' : 'Ideal']}
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #e2e8f0', 
                  borderRadius: '12px',
                  boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Scatter dataKey="observed" fill="#3b82f6" name="Observed" />
              <Line type="monotone" dataKey="predicted" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" name="Perfect Calibration" />
              <Legend />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    )}

    {activeTab === 'predictions' && (
      <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Risk Score Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={predictionDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" />
                <YAxis />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e2e8f0', 
                    borderRadius: '12px',
                    boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
                  }}
                />
                <Bar dataKey="count">
                  {predictionDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Patient Risk Categories</h3>
            <div className="h-64 flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskCategories}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    innerRadius={40}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {riskCategories.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                    <LabelList 
                      dataKey="name"
                      position="outside"
                      offset={20}
                      style={{
                        fontSize: '12px',
                        fontWeight: 'bold',
                        fill: '#4B5563'
                      }}
                    />
                  </Pie>
                  <Tooltip 
                    formatter={(value, name) => [`${value} patients`, name]}
                    contentStyle={{ 
                      backgroundColor: 'white', 
                      border: '1px solid #e2e8f0', 
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 grid grid-cols-3 gap-2 text-center">
              {riskCategories.map((category, index) => (
                <div key={index} className="text-sm">
                  <div className="font-medium" style={{ color: category.color }}>
                    {category.name}
                  </div>
                  <div className="text-gray-600">
                    {category.value} patients
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    )}
        </div>
    </div>
  );

};

export default Analytics;
