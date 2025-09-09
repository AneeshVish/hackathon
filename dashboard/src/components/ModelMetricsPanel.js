import React from 'react';
import { 
  ChartBarIcon, 
  BeakerIcon, 
  CheckCircleIcon,
  ExclamationTriangleIcon 
} from '@heroicons/react/24/outline';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ModelMetricsPanel = ({ 
  modelMetrics, 
  confusionMatrix, 
  featureImportance, 
  calibrationData 
}) => {
  
  // Prepare confusion matrix data for visualization
  const confusionData = [
    { name: 'True Positive', value: confusionMatrix.truePositive, color: '#22c55e' },
    { name: 'False Positive', value: confusionMatrix.falsePositive, color: '#ef4444' },
    { name: 'True Negative', value: confusionMatrix.trueNegative, color: '#22c55e' },
    { name: 'False Negative', value: confusionMatrix.falseNegative, color: '#ef4444' }
  ];

  return (
    <div className="space-y-6">
      {/* Model Performance Header */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-2xl border border-blue-100">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-2 bg-blue-100 rounded-lg">
            <BeakerIcon className="h-6 w-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Model Evaluation Metrics</h2>
            <p className="text-sm text-gray-600">90-Day Patient Deterioration Prediction Performance</p>
          </div>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">AUROC</p>
              <p className="text-3xl font-bold text-blue-600">{modelMetrics.auc?.toFixed(3) || 'N/A'}</p>
              <p className="text-xs text-gray-500 mt-1">Area Under ROC Curve</p>
            </div>
            <div className="p-3 bg-blue-100 rounded-lg">
              <ChartBarIcon className="h-6 w-6 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">AUPRC</p>
              <p className="text-3xl font-bold text-green-600">{modelMetrics.avg_precision?.toFixed(3) || 'N/A'}</p>
              <p className="text-xs text-gray-500 mt-1">Area Under PR Curve</p>
            </div>
            <div className="p-3 bg-green-100 rounded-lg">
              <CheckCircleIcon className="h-6 w-6 text-green-600" />
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Accuracy</p>
              <p className="text-3xl font-bold text-purple-600">{(modelMetrics.accuracy * 100)?.toFixed(1) || 'N/A'}%</p>
              <p className="text-xs text-gray-500 mt-1">Overall Accuracy</p>
            </div>
            <div className="p-3 bg-purple-100 rounded-lg">
              <CheckCircleIcon className="h-6 w-6 text-purple-600" />
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Brier Score</p>
              <p className="text-3xl font-bold text-orange-600">{modelMetrics.brier_score?.toFixed(3) || 'N/A'}</p>
              <p className="text-xs text-gray-500 mt-1">Calibration Quality</p>
            </div>
            <div className="p-3 bg-orange-100 rounded-lg">
              <ExclamationTriangleIcon className="h-6 w-6 text-orange-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Confusion Matrix</h3>
        <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
          <div className="text-center">
            <div className="text-xs font-medium text-gray-500 mb-2">Predicted</div>
            <div className="flex">
              <div className="w-16 text-xs font-medium text-gray-500 flex items-center justify-center">Actual</div>
              <div className="flex-1">
                <div className="grid grid-cols-2 gap-1 text-xs font-medium text-gray-600 mb-1">
                  <div className="text-center">Negative</div>
                  <div className="text-center">Positive</div>
                </div>
                <div className="grid grid-cols-2 gap-1">
                  <div className="bg-green-100 border-2 border-green-300 p-4 text-center">
                    <div className="text-xs text-gray-600">TN</div>
                    <div className="text-lg font-bold text-green-700">{confusionMatrix.trueNegative}</div>
                  </div>
                  <div className="bg-red-100 border-2 border-red-300 p-4 text-center">
                    <div className="text-xs text-gray-600">FP</div>
                    <div className="text-lg font-bold text-red-700">{confusionMatrix.falsePositive}</div>
                  </div>
                  <div className="bg-red-100 border-2 border-red-300 p-4 text-center">
                    <div className="text-xs text-gray-600">FN</div>
                    <div className="text-lg font-bold text-red-700">{confusionMatrix.falseNegative}</div>
                  </div>
                  <div className="bg-green-100 border-2 border-green-300 p-4 text-center">
                    <div className="text-xs text-gray-600">TP</div>
                    <div className="text-lg font-bold text-green-700">{confusionMatrix.truePositive}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Additional Metrics */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-sm font-medium text-blue-900">Sensitivity</div>
            <div className="text-lg font-bold text-blue-600">{(modelMetrics.recall * 100)?.toFixed(1) || 'N/A'}%</div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-sm font-medium text-green-900">Specificity</div>
            <div className="text-lg font-bold text-green-600">{(modelMetrics.specificity * 100)?.toFixed(1) || 'N/A'}%</div>
          </div>
          <div className="text-center p-3 bg-purple-50 rounded-lg">
            <div className="text-sm font-medium text-purple-900">PPV</div>
            <div className="text-lg font-bold text-purple-600">{(modelMetrics.ppv * 100)?.toFixed(1) || 'N/A'}%</div>
          </div>
          <div className="text-center p-3 bg-orange-50 rounded-lg">
            <div className="text-sm font-medium text-orange-900">NPV</div>
            <div className="text-lg font-bold text-orange-600">{(modelMetrics.npv * 100)?.toFixed(1) || 'N/A'}%</div>
          </div>
        </div>
      </div>

      {/* Feature Importance - Global Explainability */}
      <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Global Feature Importance</h3>
        <p className="text-sm text-gray-600 mb-4">
          Key clinical factors driving 90-day deterioration predictions across all patients
        </p>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={featureImportance.slice(0, 10)} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis type="number" tick={{ fontSize: 12 }} stroke="#64748b" />
              <YAxis 
                type="category" 
                dataKey="feature" 
                tick={{ fontSize: 11 }} 
                stroke="#64748b"
                width={120}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #e2e8f0', 
                  borderRadius: '12px',
                  boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
                }}
                formatter={(value, name) => [
                  `${(value * 100).toFixed(2)}%`, 
                  'Importance'
                ]}
              />
              <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Calibration Plot */}
      <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Model Calibration</h3>
        <p className="text-sm text-gray-600 mb-4">
          How well predicted probabilities match actual outcomes
        </p>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={calibrationData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis 
                dataKey="predicted" 
                tick={{ fontSize: 12 }} 
                stroke="#64748b"
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <YAxis 
                tick={{ fontSize: 12 }} 
                stroke="#64748b"
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #e2e8f0', 
                  borderRadius: '12px',
                  boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
                }}
                formatter={(value, name) => [
                  `${(value * 100).toFixed(1)}%`, 
                  name === 'predicted' ? 'Predicted' : 'Observed'
                ]}
              />
              <Bar dataKey="predicted" fill="#94a3b8" name="Predicted" />
              <Bar dataKey="observed" fill="#3b82f6" name="Observed" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default ModelMetricsPanel;
