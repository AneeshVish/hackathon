import React from 'react';

const ConfusionMatrix = ({ data }) => {
  if (!data) return null;
  
  const { truePositive, falsePositive, falseNegative, trueNegative } = data;
  const total = truePositive + falsePositive + trueNegative + falseNegative;

  // Calculate metrics
  const accuracy = ((truePositive + trueNegative) / total * 100).toFixed(1);
  const precision = (truePositive / (truePositive + falsePositive) * 100).toFixed(1);
  const recall = (truePositive / (truePositive + falseNegative) * 100).toFixed(1);
  const f1Score = (2 * (precision * recall) / (parseFloat(precision) + parseFloat(recall))).toFixed(1);

  // Cell style function
  const cellStyle = (value, isGood) => ({
    backgroundColor: isGood 
      ? `rgba(16, 185, 129, ${0.2 + (value / total) * 0.8})` 
      : `rgba(239, 68, 68, ${0.2 + (value / total) * 0.8})`,
    border: '1px solid #e5e7eb',
    padding: '1.5rem',
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: '1.25rem',
    borderRadius: '0.5rem',
    color: isGood ? '#065f46' : '#991b1b',
    minWidth: '120px'
  });

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        {/* Top Left: True Positives */}
        <div style={cellStyle(truePositive, true)}>
          <div className="text-sm font-medium">True Positives</div>
          <div className="text-3xl font-bold">{truePositive}</div>
        </div>
        
        {/* Top Right: False Positives */}
        <div style={cellStyle(falsePositive, false)}>
          <div className="text-sm font-medium">False Positives</div>
          <div className="text-3xl font-bold">{falsePositive}</div>
        </div>
        
        {/* Bottom Left: False Negatives */}
        <div style={cellStyle(falseNegative, false)}>
          <div className="text-sm font-medium">False Negatives</div>
          <div className="text-3xl font-bold">{falseNegative}</div>
        </div>
        
        {/* Bottom Right: True Negatives */}
        <div style={cellStyle(trueNegative, true)}>
          <div className="text-sm font-medium">True Negatives</div>
          <div className="text-3xl font-bold">{trueNegative}</div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-sm font-medium text-green-800">Accuracy</div>
          <div className="text-2xl font-bold text-green-600">{accuracy}%</div>
        </div>
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-sm font-medium text-blue-800">Precision</div>
          <div className="text-2xl font-bold text-blue-600">{precision}%</div>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-sm font-medium text-purple-800">Recall</div>
          <div className="text-2xl font-bold text-purple-600">{recall}%</div>
        </div>
        <div className="bg-yellow-50 p-4 rounded-lg">
          <div className="text-sm font-medium text-yellow-800">F1 Score</div>
          <div className="text-2xl font-bold text-yellow-600">{f1Score}%</div>
        </div>
      </div>
    </div>
  );
};

export default ConfusionMatrix;
