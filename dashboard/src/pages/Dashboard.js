import React, { useState, useEffect } from 'react';
import {
  UserGroupIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
} from '@heroicons/react/24/outline';
import { 
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, 
  CartesianGrid, Tooltip, PieChart, Pie 
} from 'recharts';
import { patientData } from '../data/patientData';
import ModelMetricsPanel from '../components/ModelMetricsPanel';

const Dashboard = () => {
  // Calculate patient statistics
  const patientStats = {
    total: patientData.length,
    high: patientData.filter(p => p.risk_category === 'high').length,
    medium: patientData.filter(p => p.risk_category === 'medium').length,
    low: patientData.filter(p => p.risk_category === 'low').length,
  };

  // Calculate risk distribution
  const riskDistribution = [
    { name: 'High', value: patientStats.high, color: '#EF4444' },
    { name: 'Medium', value: patientStats.medium, color: '#F59E0B' },
    { name: 'Low', value: patientStats.low, color: '#10B981' }
  ];

  // Mock model metrics based on patient data
  const modelMetrics = {
    accuracy: 0.8545,
    precision: 0.8636,
    recall: 0.9048,
    f1: 0.8837,
    auc: 0.9231,
  };

  const statsData = [
    {
      name: 'Total Patients',
      value: patientStats.total,
      change: `${Math.round((patientStats.total / 50 - 1) * 100)}%`,
      changeType: patientStats.total >= 50 ? 'positive' : 'negative',
      icon: UserGroupIcon,
      description: 'Active patients in monitoring system'
    },
    {
      name: 'High Risk',
      value: patientStats.high,
      change: patientStats.high > 0 ? 'Needs Attention' : 'Stable',
      changeType: patientStats.high > 0 ? 'negative' : 'positive',
      icon: ExclamationTriangleIcon,
      description: 'Patients with >70% deterioration risk'
    },
    {
      name: 'Medium Risk (30-70%)',
      value: patientStats.medium.toString(),
      change: '-2.1%',
      changeType: 'positive',
      icon: ClockIcon,
      description: 'Patients requiring monitoring'
    },
    {
      name: 'Low Risk (<30%)',
      value: patientStats.low.toString(),
      change: '+1.2%',
      changeType: 'positive',
      icon: CheckCircleIcon,
      description: 'Stable patients'
    }
  ];

  const riskTrendData = [
    { month: 'Jan', high: patientStats.high - 1, medium: patientStats.medium + 1, low: patientStats.low },
    { month: 'Feb', high: patientStats.high - 1, medium: patientStats.medium, low: patientStats.low + 1 },
    { month: 'Mar', high: patientStats.high, medium: patientStats.medium, low: patientStats.low },
    { month: 'Apr', high: patientStats.high, medium: patientStats.medium, low: patientStats.low },
    { month: 'May', high: patientStats.high, medium: patientStats.medium, low: patientStats.low },
  ];

  const riskDistributionChart = [
    { name: 'Low Risk', value: riskDistribution.low, color: '#10b981' },
    { name: 'Medium Risk', value: riskDistribution.medium, color: '#f59e0b' },
    { name: 'High Risk', value: riskDistribution.high, color: '#ef4444' },
  ];

  const [currentTime, setCurrentTime] = useState(new Date());
  const [animatedStats, setAnimatedStats] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    const animationTimer = setTimeout(() => setAnimatedStats(true), 100);

    return () => {
      clearInterval(timer);
      clearTimeout(animationTimer);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="space-y-8 p-6">
        {/* Enhanced Page Header */}
        <div className="relative overflow-hidden bg-white rounded-2xl shadow-xl border border-gray-100">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-600 opacity-5"></div>
          <div className="relative px-8 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
                  Patient Risk Dashboard
                </h1>
                <p className="mt-2 text-lg text-gray-600">
                  Real-time deterioration risk monitoring & predictions
                </p>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 text-sm text-gray-500">
                  <ClockIcon className="h-4 w-4" />
                  <span>{currentTime.toLocaleTimeString()}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium text-green-600">Live</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Stats Grid */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {statsData.map((stat, index) => {
            const isHighRisk = stat.name === 'High Risk';
            const gradientClass = isHighRisk 
              ? 'from-red-500 to-pink-600' 
              : stat.name === 'Medium Risk' 
              ? 'from-amber-500 to-orange-600'
              : stat.name === 'Low Risk'
              ? 'from-green-500 to-emerald-600'
              : 'from-blue-500 to-indigo-600';
            
            return (
              <div
                key={stat.name}
                className={`group relative overflow-hidden rounded-2xl bg-white shadow-lg hover:shadow-2xl transition-all duration-500 transform hover:-translate-y-1 border border-gray-100 ${
                  animatedStats ? 'animate-fade-in-up' : 'opacity-0'
                }`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${gradientClass} opacity-5 group-hover:opacity-10 transition-opacity duration-300`}></div>
                <div className="relative px-6 py-6">
                  <div className="flex items-center justify-between">
                    <div className={`flex-shrink-0 p-3 rounded-xl bg-gradient-to-br ${gradientClass} shadow-lg`}>
                      <stat.icon className="h-6 w-6 text-white" aria-hidden="true" />
                    </div>
                    <div className="flex items-center space-x-1">
                      {stat.changeType === 'positive' ? (
                        <ArrowTrendingUpIcon className="h-4 w-4 text-green-500" />
                      ) : (
                        <ArrowTrendingDownIcon className="h-4 w-4 text-red-500" />
                      )}
                      <span className={`text-sm font-semibold ${
                        stat.changeType === 'positive' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {stat.change}
                      </span>
                    </div>
                  </div>
                  <div className="mt-4">
                    <p className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                      {stat.name}
                    </p>
                    <p className="text-3xl font-bold text-gray-900 mt-1">{stat.value}</p>
                  </div>
                  {isHighRisk && (
                    <div className="absolute top-2 right-2">
                      <div className="h-2 w-2 bg-red-400 rounded-full animate-pulse"></div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Risk Trend Chart - Enhanced */}
          <div className="lg:col-span-2 bg-white overflow-hidden shadow-xl rounded-2xl border border-gray-100">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 px-6 py-4 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-bold text-gray-900">
                  Risk Trend Analysis
                </h3>
                <div className="flex items-center space-x-2">
                  <div className="flex items-center space-x-1">
                    <div className="h-3 w-3 bg-red-500 rounded-full"></div>
                    <span className="text-xs font-medium text-gray-600">High</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="h-3 w-3 bg-amber-500 rounded-full"></div>
                    <span className="text-xs font-medium text-gray-600">Medium</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="h-3 w-3 bg-green-500 rounded-full"></div>
                    <span className="text-xs font-medium text-gray-600">Low</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={riskTrendData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <defs>
                      <linearGradient id="highRisk" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0.3}/>
                      </linearGradient>
                      <linearGradient id="mediumRisk" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.3}/>
                      </linearGradient>
                      <linearGradient id="lowRisk" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#22c55e" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#22c55e" stopOpacity={0.3}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="month" tick={{ fontSize: 12 }} stroke="#64748b" />
                    <YAxis tick={{ fontSize: 12 }} stroke="#64748b" domain={[0, 6]} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white', 
                        border: '1px solid #e2e8f0', 
                        borderRadius: '12px',
                        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
                      }} 
                    />
                    <Bar dataKey="low" stackId="a" fill="url(#lowRisk)" name="Low Risk" radius={[0, 0, 4, 4]} />
                    <Bar dataKey="medium" stackId="a" fill="url(#mediumRisk)" name="Medium Risk" />
                    <Bar dataKey="high" stackId="a" fill="url(#highRisk)" name="High Risk" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Risk Distribution - Enhanced */}
          <div className="bg-white overflow-hidden shadow-xl rounded-2xl border border-gray-100">
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 px-6 py-4 border-b border-gray-100">
              <h3 className="text-xl font-bold text-gray-900">
                Risk Distribution
              </h3>
              <p className="text-sm text-gray-600 mt-1">Current patient status</p>
            </div>
            <div className="p-6">
              <div className="h-64 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={riskDistributionChart}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      stroke="white"
                      strokeWidth={3}
                    >
                      {riskDistributionChart.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white', 
                        border: '1px solid #e2e8f0', 
                        borderRadius: '12px',
                        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
                      }} 
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-3">
                {riskDistributionChart.map((item, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className={`h-3 w-3 rounded-full`} style={{ backgroundColor: item.color }}></div>
                      <span className="text-sm font-medium text-gray-700">{item.name}</span>
                    </div>
                    <span className="text-sm font-bold text-gray-900">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
