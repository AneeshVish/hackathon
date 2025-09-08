import React from 'react';
import { useQuery } from 'react-query';
import {
  ChartBarIcon,
  UserGroupIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const Dashboard = () => {
  // Mock data for dashboard
  const stats = [
    {
      name: 'Total Patients',
      value: '1,247',
      change: '+4.75%',
      changeType: 'positive',
      icon: UserGroupIcon,
    },
    {
      name: 'High Risk',
      value: '89',
      change: '+12.5%',
      changeType: 'negative',
      icon: ExclamationTriangleIcon,
    },
    {
      name: 'Medium Risk',
      value: '234',
      change: '-2.1%',
      changeType: 'positive',
      icon: ChartBarIcon,
    },
    {
      name: 'Low Risk',
      value: '924',
      change: '+1.2%',
      changeType: 'positive',
      icon: CheckCircleIcon,
    },
  ];

  const riskTrendData = [
    { month: 'Jan', high: 65, medium: 180, low: 800 },
    { month: 'Feb', high: 72, medium: 195, low: 820 },
    { month: 'Mar', high: 78, medium: 210, low: 850 },
    { month: 'Apr', high: 85, medium: 225, low: 880 },
    { month: 'May', high: 89, medium: 234, low: 924 },
  ];

  const riskDistribution = [
    { name: 'Low Risk', value: 924, color: '#22c55e' },
    { name: 'Medium Risk', value: 234, color: '#f59e0b' },
    { name: 'High Risk', value: 89, color: '#ef4444' },
  ];

  const recentAlerts = [
    {
      id: 1,
      patient: 'John Smith',
      patientId: 'DEMO_PATIENT_001',
      risk: 'High',
      reason: 'Elevated BNP, recent weight gain',
      time: '2 hours ago',
      urgent: true,
    },
    {
      id: 2,
      patient: 'Mary Johnson',
      patientId: 'DEMO_PATIENT_002',
      risk: 'Medium',
      reason: 'Medication non-adherence',
      time: '4 hours ago',
      urgent: false,
    },
    {
      id: 3,
      patient: 'Robert Davis',
      patientId: 'DEMO_PATIENT_003',
      risk: 'High',
      reason: 'Multiple ED visits, poor adherence',
      time: '6 hours ago',
      urgent: true,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Overview of patient deterioration risk predictions
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div
            key={stat.name}
            className="relative overflow-hidden rounded-lg bg-white px-4 py-5 shadow sm:px-6 sm:py-6"
          >
            <dt>
              <div className="absolute rounded-md bg-primary-500 p-3">
                <stat.icon className="h-6 w-6 text-white" aria-hidden="true" />
              </div>
              <p className="ml-16 truncate text-sm font-medium text-gray-500">
                {stat.name}
              </p>
            </dt>
            <dd className="ml-16 flex items-baseline">
              <p className="text-2xl font-semibold text-gray-900">{stat.value}</p>
              <p
                className={`ml-2 flex items-baseline text-sm font-semibold ${
                  stat.changeType === 'positive'
                    ? 'text-green-600'
                    : 'text-red-600'
                }`}
              >
                {stat.change}
              </p>
            </dd>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Trend Chart */}
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
              Risk Trend Over Time
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={riskTrendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="low" stackId="a" fill="#22c55e" name="Low Risk" />
                  <Bar dataKey="medium" stackId="a" fill="#f59e0b" name="Medium Risk" />
                  <Bar dataKey="high" stackId="a" fill="#ef4444" name="High Risk" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Risk Distribution */}
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
              Current Risk Distribution
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {riskDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Alerts */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        <div className="px-4 py-5 sm:px-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900">
            Recent High-Risk Alerts
          </h3>
          <p className="mt-1 max-w-2xl text-sm text-gray-500">
            Patients requiring immediate attention
          </p>
        </div>
        <ul className="divide-y divide-gray-200">
          {recentAlerts.map((alert) => (
            <li key={alert.id}>
              <div className="px-4 py-4 flex items-center justify-between">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div
                      className={`h-3 w-3 rounded-full ${
                        alert.urgent ? 'bg-red-400' : 'bg-yellow-400'
                      }`}
                    />
                  </div>
                  <div className="ml-4">
                    <div className="flex items-center">
                      <p className="text-sm font-medium text-gray-900">
                        {alert.patient}
                      </p>
                      <span
                        className={`ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          alert.risk === 'High'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-yellow-100 text-yellow-800'
                        }`}
                      >
                        {alert.risk} Risk
                      </span>
                    </div>
                    <p className="text-sm text-gray-500">{alert.reason}</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <p className="text-sm text-gray-500">{alert.time}</p>
                  <button className="ml-4 text-primary-600 hover:text-primary-900 text-sm font-medium">
                    View Details
                  </button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Dashboard;
