import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { Link } from 'react-router-dom';
import {
  FunnelIcon,
  MagnifyingGlassIcon,
  ExclamationCircleIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ChevronDownIcon,
  ArrowPathIcon,
  UserGroupIcon,
  ShieldCheckIcon,
  UserCircleIcon
} from '@heroicons/react/24/outline';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { trainedPatientPredictions, trainedRiskDistribution, trainedPatientStats } from '../data/trainedModelData';

// Custom Components
const RiskBadge = ({ riskLevel }) => {
  const styles = {
    High: 'bg-red-100 text-red-800 border-red-200',
    Medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    Low: 'bg-green-100 text-green-800 border-green-200'
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${styles[riskLevel] || 'bg-gray-100 text-gray-800'}`}>
      {riskLevel} Risk
    </span>
  );
};

const StatusBadge = ({ needsAttention }) => (
  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${needsAttention ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
    {needsAttention ? 'Needs Attention' : 'Stable'}
  </span>
);

const COLORS = {
  High: '#EF4444',
  Medium: '#F59E0B',
  Low: '#10B981'
};

const CohortView = () => {
  const [filters, setFilters] = useState({
    riskBucket: '',
    searchTerm: ''
  });

  // Process data for charts - using patientData as the source of truth
  const { data: cohortData, isLoading, error, refetch } = useQuery(
    ['cohort', filters],
    () => {
      // Import patientData directly
      const patientData = require('../data/patientData').patientData;
      
      // Filter patients based on search term and risk level
      const filteredPatients = patientData.filter(patient => {
        const matchesSearch = !filters.searchTerm || 
          patient.name.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
          patient.patient_id.toLowerCase().includes(filters.searchTerm.toLowerCase());
        const matchesRisk = !filters.riskBucket || patient.risk_category === filters.riskBucket.toLowerCase();
        return matchesSearch && matchesRisk;
      });

      // Calculate risk distribution
      const riskDistribution = {
        low: filteredPatients.filter(p => p.risk_category === 'low').length,
        medium: filteredPatients.filter(p => p.risk_category === 'medium').length,
        high: filteredPatients.filter(p => p.risk_category === 'high').length
      };

      // Map patient data to match expected format
      const mappedPatients = filteredPatients.map(patient => ({
        ...patient,
        risk_bucket: patient.risk_category.charAt(0).toUpperCase() + patient.risk_category.slice(1), // Capitalize first letter
        risk_score: patient.risk_score || 0.5, // Default risk score if not present
        last_visit: patient.last_visit || new Date().toISOString(),
        needs_intervention: patient.risk_category === 'high' // Example logic for intervention
      }));

      return Promise.resolve({
        data: {
          patients: mappedPatients,
          total_count: mappedPatients.length,
          risk_distribution: riskDistribution
        }
      });
    },
    {
      keepPreviousData: true,
    }
  );

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters(prev => ({
      ...prev,
      [name]: value
    }));
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

  const getRiskIcon = (riskBucket) => {
    switch (riskBucket) {
      case 'High':
        return <ExclamationTriangleIcon className="h-4 w-4" />;
      case 'Medium':
        return <ClockIcon className="h-4 w-4" />;
      case 'Low':
        return <CheckCircleIcon className="h-4 w-4" />;
      default:
        return null;
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const filteredPatients = cohortData?.data?.patients?.filter((patient) => {
    const matchesRisk = !filters.riskBucket || patient.risk_bucket === filters.riskBucket;
    const matchesSearch = !filters.searchTerm || 
      patient.name.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
      patient.patient_id.toLowerCase().includes(filters.searchTerm.toLowerCase());
    
    return matchesRisk && matchesSearch;
  }) || [];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md bg-red-50 p-4 my-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <ExclamationCircleIcon className="h-5 w-5 text-red-400" />
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error loading cohort data</h3>
            <div className="mt-2 text-sm text-red-700">
              <p>Failed to load patient data. Please try again later.</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Risk distribution data for charts
  const riskDistributionData = [
    { name: 'High', value: cohortData?.data?.risk_distribution?.high || 0, color: COLORS.High },
    { name: 'Medium', value: cohortData?.data?.risk_distribution?.medium || 0, color: COLORS.Medium },
    { name: 'Low', value: cohortData?.data?.risk_distribution?.low || 0, color: COLORS.Low }
  ];

  return (
    <div className="space-y-6 px-4 py-6 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="sm:flex sm:items-center sm:justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Patient Risk Analysis</h1>
          <p className="mt-1 text-sm text-gray-500">
            Monitor and manage patient risk assessments
          </p>
          <p className="mt-1 text-sm text-gray-500">
            Monitor and manage patient deterioration risk across your cohort
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <button
            type="button"
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          >
            Export Report
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="h-8 w-8 bg-gray-500 rounded-full flex items-center justify-center">
                  <span className="text-white text-sm font-medium">
                    {cohortData?.data?.total_count || 0}
                  </span>
                </div>
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Total Patients
                  </dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {cohortData?.data?.total_count || 0}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        {Object.entries(cohortData?.data?.risk_distribution || {}).map(([risk, count]) => (
          <div key={risk} className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className={`h-8 w-8 rounded-full flex items-center justify-center ${
                    risk === 'High' ? 'bg-red-500' : 
                    risk === 'Medium' ? 'bg-yellow-500' : 'bg-green-500'
                  }`}>
                    {getRiskIcon(risk)}
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {risk} Risk
                    </dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {count}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4 mb-6">
        {riskDistributionData.map((item) => (
          <div key={item.name} className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center">
                <div 
                  className="flex-shrink-0 rounded-md p-3" 
                  style={{ backgroundColor: `${item.color}20` }}
                >
                  <ShieldCheckIcon 
                    className="h-6 w-6" 
                    style={{ color: item.color }} 
                    aria-hidden="true" 
                  />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {item.name} Risk
                    </dt>
                    <dd className="flex items-baseline">
                      <div className="text-2xl font-semibold text-gray-900">
                        {item.value}
                      </div>
                      <div className="ml-2 flex items-baseline text-sm font-semibold text-green-600">
                        {cohortData?.data?.total_count > 0 
                          ? Math.round((item.value / cohortData.data.total_count) * 100) 
                          : 0}%
                      </div>
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div>
              <label htmlFor="search" className="block text-sm font-medium text-gray-700">
                Search Patients
              </label>
              <div className="mt-1 relative rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="text"
                  name="searchTerm"
                  id="search"
                  className="focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 sm:text-sm border-gray-300 rounded-md h-10"
                  placeholder="Search by name or ID..."
                  value={filters.searchTerm}
                  onChange={handleFilterChange}
                />
              </div>
            </div>

            <div>
              <label htmlFor="risk-filter" className="block text-sm font-medium text-gray-700">
                Risk Level
              </label>
              <select
                id="risk-filter"
                name="riskBucket"
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md h-10"
                value={filters.riskBucket}
                onChange={handleFilterChange}
              >
                <option value="">All Risk Levels</option>
                <option value="High">High Risk</option>
                <option value="Medium">Medium Risk</option>
                <option value="Low">Low Risk</option>
              </select>
            </div>

            <div className="flex items-end">
              <button
                type="button"
                onClick={() => refetch()}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 h-10"
              >
                <ArrowPathIcon className="-ml-1 mr-2 h-4 w-4" />
                Refresh Data
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Patient Table */}
      <div className="bg-white shadow overflow-hidden sm:rounded-lg">
        <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-medium text-gray-900">
                Patient List
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                {filteredPatients.length} patients found
                {filters.riskBucket ? ` (${filters.riskBucket} Risk)` : ''}
              </p>
            </div>
            <div className="flex space-x-3">
              <button
                type="button"
                className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <svg className="-ml-1 mr-1.5 h-4 w-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export
              </button>
            </div>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Patient
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Risk Level
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Last Visit
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th scope="col" className="relative px-6 py-3">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredPatients.map((patient) => (
                <tr key={patient.patient_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="flex-shrink-0 h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
                        <span className="text-blue-800 font-medium">
                          {patient.name?.charAt(0) || 'P'}
                        </span>
                      </div>
                      <div className="ml-4">
                        <div className="text-sm font-medium text-gray-900">
                          {patient.name || 'Patient Name'}
                        </div>
                        <div className="text-sm text-gray-500">
                          ID: {patient.patient_id}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex flex-col">
                      <RiskBadge riskLevel={patient.risk_bucket} />
                      <span className="mt-1 text-xs text-gray-500">
                        Score: {Math.round(patient.risk_score * 100)}/100
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">
                      {formatDate(patient.last_visit) || 'N/A'}
                    </div>
                    <div className="text-xs text-gray-500">
                      {patient.days_since_last_visit ? `${patient.days_since_last_visit} days ago` : ''}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <StatusBadge needsAttention={patient.needs_intervention} />
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <Link
                      to={`/patient/${patient.patient_id}`}
                      className="text-blue-600 hover:text-blue-900"
                    >
                      View Details
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredPatients.length === 0 && (
          <div className="text-center py-12">
            <FunnelIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No patients found</h3>
            <p className="mt-1 text-sm text-gray-500">
              Try adjusting your search criteria or filters.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default CohortView;
