import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { Link } from 'react-router-dom';
import {
  FunnelIcon,
  MagnifyingGlassIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import axios from 'axios';

const CohortView = () => {
  const [filters, setFilters] = useState({
    riskBucket: '',
    clinicId: '',
    searchTerm: '',
  });

  // Mock data for cohort
  const mockCohortData = {
    patients: [
      {
        patient_id: 'DEMO_PATIENT_001',
        name: 'John Smith',
        age: 67,
        last_visit: '2025-01-05T10:30:00Z',
        risk_score: 0.72,
        risk_bucket: 'High',
        top_driver: 'Elevated BNP',
      },
      {
        patient_id: 'DEMO_PATIENT_002',
        name: 'Mary Johnson',
        age: 54,
        last_visit: '2025-01-04T14:15:00Z',
        risk_score: 0.23,
        risk_bucket: 'Medium',
        top_driver: 'Poor adherence',
      },
      {
        patient_id: 'DEMO_PATIENT_003',
        name: 'Robert Davis',
        age: 71,
        last_visit: '2025-01-03T09:45:00Z',
        risk_score: 0.68,
        risk_bucket: 'High',
        top_driver: 'Recent ED visit',
      },
      {
        patient_id: 'DEMO_PATIENT_004',
        name: 'Sarah Wilson',
        age: 45,
        last_visit: '2025-01-02T16:20:00Z',
        risk_score: 0.08,
        risk_bucket: 'Low',
        top_driver: 'Stable vitals',
      },
      {
        patient_id: 'DEMO_PATIENT_005',
        name: 'Michael Brown',
        age: 62,
        last_visit: '2025-01-01T11:00:00Z',
        risk_score: 0.45,
        risk_bucket: 'Medium',
        top_driver: 'Weight gain',
      },
    ],
    total_count: 5,
    risk_distribution: {
      Low: 1,
      Medium: 2,
      High: 2,
    },
  };

  const { data: cohortData, isLoading, error } = useQuery(
    ['cohort', filters],
    () => {
      // In a real app, this would make an API call
      // return axios.get('/cohort', { params: filters });
      return Promise.resolve({ data: mockCohortData });
    },
    {
      keepPreviousData: true,
    }
  );

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
      <div className="flex items-center justify-center h-64">
        <div className="spinner"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">Error loading cohort data</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="sm:flex sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Patient Cohort</h1>
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
                  id="search"
                  className="focus:ring-primary-500 focus:border-primary-500 block w-full pl-10 sm:text-sm border-gray-300 rounded-md"
                  placeholder="Search by name or ID..."
                  value={filters.searchTerm}
                  onChange={(e) => setFilters({ ...filters, searchTerm: e.target.value })}
                />
              </div>
            </div>

            <div>
              <label htmlFor="risk-filter" className="block text-sm font-medium text-gray-700">
                Risk Level
              </label>
              <select
                id="risk-filter"
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
                value={filters.riskBucket}
                onChange={(e) => setFilters({ ...filters, riskBucket: e.target.value })}
              >
                <option value="">All Risk Levels</option>
                <option value="High">High Risk</option>
                <option value="Medium">Medium Risk</option>
                <option value="Low">Low Risk</option>
              </select>
            </div>

            <div>
              <label htmlFor="clinic-filter" className="block text-sm font-medium text-gray-700">
                Clinic
              </label>
              <select
                id="clinic-filter"
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
                value={filters.clinicId}
                onChange={(e) => setFilters({ ...filters, clinicId: e.target.value })}
              >
                <option value="">All Clinics</option>
                <option value="DEMO_CLINIC_001">Demo Clinic 001</option>
                <option value="DEMO_CLINIC_002">Demo Clinic 002</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Patient Table */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
          <h3 className="text-lg leading-6 font-medium text-gray-900">
            Patients ({filteredPatients.length})
          </h3>
        </div>
        <ul className="divide-y divide-gray-200">
          {filteredPatients.map((patient) => (
            <li key={patient.patient_id}>
              <Link
                to={`/patient/${patient.patient_id}`}
                className="block hover:bg-gray-50 transition-colors duration-150"
              >
                <div className="px-4 py-4 sm:px-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div className="flex-shrink-0">
                        <div className="h-10 w-10 bg-gray-300 rounded-full flex items-center justify-center">
                          <span className="text-gray-600 font-medium text-sm">
                            {patient.name.charAt(0)}
                          </span>
                        </div>
                      </div>
                      <div className="ml-4">
                        <div className="flex items-center">
                          <p className="text-sm font-medium text-gray-900">
                            {patient.name}
                          </p>
                          <span
                            className={`ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getRiskBadgeColor(
                              patient.risk_bucket
                            )}`}
                          >
                            {getRiskIcon(patient.risk_bucket)}
                            <span className="ml-1">{patient.risk_bucket}</span>
                          </span>
                        </div>
                        <div className="mt-1 flex items-center text-sm text-gray-500">
                          <p>Age {patient.age} • ID: {patient.patient_id}</p>
                          <span className="mx-2">•</span>
                          <p>Risk: {(patient.risk_score * 100).toFixed(0)}%</p>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className="text-sm text-gray-900">{patient.top_driver}</p>
                        <p className="text-sm text-gray-500">
                          Last visit: {formatDate(patient.last_visit)}
                        </p>
                      </div>
                      <div className="flex-shrink-0">
                        <svg
                          className="h-5 w-5 text-gray-400"
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fillRule="evenodd"
                            d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      </div>
                    </div>
                  </div>
                </div>
              </Link>
            </li>
          ))}
        </ul>

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
