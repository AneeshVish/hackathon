import React, { useState, useMemo, useEffect } from 'react';
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
  UserCircleIcon,
  PhoneIcon,
  EnvelopeIcon,
  MapPinIcon,
  CalendarIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getProcessedPatientData, getProcessedRiskDistribution, getProcessedPatientStats } from '../data/patientDataAdapter';

// Custom Components
const RiskBadge = ({ riskLevel }) => {
  const styles = {
    high: 'bg-red-100 text-red-800 border-red-200',
    medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    low: 'bg-green-100 text-green-800 border-green-200'
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${styles[riskLevel?.toLowerCase()] || ''}`}>
      {riskLevel}
    </span>
  );
};

const StatusBadge = ({ needsAttention }) => (
  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${needsAttention ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
    {needsAttention ? 'Needs Attention' : 'Stable'}
  </span>
);

const COLORS = {
  high: '#EF4444',
  medium: '#F59E0B',
  low: '#10B981'
};

const CohortView = () => {
  // State for patient data and filters
  const [patients, setPatients] = useState([]);
  const [riskDistribution, setRiskDistribution] = useState([]);
  const [patientStats, setPatientStats] = useState({});
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState({
    riskLevel: 'all',
    needsAttention: false,
    recentActivity: false
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load and process patient data on component mount
  useEffect(() => {
    try {
      const processedPatients = getProcessedPatientData();
      setPatients(processedPatients);
      setRiskDistribution(getProcessedRiskDistribution(processedPatients));
      setPatientStats(getProcessedPatientStats(processedPatients));
      setIsLoading(false);
    } catch (err) {
      setError(err.message);
      setIsLoading(false);
    }
  }, []);

  // Filter and sort patients based on search and filters
  const filteredPatients = useMemo(() => {
    if (!patients.length) return [];
    
    return patients.filter(patient => {
      const searchTermLower = searchTerm.toLowerCase();
      const matchesSearch = !searchTerm || 
        (patient.name && patient.name.toLowerCase().includes(searchTermLower)) ||
        (patient.patient_id && patient.patient_id.toLowerCase().includes(searchTermLower)) ||
        (patient.mrn && patient.mrn.toLowerCase().includes(searchTermLower));
        
      const matchesRisk = filters.riskLevel === 'all' || 
        (patient.risk_bucket && patient.risk_bucket.toLowerCase() === filters.riskLevel.toLowerCase());
        
      const needsAttention = !filters.needsAttention || 
        (patient.key_risk_factors && patient.key_risk_factors.length > 0);
        
      return matchesSearch && matchesRisk && needsAttention;
    }).sort((a, b) => b.risk_score - a.risk_score);
  }, [patients, searchTerm, filters]);

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return isNaN(date.getTime()) 
      ? 'Invalid date' 
      : date.toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        });
  };

  const handleFilterChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFilters(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

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
            <ExclamationCircleIcon className="h-5 w-5 text-red-400" aria-hidden="true" />
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error loading patient data</h3>
            <div className="mt-2 text-sm text-red-700">
              <p>{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-6">
      {/* Header with search and filters */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 space-y-4 md:space-y-0">
        <h1 className="text-2xl font-bold text-gray-900">Patient Cohort</h1>
        <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4 w-full md:w-auto">
          <div className="relative flex-grow max-w-md">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              placeholder="Search patients..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="flex space-x-4">
            <select
              name="riskLevel"
              className="block w-full pl-3 pr-10 py-2 text-base border border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
              value={filters.riskLevel}
              onChange={handleFilterChange}
            >
              <option value="all">All Risk Levels</option>
              <option value="high">High Risk</option>
              <option value="medium">Medium Risk</option>
              <option value="low">Low Risk</option>
            </select>
            <button
              type="button"
              className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              onClick={() => {
                setSearchTerm('');
                setFilters({
                  riskLevel: 'all',
                  needsAttention: false,
                  recentActivity: false
                });
              }}
            >
              <ArrowPathIcon className="-ml-1 mr-2 h-4 w-4" />
              Reset Filters
            </button>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4 mb-6">
        {riskDistribution.map((item) => (
          <div key={item.name} className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex items-center">
                <div 
                  className="flex-shrink-0 rounded-md p-3" 
                  style={{ backgroundColor: `${COLORS[item.name.toLowerCase()]}20` }}
                >
                  <ShieldCheckIcon 
                    className="h-6 w-6" 
                    style={{ color: COLORS[item.name.toLowerCase()] }} 
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
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Patient List */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        <ul className="divide-y divide-gray-200">
          {filteredPatients.length > 0 ? (
            filteredPatients.map((patient) => (
              <li key={patient.patient_id}>
                <Link to={`/patient/${patient.patient_id}`} className="block hover:bg-gray-50">
                  <div className="px-4 py-4 sm:px-6">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="flex-shrink-0">
                          <UserCircleIcon className="h-10 w-10 text-gray-400" aria-hidden="true" />
                        </div>
                        <div className="ml-4">
                          <div className="flex items-center text-sm font-medium text-blue-600 truncate">
                            {patient.name}
                            <span className="ml-2 text-gray-500 text-xs">
                              {patient.mrn}
                            </span>
                          </div>
                          <div className="flex mt-1">
                            <RiskBadge riskLevel={patient.risk_bucket} />
                            <StatusBadge needsAttention={patient.key_risk_factors?.length > 0} className="ml-2" />
                          </div>
                        </div>
                      </div>
                      <div className="ml-2 flex-shrink-0 flex">
                        <div className="text-right">
                          <p className="text-sm text-gray-500">
                            Risk Score: <span className="font-medium">
                              {(patient.risk_score * 100).toFixed(0)}%
                            </span>
                          </p>
                          <p className="text-xs text-gray-400">
                            Last updated: {formatDate(patient.last_updated)}
                          </p>
                        </div>
                        <ChevronDownIcon className="ml-2 h-5 w-5 text-gray-400" aria-hidden="true" />
                      </div>
                    </div>
                  </div>
                </Link>
              </li>
            ))
          ) : (
            <li className="py-8 text-center text-gray-500">
              No patients found matching your criteria.
            </li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default CohortView;