import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API endpoints
export const authAPI = {
  login: (credentials) => api.post('/auth/login', credentials),
  logout: () => api.post('/auth/logout'),
  validateToken: () => api.get('/auth/validate'),
};

export const patientAPI = {
  getCohort: (params) => api.get('/cohort', { params }),
  getPatient: (patientId) => api.get(`/patient/${patientId}`),
  getPatientFeatures: (patientId) => api.get(`/patient/${patientId}/features`),
  predictSingle: (patientData) => api.post('/predict/single', patientData),
  predictBatch: (patientIds) => api.post('/predict/batch', { patient_ids: patientIds }),
  submitFeedback: (feedback) => api.post('/feedback', feedback),
};

export const modelAPI = {
  getModelInfo: () => api.get('/model/info'),
  getMetrics: () => api.get('/metrics'),
  getHealth: () => api.get('/health'),
};

export default api;
