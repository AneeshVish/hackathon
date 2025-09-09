import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState(localStorage.getItem('token'));

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch (e) {
        console.error('Failed to parse user data', e);
        localStorage.removeItem('user');
      }
    }
    
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      validateToken();
    } else {
      setLoading(false);
    }
  }, [token]);

  const validateToken = async () => {
    try {
      // For demo purposes, create a mock user from token
      const mockUser = {
        id: 'demo_clinician_001',
        username: 'demo_clinician',
        role: 'clinician',
        name: 'Dr. Sarah Johnson',
        email: 'sarah.johnson@hospital.com',
        permissions: ['patient_access', 'cohort_access', 'submit_feedback'],
        clinic_access: ['DEMO_CLINIC_001', 'DEMO_CLINIC_002'],
        care_team_access: ['DEMO_TEAM_001', 'DEMO_TEAM_002']
      };
      
      setUser(mockUser);
    } catch (error) {
      console.error('Token validation failed:', error);
      logout();
    } finally {
      setLoading(false);
    }
  };

  const login = async (credentials) => {
    try {
      // In a real app, you would make an API call here
      // const response = await axios.post('/api/auth/login', credentials);
      const mockUser = {
        id: 'demo_clinician_001',
        username: credentials.username,
        role: 'clinician',
        name: credentials.username === 'demo' ? 'Demo Clinician' : 'Dr. Sarah Johnson',
        email: `${credentials.username}@example.com`,
        permissions: ['patient_access', 'cohort_access', 'submit_feedback'],
        clinic_access: ['DEMO_CLINIC_001', 'DEMO_CLINIC_002'],
        care_team_access: ['DEMO_TEAM_001', 'DEMO_TEAM_002']
      };
      
      const mockToken = 'demo_token_' + Math.random().toString(36).substring(2);
      
      localStorage.setItem('token', mockToken);
      localStorage.setItem('user', JSON.stringify(mockUser));
      
      setUser(mockUser);
      setToken(mockToken);
      axios.defaults.headers.common['Authorization'] = `Bearer ${mockToken}`;
      
      return mockUser;
    } catch (error) {
      console.error('Login failed:', error);
      throw new Error('Login failed. Please check your credentials and try again.');
    }
  };
  
  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    delete axios.defaults.headers.common['Authorization'];
    setUser(null);
    setToken(null);
  };

  const value = {
    user,
    login,
    logout,
    loading,
    isAuthenticated: !!user,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
