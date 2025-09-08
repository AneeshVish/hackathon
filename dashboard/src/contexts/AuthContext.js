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
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      // Validate token and get user info
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
      // For demo purposes, simulate login
      const demoToken = 'demo_token_' + Date.now();
      
      localStorage.setItem('token', demoToken);
      setToken(demoToken);
      
      const mockUser = {
        id: 'demo_clinician_001',
        username: credentials.username || 'demo_clinician',
        role: 'clinician',
        name: 'Dr. Sarah Johnson',
        email: 'sarah.johnson@hospital.com',
        permissions: ['patient_access', 'cohort_access', 'submit_feedback'],
        clinic_access: ['DEMO_CLINIC_001', 'DEMO_CLINIC_002'],
        care_team_access: ['DEMO_TEAM_001', 'DEMO_TEAM_002']
      };
      
      setUser(mockUser);
      axios.defaults.headers.common['Authorization'] = `Bearer ${demoToken}`;
      
      return { success: true };
    } catch (error) {
      console.error('Login failed:', error);
      return { success: false, error: error.message };
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
    delete axios.defaults.headers.common['Authorization'];
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
