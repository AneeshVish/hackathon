export const RISK_LEVELS = {
  HIGH: 'High',
  MEDIUM: 'Medium',
  LOW: 'Low',
};

export const RISK_COLORS = {
  [RISK_LEVELS.HIGH]: {
    bg: 'bg-red-100',
    text: 'text-red-800',
    border: 'border-red-200',
    dot: 'bg-red-400',
  },
  [RISK_LEVELS.MEDIUM]: {
    bg: 'bg-yellow-100',
    text: 'text-yellow-800',
    border: 'border-yellow-200',
    dot: 'bg-yellow-400',
  },
  [RISK_LEVELS.LOW]: {
    bg: 'bg-green-100',
    text: 'text-green-800',
    border: 'border-green-200',
    dot: 'bg-green-400',
  },
};

export const PRIORITY_COLORS = {
  high: 'text-red-600 bg-red-50',
  medium: 'text-yellow-600 bg-yellow-50',
  low: 'text-green-600 bg-green-50',
};

export const USER_ROLES = {
  CLINICIAN: 'clinician',
  ADMIN: 'admin',
  RESEARCHER: 'researcher',
};

export const PERMISSIONS = {
  PATIENT_ACCESS: 'patient_access',
  COHORT_ACCESS: 'cohort_access',
  SUBMIT_FEEDBACK: 'submit_feedback',
  ADMIN_ACCESS: 'admin_access',
};

export const CHART_COLORS = {
  primary: '#3b82f6',
  secondary: '#10b981',
  danger: '#ef4444',
  warning: '#f59e0b',
  info: '#06b6d4',
  success: '#22c55e',
};
