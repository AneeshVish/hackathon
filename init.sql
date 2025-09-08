-- Initialize Patient ML Database

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS patient_ml;

-- Use the database
\c patient_ml;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS patient_data;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS audit;

-- Patients table
CREATE TABLE IF NOT EXISTS patient_data.patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    mrn VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Demographics table
CREATE TABLE IF NOT EXISTS patient_data.demographics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) REFERENCES patient_data.patients(patient_id),
    insurance VARCHAR(100),
    primary_care_physician VARCHAR(255),
    emergency_contact VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vitals table
CREATE TABLE IF NOT EXISTS patient_data.vitals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) REFERENCES patient_data.patients(patient_id),
    timestamp TIMESTAMP NOT NULL,
    systolic_bp INTEGER,
    diastolic_bp INTEGER,
    heart_rate INTEGER,
    temperature DECIMAL(4,1),
    respiratory_rate INTEGER,
    oxygen_saturation INTEGER,
    weight DECIMAL(5,1),
    height DECIMAL(5,1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Lab results table
CREATE TABLE IF NOT EXISTS patient_data.lab_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) REFERENCES patient_data.patients(patient_id),
    timestamp TIMESTAMP NOT NULL,
    test_name VARCHAR(100) NOT NULL,
    value DECIMAL(10,3),
    unit VARCHAR(20),
    reference_range VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Medications table
CREATE TABLE IF NOT EXISTS patient_data.medications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) REFERENCES patient_data.patients(patient_id),
    medication_name VARCHAR(255) NOT NULL,
    dosage VARCHAR(100),
    frequency VARCHAR(100),
    start_date DATE,
    end_date DATE,
    adherence_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table
CREATE TABLE IF NOT EXISTS ml_models.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) REFERENCES patient_data.patients(patient_id),
    model_version VARCHAR(50) NOT NULL,
    risk_score DECIMAL(5,4) NOT NULL,
    risk_bucket VARCHAR(20) NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    features JSONB,
    explanation JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model metadata table
CREATE TABLE IF NOT EXISTS ml_models.model_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    training_date TIMESTAMP NOT NULL,
    performance_metrics JSONB,
    feature_importance JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feedback table
CREATE TABLE IF NOT EXISTS ml_models.feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID REFERENCES ml_models.predictions(id),
    patient_id VARCHAR(50) REFERENCES patient_data.patients(patient_id),
    feedback_type VARCHAR(50) NOT NULL,
    outcome VARCHAR(100),
    comments TEXT,
    clinician_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    permissions JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    integrity_hash VARCHAR(64)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_patients_mrn ON patient_data.patients(mrn);
CREATE INDEX IF NOT EXISTS idx_vitals_patient_timestamp ON patient_data.vitals(patient_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_labs_patient_timestamp ON patient_data.lab_results(patient_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_medications_patient ON patient_data.medications(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_patient ON ml_models.predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON ml_models.predictions(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_user_timestamp ON audit.audit_logs(user_id, timestamp);

-- Insert demo data
INSERT INTO users (username, email, password_hash, role, permissions) VALUES
('demo_clinician', 'demo@hospital.com', '$2b$12$demo_hash', 'clinician', '["patient_access", "cohort_access", "submit_feedback"]'),
('demo_admin', 'admin@hospital.com', '$2b$12$demo_hash', 'admin', '["patient_access", "cohort_access", "submit_feedback", "admin_access"]')
ON CONFLICT (username) DO NOTHING;

-- Insert demo patients
INSERT INTO patient_data.patients (patient_id, mrn, name, date_of_birth, gender) VALUES
('DEMO_PATIENT_001', 'MRN123456', 'John Smith', '1957-03-15', 'M'),
('DEMO_PATIENT_002', 'MRN123457', 'Mary Johnson', '1970-08-22', 'F'),
('DEMO_PATIENT_003', 'MRN123458', 'Robert Davis', '1953-11-10', 'M'),
('DEMO_PATIENT_004', 'MRN123459', 'Sarah Wilson', '1979-05-18', 'F'),
('DEMO_PATIENT_005', 'MRN123460', 'Michael Brown', '1962-12-03', 'M')
ON CONFLICT (patient_id) DO NOTHING;
