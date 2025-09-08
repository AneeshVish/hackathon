# Patient Deterioration Prediction ML Service

A secure, explainable machine learning service that predicts patient deterioration risk within 90 days using longitudinal patient data.

## Overview

This system consumes 30-180 days of patient data (vitals, labs, medications, adherence, lifestyle) and outputs:
- Probability of deterioration in next 90 days
- Clinician-friendly explanations
- Recommended actions
- Web dashboard for cohort and patient-level insights

## Architecture

```
├── data/                   # Data processing and ETL
├── models/                 # ML model training and evaluation
├── inference/              # FastAPI service for predictions
├── dashboard/              # React frontend
├── infrastructure/         # Docker, K8s, monitoring
├── tests/                  # Unit and integration tests
└── docs/                   # Documentation and API specs
```

## Key Features

- **Clinical Validation**: AUROC, AUPRC, calibration metrics with subgroup analysis
- **Explainability**: SHAP-based global and local explanations
- **Security**: HIPAA/GDPR compliance, encryption, audit trails
- **Scalability**: Kubernetes deployment with monitoring
- **Integration**: FHIR/HL7 APIs for EHR integration

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment: `cp .env.example .env`
3. Run data pipeline: `python -m data.pipeline`
4. Train models: `python -m models.train`
5. Start API service: `uvicorn inference.main:app --reload`
6. Launch dashboard: `cd dashboard && npm start`

## Clinical Workflow

1. **Cohort View**: Risk-stratified patient list with filters
2. **Patient Detail**: Trends, explanations, recommended actions
3. **Integration**: EHR integration via FHIR APIs
4. **Monitoring**: Continuous model performance tracking

## Compliance & Security

- Data encryption at rest and in transit
- Role-based access control (RBAC)
- Audit logging for all predictions
- De-identification for model training
- Consent management

## Development

See `docs/development.md` for detailed setup and contribution guidelines.
