# Patient Deterioration Prediction ML Service

A secure, explainable ML service that predicts patient deterioration risk within 90 days using longitudinal patient data.

## Overview

This system consumes 30-180 days of patient data (vitals, labs, medications, adherence, lifestyle) and outputs:
- Probability of deterioration in next 90 days
- Clinician-friendly explanations
- Recommended actions
- Web dashboard and APIs

## Architecture

```
Data Sources → Ingestion & ETL → Feature Store → ML Models → Inference API → Dashboard
     ↓              ↓               ↓            ↓           ↓            ↓
   EHR/Labs    Kafka/Airflow    Feast/Redis   MLflow    FastAPI      React UI
```

## Key Features

- **Clinical Validation**: AUROC/AUPRC metrics, calibration, subgroup analysis
- **Explainability**: SHAP-based global and local explanations
- **Privacy**: HIPAA/GDPR compliance, encryption, audit trails
- **Deployment**: Kubernetes, CI/CD, monitoring, rollback capabilities

## Quick Start

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Data Pipeline**
   ```bash
   python src/data/pipeline.py
   ```

3. **Train Models**
   ```bash
   python src/models/train.py
   ```

4. **Start API Service**
   ```bash
   uvicorn src.api.main:app --reload
   ```

5. **Launch Dashboard**
   ```bash
   cd frontend && npm start
   ```

## Project Structure

```
├── src/
│   ├── data/           # Data ingestion, cleaning, feature engineering
│   ├── models/         # ML training, evaluation, registry
│   ├── api/           # FastAPI inference service
│   ├── explainability/ # SHAP integration, explanations
│   └── monitoring/     # Model drift, performance tracking
├── frontend/          # React dashboard
├── infrastructure/    # Kubernetes, Docker, CI/CD
├── notebooks/         # Jupyter notebooks for exploration
└── tests/            # Unit and integration tests
```

## Clinical Workflow

1. **Cohort View**: Risk-stratified patient list with filters
2. **Patient Detail**: Trends, drivers, recommended actions
3. **Explainability**: "Why is this patient at risk?"
4. **Actions**: Schedule visits, medication reviews, alerts

## Compliance & Security

- Data encryption at rest and in transit
- Role-based access control (RBAC)
- Audit logging for all predictions
- De-identification for model training
- Consent management

## Validation Plan

1. **Retrospective**: Historical EHR validation
2. **Silent Prospective**: Live predictions without clinical action
3. **Pilot**: Limited clinic deployment with outcome tracking
4. **Full Deployment**: Continuous monitoring and re-calibration

## Team & Timeline

- **Duration**: 16 weeks (8 sprints × 2 weeks)
- **Team**: 8-10 members (ML, backend, frontend, clinical, DevOps)
- **Deliverables**: Working system, evaluation report, documentation

## License

Proprietary - Healthcare AI System
