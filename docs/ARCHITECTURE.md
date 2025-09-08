# System Architecture

## Overview
The Patient Deterioration ML System is a microservices-based application designed for predicting patient deterioration risk using machine learning models.

## Architecture Diagram
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Web     │    │   Mobile App    │    │   EHR System    │
│   Dashboard     │    │   (Future)      │    │   Integration   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      Load Balancer        │
                    │      (Nginx/Ingress)      │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      FastAPI Backend      │
                    │   (Authentication, API)   │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────┴────────┐    ┌──────────┴──────────┐    ┌─────────┴────────┐
│   PostgreSQL   │    │    Redis Cache      │    │   MLflow Model   │
│   Database     │    │   (Sessions, etc)   │    │    Registry      │
└────────────────┘    └─────────────────────┘    └──────────────────┘
        │                         │                         │
        │              ┌──────────┴──────────┐              │
        │              │   Monitoring Stack  │              │
        │              │ (Prometheus/Grafana)│              │
        │              └─────────────────────┘              │
        │                                                   │
┌───────┴────────┐                                ┌─────────┴────────┐
│   Audit Logs   │                                │   Feature Store  │
│   (Compliance) │                                │   (Feast/Custom) │
└────────────────┘                                └──────────────────┘
```

## Components

### Frontend (React Dashboard)
- **Technology**: React 18, Tailwind CSS, React Query
- **Purpose**: Clinician-facing dashboard for risk assessment
- **Features**:
  - Patient cohort management
  - Individual patient risk profiles
  - Interactive explanations
  - Real-time alerts

### Backend API (FastAPI)
- **Technology**: FastAPI, Python 3.9+
- **Purpose**: Core API for predictions and data management
- **Features**:
  - JWT authentication
  - Role-based access control
  - ML model inference
  - HIPAA compliance
  - Audit logging

### Database (PostgreSQL)
- **Purpose**: Primary data storage
- **Schema**:
  - Patient demographics
  - Clinical data (vitals, labs, medications)
  - Predictions and outcomes
  - User management
  - Audit trails

### Cache (Redis)
- **Purpose**: Session management and caching
- **Features**:
  - JWT token storage
  - Prediction result caching
  - Rate limiting
  - Real-time notifications

### Model Registry (MLflow)
- **Purpose**: ML model lifecycle management
- **Features**:
  - Model versioning
  - Experiment tracking
  - Model deployment
  - Performance monitoring

### Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and alerting
- **Features**:
  - System health monitoring
  - Model performance tracking
  - Business metrics
  - Alerting and notifications

## Data Flow

### Prediction Request Flow
1. User authenticates via JWT
2. Frontend sends prediction request to API
3. API validates request and user permissions
4. Model registry loads appropriate model
5. Feature engineering pipeline processes data
6. ML model generates prediction
7. Explainability engine generates explanations
8. Results cached in Redis
9. Audit log entry created
10. Response returned to frontend

### Training Pipeline Flow
1. Data ingestion from EHR systems
2. Data validation and preprocessing
3. Feature engineering
4. Model training with cross-validation
5. Model evaluation and selection
6. Model registration in MLflow
7. Deployment to production

## Security Architecture

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- Patient-level access restrictions
- API rate limiting

### Data Protection
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- PII tokenization
- Secure key management

### Compliance
- HIPAA compliance
- GDPR compliance
- Audit logging
- Data retention policies

## Scalability Considerations

### Horizontal Scaling
- Stateless API design
- Load balancer distribution
- Database read replicas
- Redis clustering

### Performance Optimization
- Model result caching
- Database query optimization
- Async processing
- CDN for static assets

### High Availability
- Multi-zone deployment
- Database failover
- Health checks
- Circuit breakers

## Deployment Patterns

### Development
- Docker Compose
- Local development environment
- Hot reloading
- Debug logging

### Staging
- Kubernetes deployment
- Blue-green deployment
- Automated testing
- Performance testing

### Production
- Multi-region deployment
- Auto-scaling
- Monitoring and alerting
- Disaster recovery

## Integration Points

### EHR Systems
- FHIR R4 API integration
- HL7 message processing
- Real-time data sync
- Data mapping and validation

### External Services
- Identity providers (SAML/OAuth)
- Notification services
- Backup services
- Monitoring services
