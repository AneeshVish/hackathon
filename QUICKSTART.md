# 🚀 Quick Start Guide

## System Overview
Complete Patient Deterioration ML System with:
- ✅ FastAPI Backend with ML predictions
- ✅ React Dashboard with modern UI
- ✅ Docker deployment ready
- ✅ Kubernetes manifests
- ✅ Monitoring & logging
- ✅ HIPAA compliance

## 🏃‍♂️ Quick Start (5 minutes)

### 1. Setup Environment
```bash
# Copy environment file
cp .env.example .env

# Make scripts executable (Linux/Mac)
chmod +x scripts/setup.sh scripts/deploy.sh

# Run setup script
./scripts/setup.sh
```

### 2. Start All Services
```bash
docker-compose up -d
```

### 3. Access Applications
- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090

### 4. Demo Login
- **Username**: `demo_clinician`
- **Password**: `demo123`

## 📁 Project Structure
```
hackathon/
├── dashboard/          # React frontend
│   ├── src/
│   │   ├── components/ # UI components
│   │   ├── pages/      # Main pages
│   │   ├── contexts/   # React contexts
│   │   └── utils/      # Helper functions
│   └── public/         # Static assets
├── data/               # Data processing
│   ├── schemas.py      # Data models
│   ├── pipeline.py     # ETL pipeline
│   ├── preprocessing.py # Feature engineering
│   └── validation.py   # Data validation
├── inference/          # API service
│   ├── main.py         # FastAPI app
│   ├── security.py     # Authentication
│   ├── model_registry.py # Model management
│   ├── monitoring.py   # System monitoring
│   └── audit.py        # Compliance logging
├── models/             # ML models
│   ├── train.py        # Model training
│   ├── evaluation.py   # Model evaluation
│   └── explainability.py # SHAP explanations
├── k8s/                # Kubernetes manifests
├── monitoring/         # Prometheus/Grafana
├── scripts/            # Deployment scripts
├── tests/              # Unit & integration tests
└── docs/               # Documentation
```

## 🔧 Development Commands

### Backend Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run backend locally
cd inference
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
# Install dependencies
cd dashboard
npm install

# Run frontend locally
npm start
```

### Training Models
```bash
# Train ML models
python scripts/train_model.py --model-type all
```

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests with coverage
pytest --cov=. --cov-report=html
```

## 🚀 Production Deployment

### Kubernetes
```bash
# Deploy to Kubernetes
./scripts/deploy.sh

# Check status
kubectl get pods -n patient-ml
```

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml patient-ml
```

## 🔍 Monitoring & Debugging

### Check Service Health
```bash
# All services
docker-compose ps

# Specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Database Access
```bash
# Connect to PostgreSQL
docker-compose exec db psql -U postgres -d patient_ml
```

### Redis Access
```bash
# Connect to Redis
docker-compose exec redis redis-cli
```

## 🎯 Key Features

### ML Predictions
- Risk scoring (0-1 probability)
- Risk buckets (Low/Medium/High)
- SHAP explanations
- Clinical recommendations

### Dashboard Features
- Patient cohort management
- Individual patient profiles
- Risk trend visualization
- Interactive explanations

### Security & Compliance
- JWT authentication
- Role-based access control
- Data encryption
- HIPAA audit logs
- GDPR compliance

### Monitoring
- System health metrics
- Model performance tracking
- Error rate monitoring
- Custom alerts

## 🆘 Troubleshooting

### Common Issues
1. **Port conflicts**: Change ports in docker-compose.yml
2. **Memory issues**: Increase Docker memory limit
3. **Permission errors**: Run with sudo or fix file permissions
4. **Database connection**: Check DATABASE_URL in .env

### Get Help
- Check logs: `docker-compose logs [service]`
- Restart services: `docker-compose restart`
- Clean rebuild: `docker-compose down && docker-compose up --build`

## 📚 Next Steps
1. Customize the ML models for your data
2. Integrate with your EHR system
3. Configure production secrets
4. Set up CI/CD pipeline
5. Scale for production load
