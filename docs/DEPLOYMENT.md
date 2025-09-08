# Deployment Guide

## Quick Start with Docker Compose

1. **Clone and Setup**
```bash
git clone <repository>
cd hackathon
cp .env.example .env
# Edit .env with your configuration
```

2. **Run Setup Script**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

3. **Access Services**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- MLflow: http://localhost:5000
- Grafana: http://localhost:3001 (admin/admin)

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Ingress controller (nginx)
- Cert-manager (for TLS)

### Deploy to Kubernetes
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

### Manual Deployment
```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply configurations
kubectl apply -f k8s/configmap.yaml

# Deploy services
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/frontend.yaml
kubectl apply -f k8s/ingress.yaml
```

## Production Considerations

### Security
- Change default passwords in production
- Use proper TLS certificates
- Configure firewall rules
- Enable audit logging
- Use secrets management (Vault, etc.)

### Scaling
- Backend: Scale replicas based on load
- Database: Use managed PostgreSQL service
- Redis: Use Redis Cluster for high availability
- Storage: Use persistent volumes

### Monitoring
- Prometheus for metrics collection
- Grafana for visualization
- ELK stack for log aggregation
- Health checks and alerting

### Backup
- Database backups (automated)
- Model artifacts backup
- Configuration backup
- Disaster recovery plan

## Environment Variables

### Required
```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379
API_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key
```

### Optional
```bash
MLFLOW_TRACKING_URI=http://mlflow:5000
DEVELOPMENT_MODE=false
LOG_LEVEL=INFO
PROMETHEUS_PORT=8080
```

## Troubleshooting

### Common Issues
1. **Database Connection Failed**
   - Check DATABASE_URL
   - Verify PostgreSQL is running
   - Check network connectivity

2. **Model Loading Error**
   - Ensure model files exist
   - Check MLflow connection
   - Verify model registry

3. **Authentication Issues**
   - Check JWT secret key
   - Verify token expiration
   - Check user permissions

### Logs
```bash
# Docker Compose
docker-compose logs backend
docker-compose logs frontend

# Kubernetes
kubectl logs -f deployment/backend -n patient-ml
kubectl logs -f deployment/frontend -n patient-ml
```
