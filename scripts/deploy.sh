#!/bin/bash

# Patient ML Deployment Script
set -e

echo "🚀 Deploying Patient Deterioration ML System..."

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if we're connected to a cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Not connected to a Kubernetes cluster. Please configure kubectl."
    exit 1
fi

# Create namespace
echo "📁 Creating namespace..."
kubectl apply -f k8s/namespace.yaml

# Apply configurations
echo "⚙️  Applying configurations..."
kubectl apply -f k8s/configmap.yaml

# Deploy database
echo "🗄️  Deploying PostgreSQL..."
kubectl apply -f k8s/postgres.yaml

# Deploy Redis
echo "🔄 Deploying Redis..."
kubectl apply -f k8s/redis.yaml

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n patient-ml --timeout=300s

# Deploy backend
echo "🔧 Deploying backend..."
kubectl apply -f k8s/backend.yaml

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
kubectl wait --for=condition=ready pod -l app=backend -n patient-ml --timeout=300s

# Deploy frontend
echo "🌐 Deploying frontend..."
kubectl apply -f k8s/frontend.yaml

# Deploy ingress
echo "🌍 Deploying ingress..."
kubectl apply -f k8s/ingress.yaml

# Check deployment status
echo "🔍 Checking deployment status..."
kubectl get pods -n patient-ml
kubectl get services -n patient-ml

echo "✅ Deployment complete!"
echo "🌐 Access the application at: https://patient-ml.example.com"
echo "📊 Monitor with: kubectl get pods -n patient-ml"
