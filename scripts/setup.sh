#!/bin/bash

# Patient ML Setup Script
set -e

echo "🏥 Setting up Patient Deterioration ML System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models/saved feature_store data/raw data/processed

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please update .env file with your configuration"
fi

# Build and start services
echo "🐳 Building and starting Docker services..."
docker-compose build
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Run database migrations (if needed)
echo "🗄️  Setting up database..."
docker-compose exec backend python -c "
import asyncio
from data.schemas import *
print('Database schemas loaded successfully')
"

echo "✅ Setup complete!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📊 MLflow: http://localhost:5000"
echo "📈 Prometheus: http://localhost:9090"
echo "📊 Grafana: http://localhost:3001 (admin/admin)"

echo ""
echo "🚀 To get started:"
echo "1. Update .env file with your configuration"
echo "2. Visit http://localhost:3000 to access the dashboard"
echo "3. Use demo credentials: demo_clinician / demo123"
