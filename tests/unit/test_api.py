import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from inference.main import app

client = TestClient(app)

class TestHealthEndpoint:
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

class TestAuthEndpoints:
    
    @patch('inference.security.SecurityManager.authenticate_user')
    def test_login_success(self, mock_auth):
        """Test successful login"""
        mock_auth.return_value = {
            "access_token": "test_token",
            "token_type": "bearer"
        }
        
        response = client.post("/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        
        assert response.status_code == 200
        assert "access_token" in response.json()
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = client.post("/auth/login", json={
            "username": "invalid",
            "password": "invalid"
        })
        
        assert response.status_code == 401

class TestPredictionEndpoints:
    
    @patch('inference.model_registry.ModelRegistry.predict')
    @patch('inference.security.SecurityManager.get_current_user')
    def test_single_prediction(self, mock_user, mock_predict):
        """Test single patient prediction"""
        mock_user.return_value = {"user_id": "test_user", "role": "clinician"}
        mock_predict.return_value = {
            "risk_score": 0.75,
            "risk_bucket": "High",
            "explanation": "Test explanation"
        }
        
        headers = {"Authorization": "Bearer test_token"}
        response = client.post("/predict/single", 
            json={"patient_id": "TEST_001"},
            headers=headers
        )
        
        assert response.status_code == 200
        assert "risk_score" in response.json()
    
    @patch('inference.model_registry.ModelRegistry.predict_batch')
    @patch('inference.security.SecurityManager.get_current_user')
    def test_batch_prediction(self, mock_user, mock_predict):
        """Test batch prediction"""
        mock_user.return_value = {"user_id": "test_user", "role": "clinician"}
        mock_predict.return_value = [
            {"patient_id": "TEST_001", "risk_score": 0.75},
            {"patient_id": "TEST_002", "risk_score": 0.25}
        ]
        
        headers = {"Authorization": "Bearer test_token"}
        response = client.post("/predict/batch",
            json={"patient_ids": ["TEST_001", "TEST_002"]},
            headers=headers
        )
        
        assert response.status_code == 200
        assert len(response.json()) == 2

class TestCohortEndpoints:
    
    @patch('inference.security.SecurityManager.get_current_user')
    def test_get_cohort(self, mock_user):
        """Test cohort retrieval"""
        mock_user.return_value = {"user_id": "test_user", "role": "clinician"}
        
        headers = {"Authorization": "Bearer test_token"}
        response = client.get("/cohort", headers=headers)
        
        # Should return 200 even with mock data
        assert response.status_code in [200, 404]  # 404 if no data found
