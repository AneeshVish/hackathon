import pytest
import requests
import time
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class TestEndToEndWorkflow:
    """End-to-end integration tests"""
    
    BASE_URL = "http://localhost:8000"
    
    @pytest.fixture(scope="class")
    def auth_token(self):
        """Get authentication token for testing"""
        response = requests.post(f"{self.BASE_URL}/auth/demo-token", json={
            "user_id": "test_clinician",
            "role": "clinician"
        })
        
        if response.status_code == 200:
            return response.json()["access_token"]
        return None
    
    def test_health_check(self):
        """Test system health"""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_model_info(self, auth_token):
        """Test model information endpoint"""
        if not auth_token:
            pytest.skip("No auth token available")
            
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = requests.get(f"{self.BASE_URL}/model/info", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "features" in data
    
    def test_prediction_workflow(self, auth_token):
        """Test complete prediction workflow"""
        if not auth_token:
            pytest.skip("No auth token available")
            
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Test single prediction
        prediction_data = {
            "patient_id": "TEST_PATIENT_001",
            "features": {
                "age": 65,
                "gender": "M",
                "systolic_bp": 140,
                "diastolic_bp": 90,
                "heart_rate": 80,
                "weight": 180,
                "height": 70
            }
        }
        
        response = requests.post(
            f"{self.BASE_URL}/predict/single",
            json=prediction_data,
            headers=headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "risk_score" in result
        assert "risk_bucket" in result
        assert "explanation" in result
        assert 0 <= result["risk_score"] <= 1
    
    def test_cohort_workflow(self, auth_token):
        """Test cohort management workflow"""
        if not auth_token:
            pytest.skip("No auth token available")
            
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Get cohort data
        response = requests.get(f"{self.BASE_URL}/cohort", headers=headers)
        
        # Should return 200 or 404 (if no data)
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "patients" in data
            assert "total_count" in data
    
    def test_feedback_submission(self, auth_token):
        """Test feedback submission"""
        if not auth_token:
            pytest.skip("No auth token available")
            
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        feedback_data = {
            "patient_id": "TEST_PATIENT_001",
            "prediction_id": "test_prediction_123",
            "feedback_type": "outcome",
            "outcome": "no_deterioration",
            "comments": "Patient remained stable"
        }
        
        response = requests.post(
            f"{self.BASE_URL}/feedback",
            json=feedback_data,
            headers=headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
