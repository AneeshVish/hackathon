# Patient Deterioration ML API Documentation

## Overview
This API provides endpoints for predicting patient deterioration risk using machine learning models.

## Authentication
All endpoints require JWT authentication via Bearer token.

```bash
Authorization: Bearer <your_jwt_token>
```

## Endpoints

### Health Check
```
GET /health
```
Returns system health status.

### Authentication
```
POST /auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

### Single Patient Prediction
```
POST /predict/single
Authorization: Bearer <token>
Content-Type: application/json

{
  "patient_id": "string",
  "features": {
    "age": 65,
    "gender": "M",
    "systolic_bp": 140,
    "diastolic_bp": 90,
    "heart_rate": 80
  }
}
```

Response:
```json
{
  "patient_id": "PATIENT_001",
  "risk_score": 0.72,
  "risk_bucket": "High",
  "prediction_timestamp": "2025-01-08T10:30:00Z",
  "explanation": {
    "summary": "High risk due to elevated biomarkers",
    "top_factors": [
      {
        "factor": "Elevated BNP",
        "importance": 0.25,
        "value": "850 pg/mL"
      }
    ]
  },
  "recommendations": [
    {
      "category": "Immediate",
      "action": "Contact patient within 24 hours",
      "priority": "high"
    }
  ]
}
```

### Batch Prediction
```
POST /predict/batch
Authorization: Bearer <token>
Content-Type: application/json

{
  "patient_ids": ["PATIENT_001", "PATIENT_002"]
}
```

### Cohort Query
```
GET /cohort?risk_bucket=High&limit=50
Authorization: Bearer <token>
```

### Patient Features
```
GET /patient/{patient_id}/features
Authorization: Bearer <token>
```

### Submit Feedback
```
POST /feedback
Authorization: Bearer <token>
Content-Type: application/json

{
  "patient_id": "PATIENT_001",
  "prediction_id": "pred_123",
  "feedback_type": "outcome",
  "outcome": "no_deterioration",
  "comments": "Patient remained stable"
}
```

### Model Information
```
GET /model/info
Authorization: Bearer <token>
```

### System Metrics
```
GET /metrics
Authorization: Bearer <token>
```

## Error Responses

All endpoints return standard HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `500` - Internal Server Error

Error response format:
```json
{
  "error": "error_code",
  "message": "Human readable error message",
  "details": {}
}
```

## Rate Limits
- 1000 requests per hour per user
- 100 batch predictions per hour per user

## Data Privacy
All patient data is encrypted and HIPAA compliant. Audit logs are maintained for all API access.
