"""
Security module for patient deterioration prediction API.
Implements authentication, authorization, and HIPAA compliance features.
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from cryptography.fernet import Fernet
import os
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)


class SecurityManager:
    """Comprehensive security management for healthcare ML API"""
    
    def __init__(self):
        self.secret_key = os.getenv('API_SECRET_KEY', 'your-secret-key-change-in-production')
        self.algorithm = os.getenv('API_ALGORITHM', 'HS256')
        self.token_expire_minutes = int(os.getenv('API_ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
        
        # Initialize encryption
        encryption_key = os.getenv('ENCRYPTION_KEY')
        if encryption_key and len(encryption_key) == 32:
            self.cipher_suite = Fernet(Fernet.generate_key())
        else:
            logger.warning("No valid encryption key provided")
            self.cipher_suite = None
        
        # User permissions cache
        self._permissions_cache = {}
        
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token for authenticated user"""
        
        expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
        
        token_data = {
            'user_id': user_data['user_id'],
            'username': user_data['username'],
            'role': user_data['role'],
            'permissions': user_data.get('permissions', []),
            'clinic_access': user_data.get('clinic_access', []),
            'care_team_access': user_data.get('care_team_access', []),
            'exp': expire,
            'iat': datetime.utcnow(),
            'iss': 'patient-deterioration-api'
        }
        
        token = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
        return token
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return user information"""
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                raise jwt.ExpiredSignatureError("Token has expired")
            
            # Return user info
            return {
                'user_id': payload['user_id'],
                'username': payload['username'],
                'role': payload['role'],
                'permissions': payload.get('permissions', []),
                'clinic_access': payload.get('clinic_access', []),
                'care_team_access': payload.get('care_team_access', [])
            }
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise ValueError("Token validation failed")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def check_patient_access(self, user_info: Dict[str, Any], patient_id: str) -> bool:
        """
        Check if user has access to specific patient data
        
        Args:
            user_info: User information from token
            patient_id: Patient identifier
            
        Returns:
            True if user has access
        """
        
        # Admin users have access to all patients
        if user_info.get('role') == 'admin':
            return True
        
        # Check if user has explicit patient access permission
        if 'patient_access' in user_info.get('permissions', []):
            # In production, this would check against a database
            # For now, implement basic clinic/care team based access
            
            # Get patient's clinic/care team (mock implementation)
            patient_clinic = self._get_patient_clinic(patient_id)
            patient_care_team = self._get_patient_care_team(patient_id)
            
            # Check clinic access
            user_clinics = user_info.get('clinic_access', [])
            if patient_clinic and patient_clinic in user_clinics:
                return True
            
            # Check care team access
            user_care_teams = user_info.get('care_team_access', [])
            if patient_care_team and patient_care_team in user_care_teams:
                return True
        
        return False
    
    def build_cohort_filters(self, user_info: Dict[str, Any], requested_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build cohort filters based on user permissions
        
        Args:
            user_info: User information
            requested_filters: Filters requested by user
            
        Returns:
            Filtered query parameters based on user access
        """
        
        filters = requested_filters.copy()
        
        # Admin users can see all data
        if user_info.get('role') == 'admin':
            return filters
        
        # Restrict to user's accessible clinics/care teams
        user_clinics = user_info.get('clinic_access', [])
        user_care_teams = user_info.get('care_team_access', [])
        
        # If user has limited clinic access, restrict filters
        if user_clinics:
            if 'clinic_id' not in filters or filters['clinic_id'] not in user_clinics:
                filters['clinic_id'] = user_clinics[0]  # Default to first accessible clinic
        
        if user_care_teams:
            if 'care_team_id' not in filters or filters['care_team_id'] not in user_care_teams:
                filters['care_team_id'] = user_care_teams[0]  # Default to first accessible team
        
        return filters
    
    def is_admin(self, user_info: Dict[str, Any]) -> bool:
        """Check if user has admin privileges"""
        return user_info.get('role') == 'admin'
    
    def has_permission(self, user_info: Dict[str, Any], permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in user_info.get('permissions', [])
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for storage"""
        if self.cipher_suite:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return encrypted_data.decode()
        else:
            logger.warning("No encryption available - storing data in plain text")
            return data
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if self.cipher_suite:
            try:
                decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
                return decrypted_data.decode()
            except Exception as e:
                logger.error(f"Decryption failed: {e}")
                return ""
        else:
            return encrypted_data
    
    def generate_audit_hash(self, data: Dict[str, Any]) -> str:
        """Generate hash for audit trail integrity"""
        # Create deterministic hash of audit data
        audit_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(audit_string.encode()).hexdigest()
    
    def validate_hipaa_compliance(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate request for HIPAA compliance
        
        Args:
            request_data: Request data to validate
            
        Returns:
            Validation results with compliance status
        """
        
        compliance_issues = []
        
        # Check for PHI in request
        phi_fields = ['ssn', 'phone', 'email', 'address', 'full_name']
        for field in phi_fields:
            if field in request_data:
                compliance_issues.append(f"PHI field '{field}' detected in request")
        
        # Check for minimum necessary principle
        if len(request_data.get('patient_ids', [])) > 100:
            compliance_issues.append("Large batch request may violate minimum necessary principle")
        
        return {
            'compliant': len(compliance_issues) == 0,
            'issues': compliance_issues,
            'risk_level': 'high' if compliance_issues else 'low'
        }
    
    def _get_patient_clinic(self, patient_id: str) -> Optional[str]:
        """Get patient's clinic (mock implementation)"""
        # In production, this would query the database
        # For demo purposes, derive from patient ID
        if patient_id.startswith("DEMO"):
            return "DEMO_CLINIC_001"
        return None
    
    def _get_patient_care_team(self, patient_id: str) -> Optional[str]:
        """Get patient's care team (mock implementation)"""
        # In production, this would query the database
        if patient_id.startswith("DEMO"):
            return "DEMO_TEAM_001"
        return None
    
    def create_demo_user_token(self, role: str = "clinician") -> str:
        """Create demo user token for testing"""
        
        demo_user_data = {
            'user_id': f'demo_{role}_001',
            'username': f'demo_{role}',
            'role': role,
            'permissions': self._get_role_permissions(role),
            'clinic_access': ['DEMO_CLINIC_001', 'DEMO_CLINIC_002'],
            'care_team_access': ['DEMO_TEAM_001', 'DEMO_TEAM_002']
        }
        
        return self.create_access_token(demo_user_data)
    
    def _get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a role"""
        
        role_permissions = {
            'admin': [
                'patient_access', 'cohort_access', 'model_info', 
                'metrics_access', 'user_management', 'system_admin'
            ],
            'clinician': [
                'patient_access', 'cohort_access', 'submit_feedback'
            ],
            'nurse': [
                'patient_access', 'submit_feedback'
            ],
            'researcher': [
                'cohort_access', 'model_info'
            ]
        }
        
        return role_permissions.get(role, [])


# Decorator for role-based access control
def require_role(required_role: str):
    """Decorator to require specific role for endpoint access"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user_info from kwargs (injected by dependency)
            user_info = kwargs.get('user_info')
            
            if not user_info:
                raise ValueError("User information not available")
            
            if user_info.get('role') != required_role and user_info.get('role') != 'admin':
                raise ValueError(f"Role '{required_role}' required")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Decorator for permission-based access control
def require_permission(required_permission: str):
    """Decorator to require specific permission for endpoint access"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user_info = kwargs.get('user_info')
            
            if not user_info:
                raise ValueError("User information not available")
            
            permissions = user_info.get('permissions', [])
            if required_permission not in permissions and user_info.get('role') != 'admin':
                raise ValueError(f"Permission '{required_permission}' required")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


import json  # Add this import at the top
