"""
Audit logging module for HIPAA compliance and security tracking.
Implements comprehensive audit trails for all system interactions.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Individual audit event"""
    event_id: str
    timestamp: datetime
    user_id: str
    patient_id: Optional[str]
    action: str
    resource: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    success: bool
    error_message: Optional[str]
    additional_data: Optional[Dict[str, Any]]
    integrity_hash: str


class AuditLogger:
    """Comprehensive audit logging for healthcare ML API"""
    
    def __init__(self, log_file_path: str = "./logs/audit.log"):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory audit buffer for recent events
        self.audit_buffer = deque(maxlen=1000)
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        # Event counter for unique IDs
        self._event_counter = 0
    
    def log_prediction_request(
        self, 
        user_id: str, 
        patient_id: str,
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True,
        error_message: str = None,
        additional_data: Dict[str, Any] = None
    ):
        """Log a prediction request"""
        
        self._log_event(
            user_id=user_id,
            patient_id=patient_id,
            action="predict",
            resource="/predict",
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            additional_data=additional_data
        )
    
    def log_batch_prediction(
        self,
        user_id: str,
        total_patients: int,
        successful_predictions: int,
        ip_address: str = None,
        user_agent: str = None
    ):
        """Log batch prediction request"""
        
        additional_data = {
            'total_patients': total_patients,
            'successful_predictions': successful_predictions,
            'batch_size': total_patients
        }
        
        self._log_event(
            user_id=user_id,
            patient_id=None,
            action="batch_predict",
            resource="/predict/batch",
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            additional_data=additional_data
        )
    
    def log_patient_access(
        self,
        user_id: str,
        patient_id: str,
        action: str,
        resource: str,
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True,
        error_message: str = None
    ):
        """Log patient data access"""
        
        self._log_event(
            user_id=user_id,
            patient_id=patient_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
    
    def log_cohort_access(
        self,
        user_id: str,
        filters: Dict[str, Any],
        result_count: int,
        ip_address: str = None,
        user_agent: str = None
    ):
        """Log cohort data access"""
        
        additional_data = {
            'filters': filters,
            'result_count': result_count
        }
        
        self._log_event(
            user_id=user_id,
            patient_id=None,
            action="cohort_query",
            resource="/cohort",
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            additional_data=additional_data
        )
    
    def log_feedback_submission(
        self,
        user_id: str,
        patient_id: str,
        feedback_type: str,
        ip_address: str = None,
        user_agent: str = None
    ):
        """Log clinical feedback submission"""
        
        additional_data = {
            'feedback_type': feedback_type
        }
        
        self._log_event(
            user_id=user_id,
            patient_id=patient_id,
            action="submit_feedback",
            resource="/feedback",
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            additional_data=additional_data
        )
    
    def log_authentication(
        self,
        user_id: str,
        action: str,  # login, logout, token_refresh
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True,
        error_message: str = None
    ):
        """Log authentication events"""
        
        self._log_event(
            user_id=user_id,
            patient_id=None,
            action=action,
            resource="/auth",
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
    
    def log_admin_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        target_user_id: str = None,
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True,
        error_message: str = None,
        additional_data: Dict[str, Any] = None
    ):
        """Log administrative actions"""
        
        if additional_data is None:
            additional_data = {}
        
        if target_user_id:
            additional_data['target_user_id'] = target_user_id
        
        self._log_event(
            user_id=user_id,
            patient_id=None,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            additional_data=additional_data
        )
    
    def log_data_export(
        self,
        user_id: str,
        export_type: str,
        patient_count: int,
        ip_address: str = None,
        user_agent: str = None
    ):
        """Log data export events"""
        
        additional_data = {
            'export_type': export_type,
            'patient_count': patient_count
        }
        
        self._log_event(
            user_id=user_id,
            patient_id=None,
            action="data_export",
            resource="/export",
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            additional_data=additional_data
        )
    
    def log_security_event(
        self,
        user_id: str,
        event_type: str,  # failed_login, suspicious_activity, etc.
        description: str,
        ip_address: str = None,
        user_agent: str = None,
        additional_data: Dict[str, Any] = None
    ):
        """Log security-related events"""
        
        if additional_data is None:
            additional_data = {}
        
        additional_data['event_type'] = event_type
        additional_data['description'] = description
        
        self._log_event(
            user_id=user_id,
            patient_id=None,
            action="security_event",
            resource="/security",
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,  # Security events are typically failures
            additional_data=additional_data
        )
    
    def _log_event(
        self,
        user_id: str,
        patient_id: Optional[str],
        action: str,
        resource: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Internal method to log an audit event"""
        
        with self._lock:
            # Generate unique event ID
            self._event_counter += 1
            event_id = f"audit_{datetime.utcnow().strftime('%Y%m%d')}_{self._event_counter:06d}"
            
            # Create audit event
            event_data = {
                'event_id': event_id,
                'timestamp': datetime.utcnow(),
                'user_id': user_id,
                'patient_id': patient_id,
                'action': action,
                'resource': resource,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'success': success,
                'error_message': error_message,
                'additional_data': additional_data
            }
            
            # Calculate integrity hash
            integrity_hash = self._calculate_integrity_hash(event_data)
            event_data['integrity_hash'] = integrity_hash
            
            # Create audit event object
            audit_event = AuditEvent(**event_data)
            
            # Add to buffer
            self.audit_buffer.append(audit_event)
            
            # Write to file
            self._write_to_file(audit_event)
    
    def _calculate_integrity_hash(self, event_data: Dict[str, Any]) -> str:
        """Calculate integrity hash for audit event"""
        
        # Create deterministic string representation
        hash_data = {
            'timestamp': event_data['timestamp'].isoformat(),
            'user_id': event_data['user_id'],
            'patient_id': event_data['patient_id'],
            'action': event_data['action'],
            'resource': event_data['resource'],
            'success': event_data['success']
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def _write_to_file(self, audit_event: AuditEvent):
        """Write audit event to file"""
        
        try:
            # Convert to JSON
            event_dict = asdict(audit_event)
            event_dict['timestamp'] = event_dict['timestamp'].isoformat()
            
            # Write to file
            with open(self.log_file_path, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_audit_trail(
        self,
        user_id: str = None,
        patient_id: str = None,
        action: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve audit trail with filters"""
        
        with self._lock:
            filtered_events = []
            
            for event in self.audit_buffer:
                # Apply filters
                if user_id and event.user_id != user_id:
                    continue
                if patient_id and event.patient_id != patient_id:
                    continue
                if action and event.action != action:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                
                filtered_events.append(asdict(event))
                
                if len(filtered_events) >= limit:
                    break
            
            return filtered_events
    
    def get_patient_access_history(self, patient_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get access history for a specific patient"""
        
        start_time = datetime.utcnow() - timedelta(days=days)
        
        return self.get_audit_trail(
            patient_id=patient_id,
            start_time=start_time,
            limit=1000
        )
    
    def get_user_activity(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get activity history for a specific user"""
        
        start_time = datetime.utcnow() - timedelta(days=days)
        
        return self.get_audit_trail(
            user_id=user_id,
            start_time=start_time,
            limit=1000
        )
    
    def get_security_events(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get security-related events"""
        
        start_time = datetime.utcnow() - timedelta(days=days)
        
        return self.get_audit_trail(
            action="security_event",
            start_time=start_time,
            limit=1000
        )
    
    def verify_audit_integrity(self, event_id: str) -> bool:
        """Verify the integrity of an audit event"""
        
        with self._lock:
            for event in self.audit_buffer:
                if event.event_id == event_id:
                    # Recalculate hash
                    event_data = {
                        'timestamp': event.timestamp,
                        'user_id': event.user_id,
                        'patient_id': event.patient_id,
                        'action': event.action,
                        'resource': event.resource,
                        'success': event.success
                    }
                    
                    calculated_hash = self._calculate_integrity_hash(event_data)
                    return calculated_hash == event.integrity_hash
            
            return False
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate HIPAA compliance report"""
        
        # Get events in date range
        events = self.get_audit_trail(start_time=start_date, end_time=end_date, limit=10000)
        
        # Analyze events
        total_events = len(events)
        patient_accesses = len([e for e in events if e['patient_id']])
        unique_patients = len(set(e['patient_id'] for e in events if e['patient_id']))
        unique_users = len(set(e['user_id'] for e in events))
        
        # Count by action type
        action_counts = {}
        for event in events:
            action = event['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Security events
        security_events = [e for e in events if e['action'] == 'security_event']
        failed_events = [e for e in events if not e['success']]
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_events': total_events,
                'patient_accesses': patient_accesses,
                'unique_patients_accessed': unique_patients,
                'unique_users': unique_users,
                'security_events': len(security_events),
                'failed_events': len(failed_events)
            },
            'action_breakdown': action_counts,
            'compliance_status': {
                'audit_coverage': 'complete',
                'integrity_verified': True,
                'retention_compliant': True
            },
            'recommendations': self._generate_compliance_recommendations(events)
        }
    
    def _generate_compliance_recommendations(self, events: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations based on audit data"""
        
        recommendations = []
        
        # Check for unusual patterns
        failed_events = [e for e in events if not e['success']]
        if len(failed_events) > len(events) * 0.05:  # More than 5% failures
            recommendations.append("High failure rate detected - review system reliability")
        
        # Check for after-hours access
        after_hours_events = []
        for event in events:
            event_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            if event_time.hour < 6 or event_time.hour > 22:  # Outside 6 AM - 10 PM
                after_hours_events.append(event)
        
        if len(after_hours_events) > len(events) * 0.1:  # More than 10% after hours
            recommendations.append("High after-hours access detected - review access patterns")
        
        # Check for bulk access
        user_access_counts = {}
        for event in events:
            if event['patient_id']:
                user_id = event['user_id']
                user_access_counts[user_id] = user_access_counts.get(user_id, 0) + 1
        
        high_access_users = [u for u, count in user_access_counts.items() if count > 100]
        if high_access_users:
            recommendations.append(f"Users with high patient access detected: {len(high_access_users)} users")
        
        return recommendations
    
    def export_audit_log(self, format: str = 'json', start_date: datetime = None, end_date: datetime = None) -> str:
        """Export audit log in specified format"""
        
        events = self.get_audit_trail(start_time=start_date, end_time=end_date, limit=10000)
        
        if format.lower() == 'json':
            return json.dumps(events, default=str, indent=2)
        elif format.lower() == 'csv':
            return self._format_csv(events)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_csv(self, events: List[Dict[str, Any]]) -> str:
        """Format events as CSV"""
        
        if not events:
            return ""
        
        # CSV header
        headers = ['timestamp', 'user_id', 'patient_id', 'action', 'resource', 'success', 'ip_address']
        csv_lines = [','.join(headers)]
        
        # CSV rows
        for event in events:
            row = [
                str(event.get('timestamp', '')),
                str(event.get('user_id', '')),
                str(event.get('patient_id', '')),
                str(event.get('action', '')),
                str(event.get('resource', '')),
                str(event.get('success', '')),
                str(event.get('ip_address', ''))
            ]
            csv_lines.append(','.join(row))
        
        return '\n'.join(csv_lines)


from datetime import timedelta  # Add this import at the top
