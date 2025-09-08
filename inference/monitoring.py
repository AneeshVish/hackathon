"""
Monitoring service for patient deterioration prediction API.
Tracks performance metrics, model drift, and system health.
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from collections import defaultdict, deque
import numpy as np
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetric:
    """Individual prediction metric"""
    timestamp: datetime
    patient_id: str
    risk_score: float
    response_time_ms: float
    user_id: str
    model_version: str


@dataclass
class SystemMetric:
    """System performance metric"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_requests: int
    error_rate: float


class MonitoringService:
    """Comprehensive monitoring for ML API"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.prediction_metrics = deque(maxlen=10000)
        self.system_metrics = deque(maxlen=1000)
        self.error_log = deque(maxlen=1000)
        
        # Real-time counters
        self.request_count = 0
        self.error_count = 0
        self.active_requests = 0
        
        # Performance tracking
        self.response_times = deque(maxlen=1000)
        self.risk_scores = deque(maxlen=1000)
        
        # Model drift detection
        self.feature_distributions = {}
        self.prediction_distributions = deque(maxlen=5000)
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        # Start background cleanup
        self._start_cleanup_thread()
    
    def log_prediction(self, patient_id: str, risk_score: float, user_id: str, response_time_ms: float = None):
        """Log a prediction for monitoring"""
        
        with self._lock:
            metric = PredictionMetric(
                timestamp=datetime.utcnow(),
                patient_id=patient_id,
                risk_score=risk_score,
                response_time_ms=response_time_ms or 0.0,
                user_id=user_id,
                model_version="1.0.0"  # Would get from model registry
            )
            
            self.prediction_metrics.append(metric)
            self.risk_scores.append(risk_score)
            
            if response_time_ms:
                self.response_times.append(response_time_ms)
            
            self.request_count += 1
    
    def log_error(self, error_type: str, error_message: str, patient_id: str = None, user_id: str = None):
        """Log an error for monitoring"""
        
        with self._lock:
            error_entry = {
                'timestamp': datetime.utcnow(),
                'error_type': error_type,
                'error_message': error_message,
                'patient_id': patient_id,
                'user_id': user_id
            }
            
            self.error_log.append(error_entry)
            self.error_count += 1
    
    def log_system_metrics(self, cpu_usage: float, memory_usage: float):
        """Log system performance metrics"""
        
        with self._lock:
            metric = SystemMetric(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                active_requests=self.active_requests,
                error_rate=self._calculate_error_rate()
            )
            
            self.system_metrics.append(metric)
    
    def increment_active_requests(self):
        """Increment active request counter"""
        with self._lock:
            self.active_requests += 1
    
    def decrement_active_requests(self):
        """Decrement active request counter"""
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics"""
        
        with self._lock:
            now = datetime.utcnow()
            
            # Calculate time-based metrics
            last_hour_predictions = [
                m for m in self.prediction_metrics 
                if (now - m.timestamp).total_seconds() < 3600
            ]
            
            last_24h_predictions = [
                m for m in self.prediction_metrics 
                if (now - m.timestamp).total_seconds() < 86400
            ]
            
            # Performance metrics
            performance_metrics = {
                'total_requests': self.request_count,
                'total_errors': self.error_count,
                'active_requests': self.active_requests,
                'requests_last_hour': len(last_hour_predictions),
                'requests_last_24h': len(last_24h_predictions),
                'error_rate': self._calculate_error_rate(),
                'avg_response_time_ms': np.mean(self.response_times) if self.response_times else 0,
                'p95_response_time_ms': np.percentile(self.response_times, 95) if self.response_times else 0,
                'p99_response_time_ms': np.percentile(self.response_times, 99) if self.response_times else 0
            }
            
            # Risk distribution metrics
            risk_distribution = self._calculate_risk_distribution()
            
            # Model drift metrics
            drift_metrics = self._calculate_drift_metrics()
            
            # System health
            system_health = self._calculate_system_health()
            
            return {
                'timestamp': now,
                'performance': performance_metrics,
                'risk_distribution': risk_distribution,
                'model_drift': drift_metrics,
                'system_health': system_health,
                'alerts': self._generate_alerts()
            }
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    def _calculate_risk_distribution(self) -> Dict[str, Any]:
        """Calculate risk score distribution"""
        
        if not self.risk_scores:
            return {'low': 0, 'medium': 0, 'high': 0}
        
        risk_array = np.array(self.risk_scores)
        
        return {
            'low': int(np.sum(risk_array < 0.1)),
            'medium': int(np.sum((risk_array >= 0.1) & (risk_array < 0.3))),
            'high': int(np.sum(risk_array >= 0.3)),
            'mean_risk': float(np.mean(risk_array)),
            'std_risk': float(np.std(risk_array)),
            'percentiles': {
                'p25': float(np.percentile(risk_array, 25)),
                'p50': float(np.percentile(risk_array, 50)),
                'p75': float(np.percentile(risk_array, 75)),
                'p95': float(np.percentile(risk_array, 95))
            }
        }
    
    def _calculate_drift_metrics(self) -> Dict[str, Any]:
        """Calculate model drift metrics"""
        
        # Simplified drift detection
        # In production, this would compare current vs training distributions
        
        if len(self.prediction_distributions) < 100:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'message': 'Insufficient data for drift detection'
            }
        
        # Compare recent vs historical predictions
        recent_predictions = list(self.prediction_distributions)[-100:]
        historical_predictions = list(self.prediction_distributions)[:-100] if len(self.prediction_distributions) > 200 else []
        
        if not historical_predictions:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'message': 'Insufficient historical data'
            }
        
        # Simple statistical test for drift
        recent_mean = np.mean(recent_predictions)
        historical_mean = np.mean(historical_predictions)
        
        drift_score = abs(recent_mean - historical_mean) / (np.std(historical_predictions) + 1e-8)
        drift_detected = drift_score > 2.0  # 2 standard deviations
        
        return {
            'drift_detected': drift_detected,
            'drift_score': float(drift_score),
            'recent_mean': float(recent_mean),
            'historical_mean': float(historical_mean),
            'threshold': 2.0
        }
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health"""
        
        if not self.system_metrics:
            return {'status': 'unknown', 'score': 0}
        
        # Get recent system metrics
        recent_metrics = [m for m in self.system_metrics if (datetime.utcnow() - m.timestamp).total_seconds() < 300]
        
        if not recent_metrics:
            return {'status': 'unknown', 'score': 0}
        
        # Calculate health score
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
        
        # Health scoring (0-100)
        cpu_score = max(0, 100 - avg_cpu)
        memory_score = max(0, 100 - avg_memory)
        error_score = max(0, 100 - avg_error_rate * 10)  # Scale error rate
        
        overall_score = (cpu_score + memory_score + error_score) / 3
        
        # Determine status
        if overall_score >= 80:
            status = 'healthy'
        elif overall_score >= 60:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'score': float(overall_score),
            'cpu_usage': float(avg_cpu),
            'memory_usage': float(avg_memory),
            'error_rate': float(avg_error_rate)
        }
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate system alerts based on metrics"""
        
        alerts = []
        
        # High error rate alert
        error_rate = self._calculate_error_rate()
        if error_rate > 5.0:  # 5% error rate threshold
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical' if error_rate > 10 else 'warning',
                'message': f'Error rate is {error_rate:.1f}%',
                'timestamp': datetime.utcnow()
            })
        
        # High response time alert
        if self.response_times:
            avg_response_time = np.mean(self.response_times)
            if avg_response_time > 1000:  # 1 second threshold
                alerts.append({
                    'type': 'high_response_time',
                    'severity': 'warning',
                    'message': f'Average response time is {avg_response_time:.0f}ms',
                    'timestamp': datetime.utcnow()
                })
        
        # Model drift alert
        drift_metrics = self._calculate_drift_metrics()
        if drift_metrics.get('drift_detected', False):
            alerts.append({
                'type': 'model_drift',
                'severity': 'warning',
                'message': f'Model drift detected (score: {drift_metrics["drift_score"]:.2f})',
                'timestamp': datetime.utcnow()
            })
        
        # Unusual risk distribution alert
        risk_dist = self._calculate_risk_distribution()
        if risk_dist.get('high', 0) > len(self.risk_scores) * 0.2:  # More than 20% high risk
            alerts.append({
                'type': 'high_risk_patients',
                'severity': 'info',
                'message': f'{risk_dist["high"]} patients flagged as high risk',
                'timestamp': datetime.utcnow()
            })
        
        return alerts
    
    def get_prediction_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get prediction history for specified time period"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            recent_predictions = [
                asdict(m) for m in self.prediction_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        return recent_predictions
    
    def get_error_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get error history for specified time period"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            recent_errors = [
                error for error in self.error_log 
                if error['timestamp'] >= cutoff_time
            ]
        
        return recent_errors
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        
        metrics = self.get_metrics()
        
        if format.lower() == 'json':
            return json.dumps(metrics, default=str, indent=2)
        elif format.lower() == 'prometheus':
            return self._format_prometheus_metrics(metrics)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_prometheus_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for Prometheus"""
        
        prometheus_metrics = []
        
        # Performance metrics
        perf = metrics['performance']
        prometheus_metrics.extend([
            f"api_requests_total {perf['total_requests']}",
            f"api_errors_total {perf['total_errors']}",
            f"api_active_requests {perf['active_requests']}",
            f"api_error_rate {perf['error_rate']}",
            f"api_response_time_avg_ms {perf['avg_response_time_ms']}",
            f"api_response_time_p95_ms {perf['p95_response_time_ms']}"
        ])
        
        # Risk distribution
        risk_dist = metrics['risk_distribution']
        prometheus_metrics.extend([
            f"predictions_low_risk {risk_dist['low']}",
            f"predictions_medium_risk {risk_dist['medium']}",
            f"predictions_high_risk {risk_dist['high']}",
            f"predictions_mean_risk {risk_dist['mean_risk']}"
        ])
        
        # System health
        health = metrics['system_health']
        prometheus_metrics.extend([
            f"system_health_score {health['score']}",
            f"system_cpu_usage {health.get('cpu_usage', 0)}",
            f"system_memory_usage {health.get('memory_usage', 0)}"
        ])
        
        return '\n'.join(prometheus_metrics)
    
    def _start_cleanup_thread(self):
        """Start background thread for data cleanup"""
        
        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._cleanup_old_data()
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_old_data(self):
        """Remove old data beyond retention period"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            # Clean prediction metrics
            self.prediction_metrics = deque([
                m for m in self.prediction_metrics if m.timestamp >= cutoff_time
            ], maxlen=self.prediction_metrics.maxlen)
            
            # Clean system metrics
            self.system_metrics = deque([
                m for m in self.system_metrics if m.timestamp >= cutoff_time
            ], maxlen=self.system_metrics.maxlen)
            
            # Clean error log
            self.error_log = deque([
                e for e in self.error_log if e['timestamp'] >= cutoff_time
            ], maxlen=self.error_log.maxlen)
        
        logger.info(f"Cleaned up monitoring data older than {self.retention_hours} hours")
