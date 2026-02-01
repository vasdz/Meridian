"""ML Monitoring & Observability Module - Enterprise-grade monitoring.

This module provides production-ready monitoring capabilities:
- Model performance tracking
- Data drift detection
- Feature drift detection
- Prediction monitoring
- Alerting system
- SLA tracking

Designed for hyperscale ML operations.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import hashlib

from meridian.core.logging import get_logger


logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftType(Enum):
    """Type of drift detection."""
    DATA = "data"
    FEATURE = "feature"
    PREDICTION = "prediction"
    CONCEPT = "concept"


class MetricType(Enum):
    """Type of monitored metric."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    DRIFT_SCORE = "drift_score"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Monitoring alert."""

    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_name: str
    metric_value: float
    threshold: float
    message: str

    model_id: Optional[str] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "message": self.message,
            "model_id": self.model_id,
            "acknowledged": self.acknowledged,
        }


@dataclass
class DriftResult:
    """Result of drift detection."""

    drift_type: DriftType
    feature_name: Optional[str]
    drift_score: float
    is_drifted: bool
    threshold: float

    reference_stats: dict[str, float]
    current_stats: dict[str, float]

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "drift_type": self.drift_type.value,
            "feature_name": self.feature_name,
            "drift_score": self.drift_score,
            "is_drifted": self.is_drifted,
            "threshold": self.threshold,
            "reference_stats": self.reference_stats,
            "current_stats": self.current_stats,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""

    model_id: str
    timestamp: datetime

    # Prediction metrics
    prediction_count: int = 0
    mean_prediction: float = 0.0
    std_prediction: float = 0.0

    # Latency metrics
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Accuracy metrics (if ground truth available)
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None

    # Error metrics
    error_count: int = 0
    error_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "prediction_count": self.prediction_count,
            "mean_prediction": self.mean_prediction,
            "std_prediction": self.std_prediction,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "accuracy": self.accuracy,
            "rmse": self.rmse,
            "error_rate": self.error_rate,
        }


class DriftDetector:
    """
    Statistical drift detection.

    Implements multiple drift detection methods:
    - Kolmogorov-Smirnov test
    - Population Stability Index (PSI)
    - Wasserstein distance
    - Chi-squared test (categorical)
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.1,
        method: str = "psi",
    ):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.method = method

        self._reference_stats: dict[str, dict] = {}

    def set_reference(self, data: pd.DataFrame) -> None:
        """Set reference data for drift detection."""
        self.reference_data = data
        self._compute_reference_stats()
        logger.info("Reference data set", n_samples=len(data), n_features=len(data.columns))

    def _compute_reference_stats(self) -> None:
        """Compute statistics for reference data."""
        if self.reference_data is None:
            return

        for col in self.reference_data.columns:
            if self.reference_data[col].dtype in ['int64', 'float64']:
                self._reference_stats[col] = {
                    "mean": self.reference_data[col].mean(),
                    "std": self.reference_data[col].std(),
                    "min": self.reference_data[col].min(),
                    "max": self.reference_data[col].max(),
                    "median": self.reference_data[col].median(),
                    "q25": self.reference_data[col].quantile(0.25),
                    "q75": self.reference_data[col].quantile(0.75),
                }
            else:
                value_counts = self.reference_data[col].value_counts(normalize=True)
                self._reference_stats[col] = {
                    "distribution": value_counts.to_dict(),
                    "n_unique": self.reference_data[col].nunique(),
                }

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        features: Optional[list[str]] = None,
    ) -> list[DriftResult]:
        """
        Detect drift in current data compared to reference.

        Args:
            current_data: Current data to check
            features: Specific features to check (default: all)

        Returns:
            List of DriftResult for each feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        features = features or list(current_data.columns)
        results = []

        for feature in features:
            if feature not in self.reference_data.columns:
                continue

            if self.reference_data[feature].dtype in ['int64', 'float64']:
                result = self._detect_numeric_drift(feature, current_data[feature])
            else:
                result = self._detect_categorical_drift(feature, current_data[feature])

            results.append(result)

        n_drifted = sum(1 for r in results if r.is_drifted)
        logger.info(
            "Drift detection complete",
            n_features=len(features),
            n_drifted=n_drifted,
        )

        return results

    def _detect_numeric_drift(
        self,
        feature: str,
        current_values: pd.Series,
    ) -> DriftResult:
        """Detect drift for numeric feature."""
        reference_values = self.reference_data[feature].dropna()
        current_values = current_values.dropna()

        if self.method == "psi":
            drift_score = self._calculate_psi(reference_values, current_values)
        elif self.method == "ks":
            drift_score = self._calculate_ks(reference_values, current_values)
        else:
            drift_score = self._calculate_wasserstein(reference_values, current_values)

        current_stats = {
            "mean": current_values.mean(),
            "std": current_values.std(),
            "min": current_values.min(),
            "max": current_values.max(),
        }

        return DriftResult(
            drift_type=DriftType.FEATURE,
            feature_name=feature,
            drift_score=drift_score,
            is_drifted=drift_score > self.drift_threshold,
            threshold=self.drift_threshold,
            reference_stats=self._reference_stats.get(feature, {}),
            current_stats=current_stats,
        )

    def _detect_categorical_drift(
        self,
        feature: str,
        current_values: pd.Series,
    ) -> DriftResult:
        """Detect drift for categorical feature."""
        reference_dist = self._reference_stats.get(feature, {}).get("distribution", {})
        current_dist = current_values.value_counts(normalize=True).to_dict()

        # Chi-squared-like metric
        all_categories = set(reference_dist.keys()) | set(current_dist.keys())

        drift_score = 0.0
        for cat in all_categories:
            ref_prob = reference_dist.get(cat, 0.001)
            cur_prob = current_dist.get(cat, 0.001)
            drift_score += (cur_prob - ref_prob) ** 2 / ref_prob

        current_stats = {
            "distribution": current_dist,
            "n_unique": current_values.nunique(),
        }

        return DriftResult(
            drift_type=DriftType.FEATURE,
            feature_name=feature,
            drift_score=drift_score,
            is_drifted=drift_score > self.drift_threshold,
            threshold=self.drift_threshold,
            reference_stats=self._reference_stats.get(feature, {}),
            current_stats=current_stats,
        )

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index."""
        # Create bins from reference
        _, bins = pd.cut(reference, bins=n_bins, retbins=True)

        # Calculate proportions
        ref_counts = pd.cut(reference, bins=bins).value_counts(normalize=True, sort=False)
        cur_counts = pd.cut(current, bins=bins).value_counts(normalize=True, sort=False)

        # Add small value to avoid log(0)
        ref_counts = ref_counts + 0.0001
        cur_counts = cur_counts + 0.0001

        # Normalize
        ref_counts = ref_counts / ref_counts.sum()
        cur_counts = cur_counts / cur_counts.sum()

        # Calculate PSI
        psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))

        return float(psi)

    def _calculate_ks(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        from scipy import stats
        statistic, _ = stats.ks_2samp(reference, current)
        return float(statistic)

    def _calculate_wasserstein(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> float:
        """Calculate Wasserstein (Earth Mover's) distance."""
        from scipy import stats
        distance = stats.wasserstein_distance(reference, current)

        # Normalize by reference std
        ref_std = reference.std()
        if ref_std > 0:
            distance = distance / ref_std

        return float(distance)


class PredictionMonitor:
    """
    Monitor model predictions in real-time.

    Tracks:
    - Prediction distribution
    - Latency
    - Error rates
    - Anomalous predictions
    """

    def __init__(
        self,
        model_id: str,
        window_size: int = 1000,
        anomaly_threshold: float = 3.0,  # Z-score threshold
    ):
        self.model_id = model_id
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold

        self._predictions = deque(maxlen=window_size)
        self._latencies = deque(maxlen=window_size)
        self._errors = deque(maxlen=window_size)
        self._timestamps = deque(maxlen=window_size)

        # Reference statistics (from training)
        self._reference_mean: Optional[float] = None
        self._reference_std: Optional[float] = None

    def set_reference_stats(self, mean: float, std: float) -> None:
        """Set reference prediction statistics from training."""
        self._reference_mean = mean
        self._reference_std = std

    def record_prediction(
        self,
        prediction: float,
        latency_ms: float,
        is_error: bool = False,
    ) -> Optional[Alert]:
        """
        Record a prediction and check for anomalies.

        Returns Alert if anomaly detected.
        """
        timestamp = datetime.now()

        self._predictions.append(prediction)
        self._latencies.append(latency_ms)
        self._errors.append(is_error)
        self._timestamps.append(timestamp)

        # Check for anomalies
        alert = self._check_anomalies(prediction, latency_ms)

        return alert

    def _check_anomalies(
        self,
        prediction: float,
        latency_ms: float,
    ) -> Optional[Alert]:
        """Check for anomalous predictions or latency."""
        # Check prediction anomaly
        if self._reference_mean is not None and self._reference_std is not None:
            z_score = abs(prediction - self._reference_mean) / max(self._reference_std, 0.001)

            if z_score > self.anomaly_threshold:
                return Alert(
                    alert_id=self._generate_alert_id(),
                    timestamp=datetime.now(),
                    severity=AlertSeverity.WARNING,
                    metric_name="prediction_zscore",
                    metric_value=z_score,
                    threshold=self.anomaly_threshold,
                    message=f"Anomalous prediction detected: z-score={z_score:.2f}",
                    model_id=self.model_id,
                )

        # Check latency spike
        if len(self._latencies) > 10:
            latency_p95 = np.percentile(list(self._latencies), 95)
            if latency_ms > latency_p95 * 2:
                return Alert(
                    alert_id=self._generate_alert_id(),
                    timestamp=datetime.now(),
                    severity=AlertSeverity.WARNING,
                    metric_name="latency_spike",
                    metric_value=latency_ms,
                    threshold=latency_p95 * 2,
                    message=f"Latency spike detected: {latency_ms:.0f}ms (p95={latency_p95:.0f}ms)",
                    model_id=self.model_id,
                )

        return None

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{self.model_id}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def get_metrics(self) -> ModelMetrics:
        """Get current monitoring metrics."""
        predictions = list(self._predictions)
        latencies = list(self._latencies)
        errors = list(self._errors)

        return ModelMetrics(
            model_id=self.model_id,
            timestamp=datetime.now(),
            prediction_count=len(predictions),
            mean_prediction=np.mean(predictions) if predictions else 0.0,
            std_prediction=np.std(predictions) if predictions else 0.0,
            mean_latency_ms=np.mean(latencies) if latencies else 0.0,
            p50_latency_ms=np.percentile(latencies, 50) if latencies else 0.0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0.0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0.0,
            error_count=sum(errors),
            error_rate=sum(errors) / len(errors) if errors else 0.0,
        )

    def get_prediction_distribution(self) -> dict:
        """Get prediction distribution statistics."""
        predictions = list(self._predictions)

        if not predictions:
            return {}

        return {
            "count": len(predictions),
            "mean": np.mean(predictions),
            "std": np.std(predictions),
            "min": np.min(predictions),
            "max": np.max(predictions),
            "p10": np.percentile(predictions, 10),
            "p50": np.percentile(predictions, 50),
            "p90": np.percentile(predictions, 90),
        }


class SLAMonitor:
    """
    Service Level Agreement monitoring.

    Tracks:
    - Availability
    - Response time SLAs
    - Accuracy SLAs
    - Error rate SLAs
    """

    def __init__(
        self,
        model_id: str,
        latency_sla_ms: float = 100.0,
        error_rate_sla: float = 0.01,
        accuracy_sla: Optional[float] = None,
    ):
        self.model_id = model_id
        self.latency_sla_ms = latency_sla_ms
        self.error_rate_sla = error_rate_sla
        self.accuracy_sla = accuracy_sla

        self._request_count = 0
        self._error_count = 0
        self._latency_violations = 0
        self._accuracy_violations = 0

        self._start_time = datetime.now()
        self._alerts: list[Alert] = []

    def record_request(
        self,
        latency_ms: float,
        is_error: bool = False,
        accuracy: Optional[float] = None,
    ) -> Optional[Alert]:
        """Record a request and check SLA compliance."""
        self._request_count += 1

        alert = None

        if is_error:
            self._error_count += 1
            error_rate = self._error_count / self._request_count

            if error_rate > self.error_rate_sla:
                alert = self._create_alert(
                    "error_rate",
                    error_rate,
                    self.error_rate_sla,
                    AlertSeverity.CRITICAL,
                    f"Error rate SLA violated: {error_rate:.2%} > {self.error_rate_sla:.2%}",
                )

        if latency_ms > self.latency_sla_ms:
            self._latency_violations += 1
            violation_rate = self._latency_violations / self._request_count

            if violation_rate > 0.05:  # More than 5% violations
                alert = self._create_alert(
                    "latency_sla",
                    latency_ms,
                    self.latency_sla_ms,
                    AlertSeverity.WARNING,
                    f"Latency SLA violation rate: {violation_rate:.1%}",
                )

        if accuracy is not None and self.accuracy_sla is not None:
            if accuracy < self.accuracy_sla:
                self._accuracy_violations += 1
                alert = self._create_alert(
                    "accuracy_sla",
                    accuracy,
                    self.accuracy_sla,
                    AlertSeverity.CRITICAL,
                    f"Accuracy SLA violated: {accuracy:.2%} < {self.accuracy_sla:.2%}",
                )

        if alert:
            self._alerts.append(alert)

        return alert

    def _create_alert(
        self,
        metric: str,
        value: float,
        threshold: float,
        severity: AlertSeverity,
        message: str,
    ) -> Alert:
        """Create an SLA alert."""
        return Alert(
            alert_id=hashlib.md5(f"{self.model_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:12],
            timestamp=datetime.now(),
            severity=severity,
            metric_name=metric,
            metric_value=value,
            threshold=threshold,
            message=message,
            model_id=self.model_id,
        )

    def get_sla_report(self) -> dict:
        """Get SLA compliance report."""
        uptime = (datetime.now() - self._start_time).total_seconds()

        return {
            "model_id": self.model_id,
            "period_start": self._start_time.isoformat(),
            "period_end": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "total_requests": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / self._request_count if self._request_count > 0 else 0,
            "error_rate_sla": self.error_rate_sla,
            "error_sla_compliance": self._error_count / self._request_count <= self.error_rate_sla if self._request_count > 0 else True,
            "latency_violations": self._latency_violations,
            "latency_violation_rate": self._latency_violations / self._request_count if self._request_count > 0 else 0,
            "latency_sla_ms": self.latency_sla_ms,
            "total_alerts": len(self._alerts),
        }


class AlertManager:
    """
    Centralized alert management.

    Features:
    - Alert aggregation
    - Deduplication
    - Notification routing
    - Acknowledgment tracking
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._alerts = []
            cls._instance._handlers = []
            cls._instance._dedup_window_seconds = 300  # 5 minutes
        return cls._instance

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler (e.g., for notifications)."""
        self._handlers.append(handler)

    def fire_alert(self, alert: Alert) -> bool:
        """
        Fire an alert.

        Returns False if deduplicated.
        """
        # Check for duplicate
        if self._is_duplicate(alert):
            logger.debug(f"Deduplicated alert: {alert.metric_name}")
            return False

        self._alerts.append(alert)

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.warning(
            "Alert fired",
            severity=alert.severity.value,
            metric=alert.metric_name,
            value=alert.metric_value,
        )

        return True

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is duplicate of recent alert."""
        cutoff = datetime.now() - timedelta(seconds=self._dedup_window_seconds)

        for existing in reversed(self._alerts):
            if existing.timestamp < cutoff:
                break

            if (existing.model_id == alert.model_id and
                existing.metric_name == alert.metric_name and
                existing.severity == alert.severity):
                return True

        return False

    def acknowledge(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
        return False

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        model_id: Optional[str] = None,
    ) -> list[Alert]:
        """Get active (unacknowledged) alerts."""
        alerts = [a for a in self._alerts if not a.acknowledged]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]

        return alerts

    def get_alert_summary(self) -> dict:
        """Get summary of all alerts."""
        active = [a for a in self._alerts if not a.acknowledged]

        return {
            "total_alerts": len(self._alerts),
            "active_alerts": len(active),
            "acknowledged_alerts": len(self._alerts) - len(active),
            "by_severity": {
                "info": len([a for a in active if a.severity == AlertSeverity.INFO]),
                "warning": len([a for a in active if a.severity == AlertSeverity.WARNING]),
                "critical": len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
            },
        }


class ModelMonitoringService:
    """
    High-level monitoring service for ML models.

    Integrates:
    - Drift detection
    - Prediction monitoring
    - SLA tracking
    - Alert management
    """

    def __init__(
        self,
        model_id: str,
        latency_sla_ms: float = 100.0,
        drift_threshold: float = 0.1,
    ):
        self.model_id = model_id

        self.drift_detector = DriftDetector(drift_threshold=drift_threshold)
        self.prediction_monitor = PredictionMonitor(model_id)
        self.sla_monitor = SLAMonitor(model_id, latency_sla_ms=latency_sla_ms)
        self.alert_manager = AlertManager()

    def set_reference_data(self, data: pd.DataFrame, predictions: Optional[np.ndarray] = None) -> None:
        """Set reference data for monitoring."""
        self.drift_detector.set_reference(data)

        if predictions is not None:
            self.prediction_monitor.set_reference_stats(
                mean=float(np.mean(predictions)),
                std=float(np.std(predictions)),
            )

    def record_prediction(
        self,
        prediction: float,
        latency_ms: float,
        is_error: bool = False,
    ) -> list[Alert]:
        """Record a prediction and return any alerts."""
        alerts = []

        # Check prediction
        pred_alert = self.prediction_monitor.record_prediction(prediction, latency_ms, is_error)
        if pred_alert:
            alerts.append(pred_alert)

        # Check SLA
        sla_alert = self.sla_monitor.record_request(latency_ms, is_error)
        if sla_alert:
            alerts.append(sla_alert)

        # Fire alerts
        for alert in alerts:
            self.alert_manager.fire_alert(alert)

        return alerts

    def check_drift(self, current_data: pd.DataFrame) -> list[DriftResult]:
        """Check for data drift."""
        results = self.drift_detector.detect_drift(current_data)

        # Fire alerts for drifted features
        for result in results:
            if result.is_drifted:
                alert = Alert(
                    alert_id=hashlib.md5(f"{self.model_id}_drift_{result.feature_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:12],
                    timestamp=datetime.now(),
                    severity=AlertSeverity.WARNING,
                    metric_name="feature_drift",
                    metric_value=result.drift_score,
                    threshold=result.threshold,
                    message=f"Drift detected in feature '{result.feature_name}': score={result.drift_score:.3f}",
                    model_id=self.model_id,
                )
                self.alert_manager.fire_alert(alert)

        return results

    def get_health_status(self) -> dict:
        """Get overall model health status."""
        metrics = self.prediction_monitor.get_metrics()
        sla_report = self.sla_monitor.get_sla_report()
        alert_summary = self.alert_manager.get_alert_summary()

        # Determine health
        if alert_summary["by_severity"]["critical"] > 0:
            health = "critical"
        elif alert_summary["by_severity"]["warning"] > 0:
            health = "degraded"
        else:
            health = "healthy"

        return {
            "model_id": self.model_id,
            "health": health,
            "metrics": metrics.to_dict(),
            "sla_compliance": sla_report,
            "alerts": alert_summary,
        }

