"""ML Monitoring endpoints - Production observability API."""

from typing import Annotated, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from meridian.api.dependencies.auth import get_current_user, TokenData
from meridian.core.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()


# Monitoring schemas

class ModelHealthStatus(BaseModel):
    """Health status of a model."""
    model_id: str
    status: str  # healthy, degraded, critical
    last_prediction_time: Optional[datetime] = None
    prediction_count_24h: int
    error_rate_24h: float
    avg_latency_ms: float
    p95_latency_ms: float
    sla_compliance: bool


class DriftStatus(BaseModel):
    """Drift detection status."""
    feature_name: str
    drift_score: float
    threshold: float
    is_drifted: bool
    drift_type: str
    last_checked: datetime


class ModelDriftReport(BaseModel):
    """Complete drift report for a model."""
    model_id: str
    check_timestamp: datetime
    overall_drift_detected: bool
    drifted_features: int
    total_features: int
    feature_drift: list[DriftStatus]


class AlertInfo(BaseModel):
    """Alert information."""
    alert_id: str
    timestamp: datetime
    severity: str
    metric_name: str
    metric_value: float
    threshold: float
    message: str
    model_id: Optional[str] = None
    acknowledged: bool


class AlertSummary(BaseModel):
    """Summary of alerts."""
    total_alerts: int
    active_alerts: int
    by_severity: dict[str, int]


class SLAReport(BaseModel):
    """SLA compliance report."""
    model_id: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    error_count: int
    error_rate: float
    error_rate_sla: float
    error_sla_compliance: bool
    latency_violations: int
    latency_violation_rate: float
    latency_sla_ms: float
    availability_pct: float


class MetricsSnapshot(BaseModel):
    """Current metrics snapshot."""
    model_id: str
    timestamp: datetime
    prediction_count: int
    mean_prediction: float
    std_prediction: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_count: int
    error_rate: float


# In-memory storage for demo (would be database in production)
_model_metrics: dict[str, list[dict]] = {}
_alerts: list[AlertInfo] = []


@router.get("/health/{model_id}", response_model=ModelHealthStatus)
async def get_model_health(
    model_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """
    Get health status for a specific model.

    Returns current health status, performance metrics, and SLA compliance.
    """
    import random

    logger.info("Model health check", model_id=model_id)

    # Simulated health data
    error_rate = random.uniform(0.001, 0.02)
    avg_latency = random.uniform(20, 80)

    status = "healthy"
    if error_rate > 0.01:
        status = "degraded"
    if error_rate > 0.05:
        status = "critical"

    return ModelHealthStatus(
        model_id=model_id,
        status=status,
        last_prediction_time=datetime.now(),
        prediction_count_24h=random.randint(10000, 100000),
        error_rate_24h=round(error_rate, 4),
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(avg_latency * 1.5, 2),
        sla_compliance=error_rate < 0.01 and avg_latency < 100,
    )


@router.get("/health")
async def get_all_models_health(
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Get health status for all registered models."""
    import random

    models = ["uplift_v2", "demand_forecast_v2", "pricing_v1"]

    health_statuses = []
    for model_id in models:
        error_rate = random.uniform(0.001, 0.02)
        avg_latency = random.uniform(20, 80)

        status = "healthy"
        if error_rate > 0.01:
            status = "degraded"

        health_statuses.append({
            "model_id": model_id,
            "status": status,
            "error_rate_24h": round(error_rate, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "sla_compliance": error_rate < 0.01,
        })

    return {
        "models": health_statuses,
        "overall_status": "healthy" if all(m["status"] == "healthy" for m in health_statuses) else "degraded",
        "timestamp": datetime.now(),
    }


@router.get("/drift/{model_id}", response_model=ModelDriftReport)
async def check_drift(
    model_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """
    Check for data drift in model features.

    Compares current feature distributions against reference (training) data.
    """
    import random

    logger.info("Drift check", model_id=model_id)

    # Simulated drift detection
    features = ["age", "income", "purchase_frequency", "days_since_last_purchase", "avg_basket_size"]

    feature_drift = []
    drifted_count = 0

    for feature in features:
        drift_score = random.uniform(0.01, 0.15)
        threshold = 0.1
        is_drifted = drift_score > threshold

        if is_drifted:
            drifted_count += 1

        feature_drift.append(DriftStatus(
            feature_name=feature,
            drift_score=round(drift_score, 4),
            threshold=threshold,
            is_drifted=is_drifted,
            drift_type="psi",
            last_checked=datetime.now(),
        ))

    return ModelDriftReport(
        model_id=model_id,
        check_timestamp=datetime.now(),
        overall_drift_detected=drifted_count > 0,
        drifted_features=drifted_count,
        total_features=len(features),
        feature_drift=feature_drift,
    )


@router.get("/alerts", response_model=list[AlertInfo])
async def get_alerts(
    current_user: Annotated[TokenData, Depends(get_current_user)],
    severity: Optional[str] = Query(None, description="Filter by severity"),
    model_id: Optional[str] = Query(None, description="Filter by model"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment status"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Get monitoring alerts.

    Supports filtering by severity, model, and acknowledgment status.
    """
    import random

    # Generate sample alerts for demo
    sample_alerts = [
        AlertInfo(
            alert_id=f"alert-{i}",
            timestamp=datetime.now(),
            severity=random.choice(["info", "warning", "critical"]),
            metric_name=random.choice(["latency", "error_rate", "drift_score"]),
            metric_value=random.uniform(0.1, 0.5),
            threshold=0.1,
            message="Metric exceeded threshold",
            model_id=random.choice(["uplift_v2", "demand_forecast_v2"]),
            acknowledged=random.choice([True, False]),
        )
        for i in range(10)
    ]

    # Apply filters
    alerts = sample_alerts

    if severity:
        alerts = [a for a in alerts if a.severity == severity]

    if model_id:
        alerts = [a for a in alerts if a.model_id == model_id]

    if acknowledged is not None:
        alerts = [a for a in alerts if a.acknowledged == acknowledged]

    return alerts[:limit]


@router.get("/alerts/summary", response_model=AlertSummary)
async def get_alert_summary(
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Get summary of all alerts."""
    import random

    return AlertSummary(
        total_alerts=random.randint(10, 50),
        active_alerts=random.randint(0, 10),
        by_severity={
            "info": random.randint(0, 5),
            "warning": random.randint(0, 3),
            "critical": random.randint(0, 2),
        },
    )


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Acknowledge an alert."""
    logger.info("Alert acknowledged", alert_id=alert_id, user=current_user.user_id)

    return {
        "alert_id": alert_id,
        "acknowledged": True,
        "acknowledged_by": current_user.user_id,
        "acknowledged_at": datetime.now(),
    }


@router.get("/sla/{model_id}", response_model=SLAReport)
async def get_sla_report(
    model_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
    period_days: int = Query(7, ge=1, le=90),
):
    """
    Get SLA compliance report for a model.

    Reports on latency, error rate, and availability SLAs.
    """
    import random
    from datetime import timedelta

    logger.info("SLA report", model_id=model_id, period_days=period_days)

    total_requests = random.randint(100000, 1000000)
    error_count = int(total_requests * random.uniform(0.001, 0.02))
    latency_violations = int(total_requests * random.uniform(0.01, 0.05))

    return SLAReport(
        model_id=model_id,
        period_start=datetime.now() - timedelta(days=period_days),
        period_end=datetime.now(),
        total_requests=total_requests,
        error_count=error_count,
        error_rate=round(error_count / total_requests, 4),
        error_rate_sla=0.01,
        error_sla_compliance=error_count / total_requests < 0.01,
        latency_violations=latency_violations,
        latency_violation_rate=round(latency_violations / total_requests, 4),
        latency_sla_ms=100.0,
        availability_pct=round(100 - (error_count / total_requests * 100), 2),
    )


@router.get("/metrics/{model_id}", response_model=MetricsSnapshot)
async def get_metrics(
    model_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Get current metrics snapshot for a model."""
    import random

    return MetricsSnapshot(
        model_id=model_id,
        timestamp=datetime.now(),
        prediction_count=random.randint(1000, 10000),
        mean_prediction=round(random.uniform(0.1, 0.5), 4),
        std_prediction=round(random.uniform(0.05, 0.2), 4),
        mean_latency_ms=round(random.uniform(20, 50), 2),
        p50_latency_ms=round(random.uniform(15, 40), 2),
        p95_latency_ms=round(random.uniform(50, 100), 2),
        p99_latency_ms=round(random.uniform(80, 150), 2),
        error_count=random.randint(0, 10),
        error_rate=round(random.uniform(0.001, 0.01), 4),
    )


@router.get("/metrics/{model_id}/history")
async def get_metrics_history(
    model_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
    hours: int = Query(24, ge=1, le=168),
):
    """Get historical metrics for a model."""
    import random
    from datetime import timedelta

    history = []
    now = datetime.now()

    for i in range(hours):
        timestamp = now - timedelta(hours=hours - i)
        history.append({
            "timestamp": timestamp.isoformat(),
            "prediction_count": random.randint(100, 500),
            "mean_latency_ms": round(random.uniform(20, 60), 2),
            "error_rate": round(random.uniform(0.001, 0.01), 4),
        })

    return {
        "model_id": model_id,
        "period_hours": hours,
        "data_points": len(history),
        "history": history,
    }


@router.post("/record")
async def record_prediction(
    current_user: Annotated[TokenData, Depends(get_current_user)],
    model_id: str,
    prediction: float,
    latency_ms: float,
    is_error: bool = False,
):
    """
    Record a prediction for monitoring.

    Used by inference services to report predictions.
    """
    logger.debug(
        "Recording prediction",
        model_id=model_id,
        prediction=prediction,
        latency_ms=latency_ms,
    )

    if model_id not in _model_metrics:
        _model_metrics[model_id] = []

    _model_metrics[model_id].append({
        "timestamp": datetime.now().isoformat(),
        "prediction": prediction,
        "latency_ms": latency_ms,
        "is_error": is_error,
    })

    # Keep only last 10000 records per model
    if len(_model_metrics[model_id]) > 10000:
        _model_metrics[model_id] = _model_metrics[model_id][-10000:]

    return {"status": "recorded", "model_id": model_id}

