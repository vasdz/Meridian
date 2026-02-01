"""Health check endpoints."""

from datetime import datetime

from fastapi import APIRouter, status
from pydantic import BaseModel

from meridian.core.config import settings
from meridian.infrastructure.database.connection import check_db_connection
from meridian.infrastructure.cache.redis_cache import check_redis_connection


router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    environment: str
    timestamp: datetime


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    status: str
    database: str
    redis: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.utcnow(),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """Check if application is ready to accept requests."""
    db_ok = await check_db_connection()
    redis_ok = await check_redis_connection()

    all_ok = db_ok and redis_ok

    return ReadinessResponse(
        status="ready" if all_ok else "not_ready",
        database="ok" if db_ok else "error",
        redis="ok" if redis_ok else "error",
    )


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # In production, use prometheus_client
    return {
        "requests_total": 0,
        "request_duration_seconds": 0,
    }

