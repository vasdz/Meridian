"""Celery application configuration."""

from celery import Celery

from meridian.core.config import settings


# Create Celery app
celery_app = Celery(
    "meridian",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=[
        "meridian.workers.tasks.model_retrain",
        "meridian.workers.tasks.feature_computation",
        "meridian.workers.tasks.experiment_analysis",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "daily-feature-computation": {
        "task": "meridian.workers.tasks.feature_computation.compute_daily_features",
        "schedule": 86400.0,  # Every 24 hours
    },
    "hourly-experiment-check": {
        "task": "meridian.workers.tasks.experiment_analysis.check_running_experiments",
        "schedule": 3600.0,  # Every hour
    },
}

