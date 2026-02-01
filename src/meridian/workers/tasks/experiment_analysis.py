"""Experiment analysis tasks."""

from meridian.workers.celery_app import celery_app
from meridian.core.logging import get_logger


logger = get_logger(__name__)


@celery_app.task
def check_running_experiments():
    """Check running experiments for significance."""
    logger.info("Checking running experiments")

    # Would query running experiments and analyze results

    return {"status": "success", "checked": 0}


@celery_app.task
def analyze_experiment(experiment_id: str):
    """Analyze specific experiment."""
    logger.info("Analyzing experiment", experiment_id=experiment_id)

    # Perform statistical analysis

    return {
        "experiment_id": experiment_id,
        "status": "success",
    }


@celery_app.task
def send_experiment_report(experiment_id: str, recipients: list[str]):
    """Send experiment report to stakeholders."""
    logger.info(
        "Sending experiment report",
        experiment_id=experiment_id,
        n_recipients=len(recipients),
    )

    # Would generate and send report

    return {"status": "success"}

