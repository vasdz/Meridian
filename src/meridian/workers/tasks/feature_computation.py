"""Feature computation tasks."""

from meridian.workers.celery_app import celery_app
from meridian.core.logging import get_logger


logger = get_logger(__name__)


@celery_app.task
def compute_daily_features():
    """Compute daily feature updates."""
    logger.info("Starting daily feature computation")

    # Would trigger Spark job for feature computation

    logger.info("Daily feature computation complete")
    return {"status": "success"}


@celery_app.task
def compute_customer_features(customer_ids: list[str]):
    """Compute features for specific customers."""
    logger.info(
        "Computing features for customers",
        n_customers=len(customer_ids),
    )

    # Compute features

    return {"status": "success", "n_customers": len(customer_ids)}


@celery_app.task
def materialize_features(feature_view: str):
    """Materialize features to online store."""
    logger.info("Materializing features", feature_view=feature_view)

    # Would call feature store materialization

    return {"status": "success", "feature_view": feature_view}

