"""Model retraining tasks."""

from meridian.core.logging import get_logger
from meridian.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, max_retries=3)
def retrain_uplift_model(self, model_id: str, training_data_path: str):
    """Retrain uplift model."""
    try:
        logger.info("Starting model retraining", model_id=model_id)

        # Would load training data and retrain model
        # Then register with MLflow

        logger.info("Model retraining complete", model_id=model_id)
        return {"status": "success", "model_id": model_id}

    except Exception as e:
        logger.error("Model retraining failed", error=str(e))
        self.retry(exc=e, countdown=60)


@celery_app.task(bind=True)
def retrain_forecasting_model(self, model_id: str, item_ids: list[str]):
    """Retrain forecasting model."""
    try:
        logger.info(
            "Starting forecasting model retraining",
            model_id=model_id,
            n_items=len(item_ids),
        )

        # Retrain logic here

        logger.info("Forecasting model retraining complete")
        return {"status": "success", "model_id": model_id}

    except Exception as e:
        logger.error("Forecasting retraining failed", error=str(e))
        raise
