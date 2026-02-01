"""MLflow model registry integration."""

from meridian.core.config import settings
from meridian.core.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """MLflow model registry adapter."""

    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self._client = None

    def _get_client(self):
        """Get MLflow client (lazy initialization)."""
        if self._client is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                self._client = mlflow.MlflowClient()
            except ImportError:
                logger.warning("MLflow not installed")
        return self._client

    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: str,
        **kwargs,
    ) -> str:
        """Log and register a model."""
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(self.tracking_uri)

            with mlflow.start_run():
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs,
                )
                run_id = mlflow.active_run().info.run_id

            logger.info(
                "Model logged",
                model_name=registered_model_name,
                run_id=run_id,
            )
            return run_id

        except Exception as e:
            logger.error("Failed to log model", error=str(e))
            raise

    def load_model(
        self,
        model_name: str,
        version: str | None = None,
        stage: str = "Production",
    ):
        """Load model from registry."""
        try:
            import mlflow

            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"

            model = mlflow.sklearn.load_model(model_uri)

            logger.info(
                "Model loaded",
                model_name=model_name,
                model_uri=model_uri,
            )
            return model

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise

    def get_latest_version(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> str | None:
        """Get latest model version for a stage."""
        client = self._get_client()
        if client is None:
            return None

        try:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0].version
            return None
        except Exception as e:
            logger.error("Failed to get model version", error=str(e))
            return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
    ) -> bool:
        """Transition model to a new stage."""
        client = self._get_client()
        if client is None:
            return False

        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )
            logger.info(
                "Model stage transitioned",
                model_name=model_name,
                version=version,
                stage=stage,
            )
            return True
        except Exception as e:
            logger.error("Failed to transition model", error=str(e))
            return False
