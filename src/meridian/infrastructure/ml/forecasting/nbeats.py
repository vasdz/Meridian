"""N-BEATS forecasting model wrapper."""

from typing import Any

import numpy as np

from meridian.core.logging import get_logger
from meridian.infrastructure.ml.base import BaseMLModel

logger = get_logger(__name__)


class NBEATS(BaseMLModel):
    """N-BEATS neural network forecasting model."""

    def __init__(
        self,
        model_id: str = "nbeats",
        prediction_length: int = 14,
        context_length: int = 28,
        **kwargs,
    ):
        super().__init__(model_id)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.kwargs = kwargs

    def fit(self, time_series: Any, **kwargs) -> None:
        """Fit N-BEATS model."""
        logger.info("Fitting N-BEATS model")
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        # Mock predictions
        import random

        return np.array([random.uniform(80, 120) for _ in range(self.prediction_length)])

    def forecast(self, horizon: int, **kwargs) -> dict:
        """Generate forecasts."""
        predictions = self.predict(np.zeros(horizon))
        return {
            "mean": predictions.tolist(),
            "quantiles": {
                "0.1": (predictions * 0.8).tolist(),
                "0.5": predictions.tolist(),
                "0.9": (predictions * 1.2).tolist(),
            },
        }
