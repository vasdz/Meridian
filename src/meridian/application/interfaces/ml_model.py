"""ML model interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class MLModelInterface(ABC):
    """Interface for ML models."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Get model ID."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Get model version."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty bounds."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from path."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to path."""
        pass


class UpliftModelInterface(MLModelInterface):
    """Interface for uplift models."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fit the uplift model."""
        pass

    @abstractmethod
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE (Conditional Average Treatment Effect)."""
        pass


class ForecastingModelInterface(MLModelInterface):
    """Interface for forecasting models."""

    @abstractmethod
    def fit(self, time_series: Any, **kwargs) -> None:
        """Fit the forecasting model."""
        pass

    @abstractmethod
    def forecast(
        self,
        horizon: int,
        quantiles: list[float] = None,
    ) -> dict:
        """Generate forecasts."""
        pass

