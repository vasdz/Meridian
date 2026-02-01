"""Base ML model implementation."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from meridian.application.interfaces.ml_model import MLModelInterface


class BaseMLModel(MLModelInterface, ABC):
    """Base class for ML models."""

    def __init__(self, model_id: str, version: str = "1.0.0"):
        self._model_id = model_id
        self._version = version
        self._is_fitted = False
        self._model = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Default implementation returns point predictions with no uncertainty."""
        predictions = self.predict(X)
        return predictions, predictions, predictions

    def load(self, path: str) -> None:
        """Load model from path."""
        import joblib
        self._model = joblib.load(path)
        self._is_fitted = True

    def save(self, path: str) -> None:
        """Save model to path."""
        import joblib
        joblib.dump(self._model, path)

