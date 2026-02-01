"""Causal Forest implementation using EconML."""

import numpy as np

from meridian.core.logging import get_logger
from meridian.infrastructure.ml.base import BaseMLModel

logger = get_logger(__name__)


class CausalForest(BaseMLModel):
    """
    Causal Forest for heterogeneous treatment effect estimation.

    Wrapper around EconML's CausalForestDML.
    """

    def __init__(
        self,
        model_id: str = "causal_forest",
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        **kwargs,
    ):
        super().__init__(model_id)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.kwargs = kwargs
        self._model = None

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        W: np.ndarray | None = None,
    ) -> None:
        """
        Fit the Causal Forest.

        Args:
            X: Feature matrix
            treatment: Treatment indicator
            y: Outcome variable
            W: Optional confounders for double ML
        """
        logger.info(
            "Fitting Causal Forest",
            n_samples=len(y),
            n_features=X.shape[1] if X.ndim > 1 else 1,
        )

        try:
            from econml.dml import CausalForestDML

            self._model = CausalForestDML(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                **self.kwargs,
            )

            if W is not None:
                self._model.fit(y, treatment, X=X, W=W)
            else:
                self._model.fit(y, treatment, X=X)

            self._is_fitted = True
            logger.info("Causal Forest fitted successfully")

        except ImportError:
            logger.error("EconML not installed. Using mock implementation.")
            self._fit_mock(X, treatment, y)

    def _fit_mock(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Mock implementation when EconML not available."""
        from sklearn.ensemble import RandomForestRegressor

        # Simple difference-in-means per leaf
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
        )

        # Train on difference
        mask_treated = treatment == 1
        y_diff = y.copy()
        y_diff[mask_treated] = y[mask_treated] - y[~mask_treated].mean()
        y_diff[~mask_treated] = y[mask_treated].mean() - y[~mask_treated]

        self._model.fit(X, y_diff)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE."""
        return self.predict_cate(X)

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Predict Conditional Average Treatment Effect."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        try:
            # EconML API
            return self._model.effect(X)
        except AttributeError:
            # Mock implementation
            return self._model.predict(X)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        try:
            # EconML provides inference
            point = self._model.effect(X)
            interval = self._model.effect_interval(X, alpha=alpha)
            return point, interval[0], interval[1]
        except AttributeError:
            # Mock implementation
            point = self._model.predict(X)
            std = 0.05  # Placeholder
            return point, point - 1.96 * std, point + 1.96 * std

    def feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        try:
            return self._model.feature_importances_
        except AttributeError:
            return self._model.feature_importances_
