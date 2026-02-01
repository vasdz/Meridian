"""Price elasticity estimation model."""

import numpy as np

from meridian.core.logging import get_logger
from meridian.infrastructure.ml.base import BaseMLModel

logger = get_logger(__name__)


class ElasticityModel(BaseMLModel):
    """Price elasticity estimation using log-linear regression."""

    def __init__(
        self,
        model_id: str = "elasticity",
        regularization: float = 0.01,
        include_cross_elasticity: bool = False,
    ):
        super().__init__(model_id)
        self.regularization = regularization
        self.include_cross_elasticity = include_cross_elasticity
        self._coefficients = None

    def fit(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        X_controls: np.ndarray | None = None,
    ) -> None:
        """
        Fit elasticity model.

        Uses log-log regression: log(Q) = α + β·log(P) + γ·X + ε
        where β is the price elasticity.
        """
        logger.info("Fitting elasticity model")

        log_prices = np.log(prices + 1e-10)
        log_quantities = np.log(quantities + 1e-10)

        if X_controls is not None:
            X = np.column_stack([np.ones(len(prices)), log_prices, X_controls])
        else:
            X = np.column_stack([np.ones(len(prices)), log_prices])

        # Ridge regression
        from sklearn.linear_model import Ridge

        self._model = Ridge(alpha=self.regularization)
        self._model.fit(X, log_quantities)

        self._coefficients = self._model.coef_
        self._is_fitted = True

        logger.info(
            "Elasticity model fitted",
            elasticity=self.get_elasticity(),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log quantities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model.predict(X)

    def get_elasticity(self) -> float:
        """Get estimated price elasticity."""
        if self._coefficients is None:
            return -1.0
        return float(self._coefficients[1])

    def predict_demand_change(
        self,
        current_price: float,
        new_price: float,
    ) -> float:
        """Predict percent change in demand for price change."""
        elasticity = self.get_elasticity()
        price_change = (new_price - current_price) / current_price
        return elasticity * price_change
