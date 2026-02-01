"""T-Learner implementation for uplift modeling."""

import numpy as np
from typing import Optional

from meridian.infrastructure.ml.uplift.meta_learner import MetaLearner
from meridian.core.logging import get_logger


logger = get_logger(__name__)


class TLearner(MetaLearner):
    """
    T-Learner (Two-Model Learner) for heterogeneous treatment effect estimation.

    The simplest meta-learner approach:
    1. Fit separate models for treated and control groups
    2. Estimate CATE as: τ(x) = μ1(x) - μ0(x)

    Pros:
    - Simple and interpretable
    - Works well when treatment/control groups are similar

    Cons:
    - Ignores shared structure between groups
    - Can overfit when groups are small
    """

    def __init__(
        self,
        model_id: str = "t_learner",
        base_learner: str = "gradient_boosting",
        **kwargs,
    ):
        super().__init__(model_id, base_learner, **kwargs)

        # Two separate models
        self._mu0: Optional[object] = None  # E[Y|X, T=0]
        self._mu1: Optional[object] = None  # E[Y|X, T=1]

        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> "TLearner":
        """
        Fit the T-Learner.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            treatment: Binary treatment indicator (0 or 1)
            y: Outcome variable

        Returns:
            self: Fitted estimator
        """
        logger.info(
            "Fitting T-Learner",
            n_samples=len(y),
            n_features=X.shape[1] if len(X.shape) > 1 else 1,
        )

        X = np.asarray(X)
        treatment = np.asarray(treatment)
        y = np.asarray(y)

        # Validate inputs
        self._validate_inputs(X, treatment, y)

        # Split data by treatment
        mask_treated = treatment == 1
        mask_control = treatment == 0

        X_treated = X[mask_treated]
        X_control = X[mask_control]
        y_treated = y[mask_treated]
        y_control = y[mask_control]

        logger.debug(
            "Data split",
            n_treated=len(y_treated),
            n_control=len(y_control),
        )

        # Fit control model: E[Y|X, T=0]
        self._mu0 = self._get_base_learner()
        self._mu0.fit(X_control, y_control)

        # Fit treatment model: E[Y|X, T=1]
        self._mu1 = self._get_base_learner()
        self._mu1.fit(X_treated, y_treated)

        self._is_fitted = True

        logger.info("T-Learner fitted successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE (Conditional Average Treatment Effect).

        Args:
            X: Feature matrix

        Returns:
            Array of CATE predictions: τ(x) = μ1(x) - μ0(x)
        """
        self._check_is_fitted()

        X = np.asarray(X)

        # Predict outcomes under each condition
        y0_pred = self._mu0.predict(X)
        y1_pred = self._mu1.predict(X)

        # CATE = E[Y|X, T=1] - E[Y|X, T=0]
        cate = y1_pred - y0_pred

        return cate

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict() - required by base class."""
        return self.predict(X)

    def predict_outcomes(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Predict potential outcomes under treatment and control.

        Args:
            X: Feature matrix

        Returns:
            Dictionary with 'y0', 'y1', and 'cate' predictions
        """
        self._check_is_fitted()

        X = np.asarray(X)

        y0_pred = self._mu0.predict(X)
        y1_pred = self._mu1.predict(X)

        return {
            "y0": y0_pred,
            "y1": y1_pred,
            "cate": y1_pred - y0_pred,
        }

    def _validate_inputs(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Validate input arrays."""
        if len(X) != len(treatment) or len(X) != len(y):
            raise ValueError("X, treatment, and y must have the same length")

        unique_treatments = np.unique(treatment)
        if not np.array_equal(unique_treatments, [0, 1]):
            if not (set(unique_treatments) <= {0, 1}):
                raise ValueError("treatment must be binary (0 or 1)")

        if np.sum(treatment == 1) < 10:
            logger.warning("Very few treated samples", n_treated=np.sum(treatment == 1))

        if np.sum(treatment == 0) < 10:
            logger.warning("Very few control samples", n_control=np.sum(treatment == 0))

    def _check_is_fitted(self) -> None:
        """Check if the model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def get_feature_importance(self) -> dict[str, np.ndarray]:
        """
        Get feature importance from both models.

        Returns:
            Dictionary with feature importances for control and treatment models
        """
        self._check_is_fitted()

        importance = {}

        if hasattr(self._mu0, "feature_importances_"):
            importance["control"] = self._mu0.feature_importances_
            importance["treatment"] = self._mu1.feature_importances_
        elif hasattr(self._mu0, "coef_"):
            importance["control"] = np.abs(self._mu0.coef_)
            importance["treatment"] = np.abs(self._mu1.coef_)
        else:
            logger.warning("Base learner does not support feature importance")
            return {}

        # Average importance across both models
        importance["average"] = (importance["control"] + importance["treatment"]) / 2

        return importance

