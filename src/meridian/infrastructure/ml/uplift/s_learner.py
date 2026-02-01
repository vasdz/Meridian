"""S-Learner implementation for uplift modeling."""

import numpy as np

from meridian.core.logging import get_logger
from meridian.infrastructure.ml.uplift.meta_learner import MetaLearner

logger = get_logger(__name__)


class SLearner(MetaLearner):
    """
    S-Learner (Single-Model Learner) for heterogeneous treatment effect estimation.

    Approach:
    1. Fit a single model with treatment as a feature: μ(x, t)
    2. Estimate CATE as: τ(x) = μ(x, 1) - μ(x, 0)

    Pros:
    - Shares information between treatment and control
    - More sample-efficient
    - Natural regularization

    Cons:
    - May underestimate treatment effects if treatment feature is regularized
    - Requires careful feature engineering
    """

    def __init__(
        self,
        model_id: str = "s_learner",
        base_learner: str = "gradient_boosting",
        treatment_feature_position: str = "last",  # 'first' or 'last'
        **kwargs,
    ):
        super().__init__(model_id, base_learner, **kwargs)

        self._model: object | None = None
        self._treatment_position = treatment_feature_position
        self._is_fitted = False
        self._n_features: int | None = None

    def _add_treatment_feature(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
    ) -> np.ndarray:
        """Add treatment indicator as a feature."""
        treatment = treatment.reshape(-1, 1)

        if self._treatment_position == "first":
            return np.hstack([treatment, X])
        else:
            return np.hstack([X, treatment])

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> "SLearner":
        """
        Fit the S-Learner.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            treatment: Binary treatment indicator (0 or 1)
            y: Outcome variable

        Returns:
            self: Fitted estimator
        """
        logger.info(
            "Fitting S-Learner",
            n_samples=len(y),
            n_features=X.shape[1] if len(X.shape) > 1 else 1,
        )

        X = np.asarray(X)
        treatment = np.asarray(treatment)
        y = np.asarray(y)

        self._n_features = X.shape[1] if len(X.shape) > 1 else 1

        # Validate inputs
        self._validate_inputs(X, treatment, y)

        # Augment features with treatment indicator
        X_augmented = self._add_treatment_feature(X, treatment)

        logger.debug(
            "Augmented feature matrix",
            original_features=self._n_features,
            augmented_features=X_augmented.shape[1],
        )

        # Fit single model
        self._model = self._get_base_learner()
        self._model.fit(X_augmented, y)

        self._is_fitted = True

        logger.info("S-Learner fitted successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE (Conditional Average Treatment Effect).

        Args:
            X: Feature matrix

        Returns:
            Array of CATE predictions: τ(x) = μ(x, 1) - μ(x, 0)
        """
        self._check_is_fitted()

        X = np.asarray(X)
        n_samples = X.shape[0]

        # Create treatment and control versions
        treatment_1 = np.ones(n_samples)
        treatment_0 = np.zeros(n_samples)

        X_treated = self._add_treatment_feature(X, treatment_1)
        X_control = self._add_treatment_feature(X, treatment_0)

        # Predict outcomes
        y1_pred = self._model.predict(X_treated)
        y0_pred = self._model.predict(X_control)

        # CATE = μ(x, 1) - μ(x, 0)
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
        n_samples = X.shape[0]

        treatment_1 = np.ones(n_samples)
        treatment_0 = np.zeros(n_samples)

        X_treated = self._add_treatment_feature(X, treatment_1)
        X_control = self._add_treatment_feature(X, treatment_0)

        y0_pred = self._model.predict(X_control)
        y1_pred = self._model.predict(X_treated)

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
        if not (set(unique_treatments) <= {0, 1}):
            raise ValueError("treatment must be binary (0 or 1)")

    def _check_is_fitted(self) -> None:
        """Check if the model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def get_feature_importance(self) -> dict[str, np.ndarray]:
        """
        Get feature importance from the model.

        Returns:
            Dictionary with feature importances
        """
        self._check_is_fitted()

        importance = {}

        if hasattr(self._model, "feature_importances_"):
            all_importance = self._model.feature_importances_

            if self._treatment_position == "first":
                importance["treatment"] = all_importance[0]
                importance["features"] = all_importance[1:]
            else:
                importance["features"] = all_importance[:-1]
                importance["treatment"] = all_importance[-1]
        elif hasattr(self._model, "coef_"):
            coef = np.abs(self._model.coef_).flatten()

            if self._treatment_position == "first":
                importance["treatment"] = coef[0]
                importance["features"] = coef[1:]
            else:
                importance["features"] = coef[:-1]
                importance["treatment"] = coef[-1]
        else:
            logger.warning("Base learner does not support feature importance")
            return {}

        return importance
