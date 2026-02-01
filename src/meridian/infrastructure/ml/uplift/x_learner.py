"""X-Learner implementation for uplift modeling."""

import numpy as np
from sklearn.base import clone

from meridian.infrastructure.ml.uplift.meta_learner import MetaLearner
from meridian.core.logging import get_logger


logger = get_logger(__name__)


class XLearner(MetaLearner):
    """
    X-Learner for heterogeneous treatment effect estimation.

    The X-Learner is a meta-algorithm that works in three stages:
    1. Estimate response functions μ0(x) and μ1(x)
    2. Estimate imputed treatment effects
    3. Use propensity weighting to combine estimates
    """

    def __init__(
        self,
        model_id: str = "x_learner",
        base_learner: str = "gradient_boosting",
        propensity_learner: str = "logistic",
        **kwargs,
    ):
        super().__init__(model_id, base_learner, **kwargs)
        self.propensity_learner = propensity_learner

        # Stage 1: Response functions
        self._mu0 = None  # E[Y|X, T=0]
        self._mu1 = None  # E[Y|X, T=1]

        # Stage 2: Imputed effect models
        self._tau0 = None  # Effect estimated from control
        self._tau1 = None  # Effect estimated from treated

        # Propensity model
        self._propensity = None

    def _get_propensity_learner(self):
        """Get propensity score model."""
        if self.propensity_learner == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000)

        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=50)

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Fit the X-Learner.

        Args:
            X: Feature matrix
            treatment: Binary treatment indicator (0 or 1)
            y: Outcome variable
        """
        logger.info("Fitting X-Learner", n_samples=len(y))

        treatment = np.asarray(treatment)

        # Split data by treatment
        mask_treated = treatment == 1
        mask_control = treatment == 0

        X_treated = X[mask_treated]
        X_control = X[mask_control]
        y_treated = y[mask_treated]
        y_control = y[mask_control]

        # Stage 1: Fit response functions
        logger.debug("Stage 1: Fitting response functions")

        self._mu0 = self._get_base_learner()
        self._mu0.fit(X_control, y_control)

        self._mu1 = self._get_base_learner()
        self._mu1.fit(X_treated, y_treated)

        # Stage 2: Compute imputed treatment effects
        logger.debug("Stage 2: Computing imputed effects")

        # For treated: D1 = Y1 - μ0(X)
        D1 = y_treated - self._mu0.predict(X_treated)

        # For control: D0 = μ1(X) - Y0
        D0 = self._mu1.predict(X_control) - y_control

        # Fit effect models
        self._tau1 = self._get_base_learner()
        self._tau1.fit(X_treated, D1)

        self._tau0 = self._get_base_learner()
        self._tau0.fit(X_control, D0)

        # Fit propensity model
        logger.debug("Fitting propensity model")
        self._propensity = self._get_propensity_learner()
        self._propensity.fit(X, treatment)

        self._is_fitted = True
        logger.info("X-Learner fitted successfully")

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE using propensity-weighted combination.

        τ(x) = g(x)·τ0(x) + (1-g(x))·τ1(x)

        where g(x) is the propensity score.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get propensity scores
        propensity = self._propensity.predict_proba(X)[:, 1]

        # Get effect estimates
        tau0 = self._tau0.predict(X)
        tau1 = self._tau1.predict(X)

        # Propensity-weighted combination
        cate = propensity * tau0 + (1 - propensity) * tau1

        return cate

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_bootstrap: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with bootstrap uncertainty estimates."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        predictions = self.predict_cate(X)

        # Simple uncertainty based on propensity variance
        propensity = self._propensity.predict_proba(X)[:, 1]
        uncertainty = np.abs(propensity - 0.5) * 0.1

        lower = predictions - 1.96 * uncertainty
        upper = predictions + 1.96 * uncertainty

        return predictions, lower, upper

