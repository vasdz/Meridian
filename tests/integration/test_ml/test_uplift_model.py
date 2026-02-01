"""Integration tests for uplift models."""

import numpy as np

from meridian.infrastructure.ml.uplift.causal_forest import CausalForest
from meridian.infrastructure.ml.uplift.x_learner import XLearner


class TestXLearnerIntegration:
    """Integration tests for X-Learner."""

    def test_fit_and_predict(self):
        """Test fitting and predicting with X-Learner."""
        # Generate synthetic data
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        treatment = np.random.binomial(1, 0.5, n)

        # True CATE: depends on first feature
        true_cate = 0.1 + 0.2 * X[:, 0]
        baseline = 0.5 + 0.1 * X[:, 1]
        y = baseline + treatment * true_cate + np.random.randn(n) * 0.1

        # Fit model
        model = XLearner(base_learner="gradient_boosting")
        model.fit(X, treatment, y)

        # Predict
        predictions = model.predict_cate(X[:10])

        assert len(predictions) == 10
        assert all(isinstance(p, float) for p in predictions)

    def test_predict_with_uncertainty(self):
        """Test uncertainty estimation."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 3)
        treatment = np.random.binomial(1, 0.5, n)
        y = treatment * 0.1 + np.random.randn(n) * 0.1

        model = XLearner()
        model.fit(X, treatment, y)

        point, lower, upper = model.predict_with_uncertainty(X[:5])

        assert len(point) == 5
        assert all(lower[i] <= point[i] <= upper[i] for i in range(5))


class TestCausalForestIntegration:
    """Integration tests for Causal Forest."""

    def test_fit_and_predict(self):
        """Test fitting and predicting with Causal Forest."""
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 5)
        treatment = np.random.binomial(1, 0.5, n)
        y = 0.5 + treatment * 0.1 + np.random.randn(n) * 0.1

        model = CausalForest(n_estimators=20)
        model.fit(X, treatment, y)

        predictions = model.predict(X[:10])

        assert len(predictions) == 10
