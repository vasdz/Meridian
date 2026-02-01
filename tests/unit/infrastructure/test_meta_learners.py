"""Tests for uplift meta-learners: T-Learner, S-Learner."""

import numpy as np
import pytest

from meridian.infrastructure.ml.uplift import SLearner, TLearner


class TestTLearner:
    """Tests for T-Learner uplift model."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        treatment = np.random.randint(0, 2, n_samples)
        # True effect is ~0.3
        y = X[:, 0] * 0.5 + treatment * 0.3 + np.random.randn(n_samples) * 0.1

        return X, treatment, y

    def test_fit_predict(self, sample_data):
        """Test that T-Learner can fit and predict."""
        X, treatment, y = sample_data

        model = TLearner()
        model.fit(X, treatment, y)

        cate = model.predict(X)

        assert len(cate) == len(X)
        assert not np.any(np.isnan(cate))

    def test_predict_cate_alias(self, sample_data):
        """Test that predict_cate is alias for predict."""
        X, treatment, y = sample_data

        model = TLearner()
        model.fit(X, treatment, y)

        cate1 = model.predict(X[:10])
        cate2 = model.predict_cate(X[:10])

        np.testing.assert_array_equal(cate1, cate2)

    def test_predict_outcomes(self, sample_data):
        """Test predict_outcomes returns y0, y1, and cate."""
        X, treatment, y = sample_data

        model = TLearner()
        model.fit(X, treatment, y)

        outcomes = model.predict_outcomes(X[:10])

        assert "y0" in outcomes
        assert "y1" in outcomes
        assert "cate" in outcomes
        np.testing.assert_array_almost_equal(outcomes["cate"], outcomes["y1"] - outcomes["y0"])

    def test_positive_treatment_effect(self, sample_data):
        """Test that model detects positive treatment effect."""
        X, treatment, y = sample_data

        model = TLearner()
        model.fit(X, treatment, y)

        cate = model.predict(X)
        mean_cate = np.mean(cate)

        # True effect is 0.3, should be detected
        assert mean_cate > 0.1

    def test_not_fitted_error(self, sample_data):
        """Test that predict before fit raises error."""
        X, _, _ = sample_data

        model = TLearner()

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_invalid_treatment_values(self, sample_data):
        """Test that invalid treatment values raise error."""
        X, _, y = sample_data
        treatment_invalid = np.array([0, 1, 2, 0, 1] * 40)  # Contains 2

        model = TLearner()

        with pytest.raises(ValueError, match="binary"):
            model.fit(X, treatment_invalid, y)


class TestSLearner:
    """Tests for S-Learner uplift model."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        treatment = np.random.randint(0, 2, n_samples)
        y = X[:, 0] * 0.5 + treatment * 0.3 + np.random.randn(n_samples) * 0.1

        return X, treatment, y

    def test_fit_predict(self, sample_data):
        """Test that S-Learner can fit and predict."""
        X, treatment, y = sample_data

        model = SLearner()
        model.fit(X, treatment, y)

        cate = model.predict(X)

        assert len(cate) == len(X)
        assert not np.any(np.isnan(cate))

    def test_predict_cate_alias(self, sample_data):
        """Test that predict_cate is alias for predict."""
        X, treatment, y = sample_data

        model = SLearner()
        model.fit(X, treatment, y)

        cate1 = model.predict(X[:10])
        cate2 = model.predict_cate(X[:10])

        np.testing.assert_array_equal(cate1, cate2)

    def test_feature_augmentation(self, sample_data):
        """Test that treatment feature is properly added."""
        X, treatment, y = sample_data

        model = SLearner(treatment_feature_position="last")
        model.fit(X, treatment, y)

        # Check feature importance includes treatment
        importance = model.get_feature_importance()

        if importance:  # If base learner supports feature importance
            assert "treatment" in importance
            assert "features" in importance
            assert len(importance["features"]) == X.shape[1]
