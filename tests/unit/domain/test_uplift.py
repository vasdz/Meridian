"""Unit tests for uplift domain model."""

import pytest
import numpy as np

from meridian.domain.models.uplift import UpliftPrediction, ConfidenceInterval
from meridian.domain.services.uplift_calculator import UpliftCalculator


class TestUpliftPrediction:
    """Tests for UpliftPrediction model."""

    def test_should_treat_positive_cate(self):
        """Test that positive CATE recommends treatment."""
        pred = UpliftPrediction(customer_id="c1", cate=0.05)
        assert pred.should_treat(threshold=0.0) is True

    def test_should_not_treat_negative_cate(self):
        """Test that negative CATE does not recommend treatment."""
        pred = UpliftPrediction(customer_id="c1", cate=-0.02)
        assert pred.should_treat(threshold=0.0) is False

    def test_should_treat_with_threshold(self):
        """Test treatment decision with custom threshold."""
        pred = UpliftPrediction(customer_id="c1", cate=0.03)
        assert pred.should_treat(threshold=0.05) is False
        assert pred.should_treat(threshold=0.02) is True

    def test_get_expected_value(self):
        """Test expected value calculation."""
        pred = UpliftPrediction(customer_id="c1", cate=0.10)
        expected = pred.get_expected_value(treatment_cost=0.02)
        assert expected == pytest.approx(0.08, rel=1e-5)

    def test_is_confident_narrow_interval(self):
        """Test confidence check with narrow interval."""
        pred = UpliftPrediction(
            customer_id="c1",
            cate=0.05,
            confidence_interval=ConfidenceInterval(lower=0.03, upper=0.07),
        )
        assert pred.is_confident(min_width=0.1) is True

    def test_is_confident_wide_interval(self):
        """Test confidence check with wide interval."""
        pred = UpliftPrediction(
            customer_id="c1",
            cate=0.05,
            confidence_interval=ConfidenceInterval(lower=-0.10, upper=0.20),
        )
        assert pred.is_confident(min_width=0.1) is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        pred = UpliftPrediction(customer_id="c1", cate=0.05)
        result = pred.to_dict()

        assert result["customer_id"] == "c1"
        assert result["cate"] == 0.05
        assert result["should_treat"] is True


class TestUpliftCalculator:
    """Tests for UpliftCalculator service."""

    def test_calculate_cate(self):
        """Test CATE calculation."""
        calculator = UpliftCalculator()

        y_treated = np.array([0.3, 0.4, 0.5, 0.6])
        y_control = np.array([0.2, 0.2, 0.3, 0.3])

        cate = calculator.calculate_cate(y_treated, y_control)

        # Mean(treated) - Mean(control) = 0.45 - 0.25 = 0.2
        assert cate == pytest.approx(0.2, rel=1e-5)

    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        calculator = UpliftCalculator()

        y_treated = np.array([0.3, 0.4, 0.5, 0.6])
        y_control = np.array([0.2, 0.2, 0.3, 0.3])
        cate = 0.2

        ci = calculator.calculate_confidence_interval(
            cate, y_treated, y_control, confidence_level=0.95
        )

        assert ci.lower < cate
        assert ci.upper > cate
        assert ci.level == 0.95

    def test_rank_customers_by_uplift(self):
        """Test customer ranking."""
        calculator = UpliftCalculator()

        predictions = [
            UpliftPrediction(customer_id="c1", cate=0.05),
            UpliftPrediction(customer_id="c2", cate=0.15),
            UpliftPrediction(customer_id="c3", cate=-0.02),
            UpliftPrediction(customer_id="c4", cate=0.10),
        ]

        ranked = calculator.rank_customers_by_uplift(predictions)

        assert ranked[0].customer_id == "c2"
        assert ranked[1].customer_id == "c4"
        assert ranked[2].customer_id == "c1"
        assert ranked[3].customer_id == "c3"

    def test_rank_customers_top_k(self):
        """Test top-K customer ranking."""
        calculator = UpliftCalculator()

        predictions = [
            UpliftPrediction(customer_id="c1", cate=0.05),
            UpliftPrediction(customer_id="c2", cate=0.15),
            UpliftPrediction(customer_id="c3", cate=0.10),
        ]

        top_2 = calculator.rank_customers_by_uplift(predictions, top_k=2)

        assert len(top_2) == 2
        assert top_2[0].customer_id == "c2"

    def test_segment_by_cate(self):
        """Test customer segmentation by CATE."""
        calculator = UpliftCalculator()

        predictions = [
            UpliftPrediction(customer_id="c1", cate=0.15),  # persuadable
            UpliftPrediction(customer_id="c2", cate=0.05),  # sure thing
            UpliftPrediction(customer_id="c3", cate=-0.05),  # sleeping dog
        ]

        segments = calculator.segment_by_cate(predictions, thresholds=(0.0, 0.1))

        assert len(segments["persuadables"]) == 1
        assert segments["persuadables"][0].customer_id == "c1"
        assert len(segments["sleeping_dogs"]) == 1
        assert segments["sleeping_dogs"][0].customer_id == "c3"

