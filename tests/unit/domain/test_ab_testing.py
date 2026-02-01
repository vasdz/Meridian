"""Tests for A/B testing framework."""

import numpy as np
import pytest

from meridian.domain.services.ab_testing import (
    PowerAnalyzer,
    ABTestAnalyzer,
    design_experiment,
    HypothesisType,
    CorrectionMethod,
)


class TestPowerAnalyzer:
    """Tests for power analysis calculations."""

    def test_sample_size_binary_basic(self):
        """Test basic sample size calculation for binary outcome."""
        sample_size = PowerAnalyzer.calculate_sample_size_binary(
            baseline_rate=0.10,
            mde_relative=0.10,  # 10% relative lift
            alpha=0.05,
            power=0.80,
        )

        # Should be around 1400 per group for 10% baseline, 10% lift
        assert 1000 < sample_size < 2000

    def test_sample_size_increases_with_smaller_mde(self):
        """Smaller MDE requires larger sample size."""
        size_10pct = PowerAnalyzer.calculate_sample_size_binary(0.10, 0.10)
        size_5pct = PowerAnalyzer.calculate_sample_size_binary(0.10, 0.05)

        assert size_5pct > size_10pct

    def test_sample_size_increases_with_higher_power(self):
        """Higher power requires larger sample size."""
        size_80 = PowerAnalyzer.calculate_sample_size_binary(0.10, 0.10, power=0.80)
        size_90 = PowerAnalyzer.calculate_sample_size_binary(0.10, 0.10, power=0.90)

        assert size_90 > size_80

    def test_mde_calculation(self):
        """Test MDE calculation for given sample size."""
        mde = PowerAnalyzer.calculate_mde_binary(
            baseline_rate=0.10,
            sample_size=10000,
            alpha=0.05,
            power=0.80,
        )

        # With 10k samples, MDE should be around 10-15%
        assert 0.05 < mde < 0.20

    def test_mde_decreases_with_larger_sample(self):
        """Larger sample allows detecting smaller effects."""
        mde_small = PowerAnalyzer.calculate_mde_binary(0.10, sample_size=1000)
        mde_large = PowerAnalyzer.calculate_mde_binary(0.10, sample_size=10000)

        assert mde_large < mde_small

    def test_continuous_sample_size(self):
        """Test sample size for continuous outcome."""
        sample_size = PowerAnalyzer.calculate_sample_size_continuous(
            baseline_mean=100,
            baseline_std=20,
            mde_absolute=5,
            alpha=0.05,
            power=0.80,
        )

        assert sample_size > 100  # Should need significant sample

    def test_power_calculation(self):
        """Test achieved power calculation."""
        power = PowerAnalyzer.calculate_power(
            baseline_rate=0.10,
            treatment_rate=0.11,  # 10% lift
            sample_size=10000,
            alpha=0.05,
        )

        assert 0 < power < 1


class TestABTestAnalyzer:
    """Tests for A/B test analysis."""

    def test_analyze_binary_significant(self):
        """Test analysis of clearly significant result."""
        analyzer = ABTestAnalyzer(alpha=0.05)

        result = analyzer.analyze_binary(
            control_successes=500,
            control_total=10000,
            treatment_successes=600,
            treatment_total=10000,
            metric_name="conversion",
        )

        assert result.control_mean == 0.05
        assert result.treatment_mean == 0.06
        assert result.relative_effect == pytest.approx(0.20, abs=0.01)
        assert result.p_value < 0.05
        assert result.is_significant

    def test_analyze_binary_not_significant(self):
        """Test analysis of non-significant result."""
        analyzer = ABTestAnalyzer(alpha=0.05)

        result = analyzer.analyze_binary(
            control_successes=500,
            control_total=10000,
            treatment_successes=510,
            treatment_total=10000,
        )

        assert result.p_value > 0.05
        assert not result.is_significant

    def test_analyze_continuous(self):
        """Test analysis of continuous outcome."""
        np.random.seed(42)

        analyzer = ABTestAnalyzer(alpha=0.05)

        control = np.random.normal(100, 20, 1000)
        treatment = np.random.normal(105, 20, 1000)  # 5% lift

        result = analyzer.analyze_continuous(
            control_values=control,
            treatment_values=treatment,
            metric_name="revenue",
        )

        assert result.absolute_effect > 0
        assert result.is_significant  # Should detect 5% lift with 1k samples

    def test_confidence_interval(self):
        """Test that confidence interval is computed."""
        analyzer = ABTestAnalyzer(alpha=0.05)

        result = analyzer.analyze_binary(
            control_successes=500,
            control_total=10000,
            treatment_successes=550,
            treatment_total=10000,
        )

        lower, upper = result.confidence_interval
        assert lower < result.absolute_effect < upper

    def test_multiple_testing_correction_bonferroni(self):
        """Test Bonferroni correction."""
        analyzer = ABTestAnalyzer(
            alpha=0.05,
            correction_method=CorrectionMethod.BONFERRONI,
        )

        # Create multiple test results
        results = [
            analyzer.analyze_binary(500, 10000, 540 + i*10, 10000)
            for i in range(5)
        ]

        corrected = analyzer.apply_correction(results)

        # Corrected p-values should be higher
        for r in corrected:
            assert r.p_value_corrected >= r.p_value
            assert r.correction_method == "bonferroni"


class TestExperimentDesign:
    """Tests for experiment design function."""

    def test_design_binary_experiment(self):
        """Test designing binary outcome experiment."""
        design = design_experiment(
            name="conversion_test",
            baseline_rate=0.05,
            mde_relative=0.10,
            daily_traffic=5000,
        )

        assert design.name == "conversion_test"
        assert design.is_binary
        assert design.sample_size_per_group > 0
        assert design.total_sample_size == design.sample_size_per_group * 2
        assert design.expected_days > 0

    def test_design_continuous_experiment(self):
        """Test designing continuous outcome experiment."""
        design = design_experiment(
            name="revenue_test",
            baseline_mean=100,
            baseline_std=20,
            mde_absolute=5,
            daily_traffic=1000,
        )

        assert design.name == "revenue_test"
        assert not design.is_binary
        assert design.sample_size_per_group > 0

    def test_expected_duration_calculation(self):
        """Test that expected duration is calculated correctly."""
        design = design_experiment(
            name="test",
            baseline_rate=0.05,
            mde_relative=0.10,
            daily_traffic=10000,
            allocation_ratio=0.5,
        )

        # More traffic = shorter duration
        design_slow = design_experiment(
            name="test",
            baseline_rate=0.05,
            mde_relative=0.10,
            daily_traffic=1000,
            allocation_ratio=0.5,
        )

        assert design.expected_days < design_slow.expected_days

