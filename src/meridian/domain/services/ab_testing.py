"""A/B Testing Framework with power analysis, MDE, and multiple testing corrections.

This module provides comprehensive tools for designing and analyzing A/B tests,
including:
- Sample size calculation
- Power analysis
- Minimum Detectable Effect (MDE)
- Multiple testing corrections (Bonferroni, Holm-Bonferroni, FDR)
- Stratification support
- Sequential testing (optional early stopping)
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum

from meridian.core.logging import get_logger


logger = get_logger(__name__)


class CorrectionMethod(Enum):
    """Multiple testing correction methods."""
    NONE = "none"
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    FDR = "fdr"  # Benjamini-Hochberg


class HypothesisType(Enum):
    """Type of statistical test hypothesis."""
    TWO_SIDED = "two_sided"
    GREATER = "greater"
    LESS = "less"


@dataclass
class ExperimentDesign:
    """Experiment design parameters."""

    name: str
    description: str = ""

    # Sample size
    sample_size_per_group: int = 0
    total_sample_size: int = 0

    # Statistical parameters
    alpha: float = 0.05  # Significance level
    power: float = 0.8  # Statistical power
    mde: float = 0.0  # Minimum Detectable Effect

    # Baseline metrics
    baseline_rate: float = 0.0  # For binary outcomes
    baseline_mean: float = 0.0  # For continuous outcomes
    baseline_std: float = 0.0  # For continuous outcomes

    # Test configuration
    test_type: HypothesisType = HypothesisType.TWO_SIDED
    is_binary: bool = True

    # Stratification
    strata: list[str] = field(default_factory=list)

    # Duration
    expected_days: int = 14
    daily_traffic: int = 1000

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "sample_size_per_group": self.sample_size_per_group,
            "total_sample_size": self.total_sample_size,
            "alpha": self.alpha,
            "power": self.power,
            "mde": self.mde,
            "baseline_rate": self.baseline_rate,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "test_type": self.test_type.value,
            "is_binary": self.is_binary,
            "strata": self.strata,
            "expected_days": self.expected_days,
            "daily_traffic": self.daily_traffic,
        }


@dataclass
class TestResult:
    """Result of a statistical test."""

    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float  # Percentage change

    p_value: float
    p_value_corrected: Optional[float] = None
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    is_significant: bool = False
    power_achieved: float = 0.0

    n_control: int = 0
    n_treatment: int = 0

    test_statistic: float = 0.0
    correction_method: str = "none"

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "absolute_effect": self.absolute_effect,
            "relative_effect": self.relative_effect,
            "p_value": self.p_value,
            "p_value_corrected": self.p_value_corrected,
            "confidence_interval": list(self.confidence_interval),
            "is_significant": self.is_significant,
            "power_achieved": self.power_achieved,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "test_statistic": self.test_statistic,
            "correction_method": self.correction_method,
        }


class PowerAnalyzer:
    """Power analysis for A/B tests."""

    @staticmethod
    def calculate_sample_size_binary(
        baseline_rate: float,
        mde_relative: float,
        alpha: float = 0.05,
        power: float = 0.8,
        test_type: HypothesisType = HypothesisType.TWO_SIDED,
    ) -> int:
        """
        Calculate required sample size per group for binary outcome.

        Args:
            baseline_rate: Baseline conversion rate (e.g., 0.10 for 10%)
            mde_relative: Minimum detectable effect as relative change (e.g., 0.10 for 10% lift)
            alpha: Significance level
            power: Desired statistical power
            test_type: Type of test

        Returns:
            Required sample size per group
        """
        if baseline_rate <= 0 or baseline_rate >= 1:
            raise ValueError("baseline_rate must be between 0 and 1")

        if mde_relative <= 0:
            raise ValueError("mde_relative must be positive")

        # Calculate absolute effect
        treatment_rate = baseline_rate * (1 + mde_relative)

        if treatment_rate >= 1:
            treatment_rate = 0.99

        # Pooled rate
        p_pooled = (baseline_rate + treatment_rate) / 2

        # Effect size (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(baseline_rate)))

        # Z-scores
        if test_type == HypothesisType.TWO_SIDED:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        # Sample size formula
        variance = p_pooled * (1 - p_pooled)
        n = 2 * variance * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    @staticmethod
    def calculate_sample_size_continuous(
        baseline_mean: float,
        baseline_std: float,
        mde_absolute: float,
        alpha: float = 0.05,
        power: float = 0.8,
        test_type: HypothesisType = HypothesisType.TWO_SIDED,
    ) -> int:
        """
        Calculate required sample size per group for continuous outcome.

        Args:
            baseline_mean: Baseline mean value
            baseline_std: Baseline standard deviation
            mde_absolute: Minimum detectable effect (absolute)
            alpha: Significance level
            power: Desired statistical power
            test_type: Type of test

        Returns:
            Required sample size per group
        """
        if baseline_std <= 0:
            raise ValueError("baseline_std must be positive")

        if mde_absolute == 0:
            raise ValueError("mde_absolute cannot be zero")

        # Effect size (Cohen's d)
        effect_size = abs(mde_absolute) / baseline_std

        # Z-scores
        if test_type == HypothesisType.TWO_SIDED:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        # Sample size formula
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    @staticmethod
    def calculate_mde_binary(
        baseline_rate: float,
        sample_size: int,
        alpha: float = 0.05,
        power: float = 0.8,
        test_type: HypothesisType = HypothesisType.TWO_SIDED,
    ) -> float:
        """
        Calculate Minimum Detectable Effect for given sample size (binary outcome).

        Args:
            baseline_rate: Baseline conversion rate
            sample_size: Sample size per group
            alpha: Significance level
            power: Desired statistical power
            test_type: Type of test

        Returns:
            MDE as relative change (e.g., 0.10 means 10% lift)
        """
        if test_type == HypothesisType.TWO_SIDED:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        # Standard error
        se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / sample_size)

        # Absolute effect
        absolute_effect = (z_alpha + z_beta) * se

        # Relative effect
        relative_effect = absolute_effect / baseline_rate

        return relative_effect

    @staticmethod
    def calculate_mde_continuous(
        baseline_std: float,
        sample_size: int,
        alpha: float = 0.05,
        power: float = 0.8,
        test_type: HypothesisType = HypothesisType.TWO_SIDED,
    ) -> float:
        """
        Calculate Minimum Detectable Effect for given sample size (continuous outcome).

        Args:
            baseline_std: Baseline standard deviation
            sample_size: Sample size per group
            alpha: Significance level
            power: Desired statistical power
            test_type: Type of test

        Returns:
            MDE as absolute value
        """
        if test_type == HypothesisType.TWO_SIDED:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        # Standard error
        se = baseline_std * np.sqrt(2 / sample_size)

        # MDE
        mde = (z_alpha + z_beta) * se

        return mde

    @staticmethod
    def calculate_power(
        baseline_rate: float,
        treatment_rate: float,
        sample_size: int,
        alpha: float = 0.05,
        test_type: HypothesisType = HypothesisType.TWO_SIDED,
    ) -> float:
        """
        Calculate achieved power for a binary outcome experiment.

        Args:
            baseline_rate: Baseline conversion rate
            treatment_rate: Expected treatment conversion rate
            sample_size: Sample size per group
            alpha: Significance level
            test_type: Type of test

        Returns:
            Statistical power
        """
        # Effect size
        effect_size = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(baseline_rate)))

        # Standard error
        p_pooled = (baseline_rate + treatment_rate) / 2
        se = np.sqrt(2 * p_pooled * (1 - p_pooled) / sample_size)

        # Z-score for alpha
        if test_type == HypothesisType.TWO_SIDED:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        # Z-score for effect
        z_effect = abs(effect_size) / se

        # Power
        power = stats.norm.cdf(z_effect - z_alpha)

        return power


class ABTestAnalyzer:
    """Analyze A/B test results with statistical rigor."""

    def __init__(
        self,
        alpha: float = 0.05,
        correction_method: CorrectionMethod = CorrectionMethod.BONFERRONI,
    ):
        self.alpha = alpha
        self.correction_method = correction_method
        self.power_analyzer = PowerAnalyzer()

    def analyze_binary(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        metric_name: str = "conversion_rate",
        test_type: HypothesisType = HypothesisType.TWO_SIDED,
    ) -> TestResult:
        """
        Analyze binary outcome A/B test (e.g., conversion rate).

        Uses chi-squared test or z-test for proportions.

        Args:
            control_successes: Number of conversions in control
            control_total: Total samples in control
            treatment_successes: Number of conversions in treatment
            treatment_total: Total samples in treatment
            metric_name: Name of the metric
            test_type: Type of test

        Returns:
            TestResult object
        """
        logger.info(
            "Analyzing binary A/B test",
            metric=metric_name,
            control_n=control_total,
            treatment_n=treatment_total,
        )

        # Calculate rates
        control_rate = control_successes / control_total
        treatment_rate = treatment_successes / treatment_total

        # Effect
        absolute_effect = treatment_rate - control_rate
        relative_effect = absolute_effect / control_rate if control_rate > 0 else 0

        # Z-test for proportions
        pooled_rate = (control_successes + treatment_successes) / (control_total + treatment_total)
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_total + 1/treatment_total))

        if se > 0:
            z_stat = absolute_effect / se
        else:
            z_stat = 0

        # P-value
        if test_type == HypothesisType.TWO_SIDED:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif test_type == HypothesisType.GREATER:
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)

        # Confidence interval
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci = (absolute_effect - z_crit * se, absolute_effect + z_crit * se)

        # Achieved power
        power_achieved = self.power_analyzer.calculate_power(
            control_rate,
            treatment_rate,
            min(control_total, treatment_total),
            self.alpha,
            test_type,
        )

        return TestResult(
            metric_name=metric_name,
            control_mean=control_rate,
            treatment_mean=treatment_rate,
            absolute_effect=absolute_effect,
            relative_effect=relative_effect,
            p_value=p_value,
            confidence_interval=ci,
            is_significant=p_value < self.alpha,
            power_achieved=power_achieved,
            n_control=control_total,
            n_treatment=treatment_total,
            test_statistic=z_stat,
        )

    def analyze_continuous(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray,
        metric_name: str = "revenue",
        test_type: HypothesisType = HypothesisType.TWO_SIDED,
    ) -> TestResult:
        """
        Analyze continuous outcome A/B test (e.g., revenue per user).

        Uses Welch's t-test.

        Args:
            control_values: Array of values from control group
            treatment_values: Array of values from treatment group
            metric_name: Name of the metric
            test_type: Type of test

        Returns:
            TestResult object
        """
        logger.info(
            "Analyzing continuous A/B test",
            metric=metric_name,
            control_n=len(control_values),
            treatment_n=len(treatment_values),
        )

        control_values = np.asarray(control_values)
        treatment_values = np.asarray(treatment_values)

        # Calculate means
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)

        # Effect
        absolute_effect = treatment_mean - control_mean
        relative_effect = absolute_effect / control_mean if control_mean != 0 else 0

        # Welch's t-test
        if test_type == HypothesisType.TWO_SIDED:
            alternative = "two-sided"
        elif test_type == HypothesisType.GREATER:
            alternative = "greater"
        else:
            alternative = "less"

        t_stat, p_value = stats.ttest_ind(
            treatment_values,
            control_values,
            equal_var=False,
            alternative=alternative,
        )

        # Confidence interval for difference
        se = np.sqrt(
            np.var(control_values, ddof=1) / len(control_values) +
            np.var(treatment_values, ddof=1) / len(treatment_values)
        )

        # Degrees of freedom (Welch-Satterthwaite)
        n1, n2 = len(control_values), len(treatment_values)
        v1 = np.var(control_values, ddof=1)
        v2 = np.var(treatment_values, ddof=1)

        df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        ci = (absolute_effect - t_crit * se, absolute_effect + t_crit * se)

        return TestResult(
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_effect=absolute_effect,
            relative_effect=relative_effect,
            p_value=p_value,
            confidence_interval=ci,
            is_significant=p_value < self.alpha,
            n_control=len(control_values),
            n_treatment=len(treatment_values),
            test_statistic=t_stat,
        )

    def apply_correction(
        self,
        results: list[TestResult],
    ) -> list[TestResult]:
        """
        Apply multiple testing correction to a list of test results.

        Args:
            results: List of TestResult objects

        Returns:
            Updated list with corrected p-values and significance
        """
        if self.correction_method == CorrectionMethod.NONE:
            return results

        n_tests = len(results)
        p_values = [r.p_value for r in results]

        if self.correction_method == CorrectionMethod.BONFERRONI:
            corrected_alpha = self.alpha / n_tests
            corrected_p = [min(p * n_tests, 1.0) for p in p_values]

        elif self.correction_method == CorrectionMethod.HOLM:
            # Holm-Bonferroni step-down procedure
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros(n_tests)

            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = min(p_values[idx] * (n_tests - i), 1.0)

            # Enforce monotonicity
            max_so_far = 0
            for idx in sorted_indices:
                max_so_far = max(max_so_far, corrected_p[idx])
                corrected_p[idx] = max_so_far

            corrected_alpha = self.alpha
            corrected_p = corrected_p.tolist()

        elif self.correction_method == CorrectionMethod.FDR:
            # Benjamini-Hochberg procedure
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros(n_tests)

            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = p_values[idx] * n_tests / (i + 1)

            # Enforce monotonicity (backward)
            min_so_far = corrected_p[sorted_indices[-1]]
            for idx in reversed(sorted_indices):
                min_so_far = min(min_so_far, corrected_p[idx])
                corrected_p[idx] = min(min_so_far, 1.0)

            corrected_alpha = self.alpha
            corrected_p = corrected_p.tolist()

        else:
            corrected_alpha = self.alpha
            corrected_p = p_values

        # Update results
        for i, result in enumerate(results):
            result.p_value_corrected = corrected_p[i]
            result.is_significant = corrected_p[i] < corrected_alpha
            result.correction_method = self.correction_method.value

        logger.info(
            "Applied multiple testing correction",
            method=self.correction_method.value,
            n_tests=n_tests,
            n_significant=sum(r.is_significant for r in results),
        )

        return results


def design_experiment(
    name: str,
    baseline_rate: Optional[float] = None,
    baseline_mean: Optional[float] = None,
    baseline_std: Optional[float] = None,
    mde_relative: Optional[float] = None,
    mde_absolute: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    daily_traffic: int = 1000,
    allocation_ratio: float = 0.5,
    description: str = "",
) -> ExperimentDesign:
    """
    Design an A/B experiment with power analysis.

    Args:
        name: Experiment name
        baseline_rate: Baseline conversion rate (for binary outcomes)
        baseline_mean: Baseline mean (for continuous outcomes)
        baseline_std: Baseline std (for continuous outcomes)
        mde_relative: Relative MDE (e.g., 0.10 for 10% lift)
        mde_absolute: Absolute MDE (for continuous outcomes)
        alpha: Significance level
        power: Desired statistical power
        daily_traffic: Expected daily traffic
        allocation_ratio: Ratio of traffic allocated to test (default: 50%)
        description: Experiment description

    Returns:
        ExperimentDesign object
    """
    analyzer = PowerAnalyzer()

    # Determine if binary or continuous
    is_binary = baseline_rate is not None

    if is_binary:
        if mde_relative is None:
            raise ValueError("mde_relative required for binary outcomes")

        sample_size = analyzer.calculate_sample_size_binary(
            baseline_rate,
            mde_relative,
            alpha,
            power,
        )
        mde = mde_relative

    else:
        if baseline_mean is None or baseline_std is None or mde_absolute is None:
            raise ValueError("baseline_mean, baseline_std, and mde_absolute required for continuous outcomes")

        sample_size = analyzer.calculate_sample_size_continuous(
            baseline_mean,
            baseline_std,
            mde_absolute,
            alpha,
            power,
        )
        mde = mde_absolute

    total_sample_size = sample_size * 2

    # Calculate expected duration
    daily_allocated = int(daily_traffic * allocation_ratio)
    expected_days = int(np.ceil(total_sample_size / daily_allocated)) if daily_allocated > 0 else 0

    design = ExperimentDesign(
        name=name,
        description=description,
        sample_size_per_group=sample_size,
        total_sample_size=total_sample_size,
        alpha=alpha,
        power=power,
        mde=mde,
        baseline_rate=baseline_rate or 0.0,
        baseline_mean=baseline_mean or 0.0,
        baseline_std=baseline_std or 0.0,
        is_binary=is_binary,
        expected_days=expected_days,
        daily_traffic=daily_traffic,
    )

    logger.info(
        "Experiment designed",
        name=name,
        sample_size_per_group=sample_size,
        expected_days=expected_days,
        mde=mde,
    )

    return design

