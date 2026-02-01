"""Experiment design domain service."""

import math
from typing import Optional

from scipy import stats

from meridian.core.logging import get_logger


logger = get_logger(__name__)


class ExperimentDesignService:
    """Domain service for A/B experiment design."""

    def calculate_sample_size(
        self,
        baseline_rate: float,
        mde: float,
        power: float = 0.8,
        alpha: float = 0.05,
        ratio: float = 1.0,
    ) -> dict:
        """
        Calculate required sample size for experiment.

        Args:
            baseline_rate: Current conversion rate
            mde: Minimum Detectable Effect (relative)
            power: Statistical power (1 - beta)
            alpha: Significance level
            ratio: Ratio of treatment to control size

        Returns:
            Dictionary with sample size details
        """
        # Effect size
        effect = baseline_rate * mde
        p_control = baseline_rate
        p_treatment = baseline_rate + effect

        # Pooled standard deviation
        pooled_p = (p_control + p_treatment) / 2
        pooled_sd = math.sqrt(2 * pooled_p * (1 - pooled_p))

        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_power = stats.norm.ppf(power)

        # Sample size per group
        n_per_group = (
            2 * ((z_alpha + z_power) ** 2) * pooled_p * (1 - pooled_p)
        ) / (effect ** 2)

        n_per_group = int(math.ceil(n_per_group))

        return {
            "sample_size_per_variant": n_per_group,
            "total_sample_size": n_per_group * 2,
            "baseline_rate": baseline_rate,
            "mde": mde,
            "mde_absolute": effect,
            "power": power,
            "alpha": alpha,
        }

    def calculate_power(
        self,
        sample_size: int,
        baseline_rate: float,
        effect_size: float,
        alpha: float = 0.05,
    ) -> float:
        """Calculate statistical power given sample size."""
        p_control = baseline_rate
        p_treatment = baseline_rate + effect_size

        pooled_p = (p_control + p_treatment) / 2
        se = math.sqrt(2 * pooled_p * (1 - pooled_p) / sample_size)

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z = effect_size / se - z_alpha

        power = stats.norm.cdf(z)
        return float(power)

    def calculate_mde(
        self,
        sample_size: int,
        baseline_rate: float,
        power: float = 0.8,
        alpha: float = 0.05,
    ) -> float:
        """Calculate minimum detectable effect given sample size."""
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_power = stats.norm.ppf(power)

        se_factor = math.sqrt(2 * baseline_rate * (1 - baseline_rate) / sample_size)

        mde = (z_alpha + z_power) * se_factor
        return float(mde)

    def analyze_results(
        self,
        control_conversions: int,
        control_size: int,
        treatment_conversions: int,
        treatment_size: int,
    ) -> dict:
        """Analyze experiment results."""
        p_control = control_conversions / control_size
        p_treatment = treatment_conversions / treatment_size

        lift = (p_treatment - p_control) / p_control if p_control > 0 else 0

        # Z-test for proportions
        pooled_p = (control_conversions + treatment_conversions) / (control_size + treatment_size)
        se = math.sqrt(pooled_p * (1 - pooled_p) * (1/control_size + 1/treatment_size))

        z_stat = (p_treatment - p_control) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Confidence interval
        se_diff = math.sqrt(
            p_control * (1 - p_control) / control_size +
            p_treatment * (1 - p_treatment) / treatment_size
        )
        ci_low = (p_treatment - p_control) - 1.96 * se_diff
        ci_high = (p_treatment - p_control) + 1.96 * se_diff

        return {
            "control_rate": p_control,
            "treatment_rate": p_treatment,
            "lift": lift,
            "lift_absolute": p_treatment - p_control,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "confidence_interval": (ci_low, ci_high),
            "z_statistic": z_stat,
        }

