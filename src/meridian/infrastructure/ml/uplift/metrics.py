"""Uplift modeling metrics: AUUC, Qini, Uplift curves."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from meridian.core.logging import get_logger


logger = get_logger(__name__)


@dataclass
class UpliftMetrics:
    """Container for uplift model evaluation metrics."""

    auuc: float  # Area Under Uplift Curve
    qini_coefficient: float  # Qini coefficient (normalized)
    auuc_random: float  # Expected AUUC for random model
    auuc_perfect: float  # Expected AUUC for perfect model
    lift_at_10: float  # Uplift lift at top 10%
    lift_at_20: float  # Uplift lift at top 20%
    lift_at_50: float  # Uplift lift at top 50%
    ate: float  # Average Treatment Effect (overall)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "auuc": self.auuc,
            "qini_coefficient": self.qini_coefficient,
            "auuc_random": self.auuc_random,
            "auuc_perfect": self.auuc_perfect,
            "auuc_normalized": self.normalized_auuc,
            "lift_at_10": self.lift_at_10,
            "lift_at_20": self.lift_at_20,
            "lift_at_50": self.lift_at_50,
            "ate": self.ate,
        }

    @property
    def normalized_auuc(self) -> float:
        """AUUC normalized to [0, 1] scale."""
        if self.auuc_perfect == self.auuc_random:
            return 0.0
        return (self.auuc - self.auuc_random) / (self.auuc_perfect - self.auuc_random)


def calculate_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate uplift curve.

    The uplift curve shows cumulative uplift as we target more customers,
    ordered by predicted uplift score (descending).

    Args:
        y_true: True outcome values
        treatment: Binary treatment indicator (0 or 1)
        uplift_scores: Predicted uplift scores (CATE)
        n_bins: Number of points in the curve

    Returns:
        Tuple of (percentiles, cumulative_uplift)
    """
    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    uplift_scores = np.asarray(uplift_scores)

    # Sort by uplift scores descending
    sorted_idx = np.argsort(-uplift_scores)
    y_sorted = y_true[sorted_idx]
    t_sorted = treatment[sorted_idx]

    n = len(y_true)
    n_t_total = np.sum(treatment)
    n_c_total = n - n_t_total

    # Calculate cumulative response rates
    percentiles = np.linspace(0, 1, n_bins + 1)[1:]  # Skip 0
    cumulative_uplift = []

    for p in percentiles:
        k = int(np.ceil(p * n))

        y_k = y_sorted[:k]
        t_k = t_sorted[:k]

        n_t_k = np.sum(t_k)
        n_c_k = k - n_t_k

        if n_t_k > 0 and n_c_k > 0:
            response_t = np.sum(y_k[t_k == 1]) / n_t_k
            response_c = np.sum(y_k[t_k == 0]) / n_c_k
            uplift = response_t - response_c
        else:
            uplift = 0.0

        cumulative_uplift.append(uplift * p)  # Cumulative effect

    return percentiles, np.array(cumulative_uplift)


def calculate_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Qini curve.

    The Qini curve shows the cumulative incremental gains as we target
    more customers, ordered by predicted uplift score.

    Qini(k) = (# treated responders in top k) - (# control responders in top k) * n_t/n_c

    Args:
        y_true: True outcome values (binary: 0/1)
        treatment: Binary treatment indicator (0 or 1)
        uplift_scores: Predicted uplift scores (CATE)
        n_bins: Number of points in the curve

    Returns:
        Tuple of (percentiles, qini_values)
    """
    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    uplift_scores = np.asarray(uplift_scores)

    # Sort by uplift scores descending
    sorted_idx = np.argsort(-uplift_scores)
    y_sorted = y_true[sorted_idx]
    t_sorted = treatment[sorted_idx]

    n = len(y_true)
    n_t_total = np.sum(treatment)
    n_c_total = n - n_t_total

    if n_c_total == 0:
        raise ValueError("No control samples found")

    ratio = n_t_total / n_c_total

    # Calculate cumulative Qini
    percentiles = np.linspace(0, 1, n_bins + 1)[1:]
    qini_values = []

    cum_t_responders = 0
    cum_c_responders = 0
    last_k = 0

    for p in percentiles:
        k = min(int(np.ceil(p * n)), n)  # Ensure k doesn't exceed n

        # Update cumulative counts
        for i in range(last_k, k):
            if t_sorted[i] == 1:
                cum_t_responders += y_sorted[i]
            else:
                cum_c_responders += y_sorted[i]

        qini = cum_t_responders - cum_c_responders * ratio
        qini_values.append(qini)

        last_k = k

    return percentiles, np.array(qini_values)


def calculate_auuc(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 100,
) -> float:
    """
    Calculate Area Under Uplift Curve (AUUC).

    Args:
        y_true: True outcome values
        treatment: Binary treatment indicator (0 or 1)
        uplift_scores: Predicted uplift scores (CATE)
        n_bins: Number of bins for integration

    Returns:
        AUUC value
    """
    percentiles, cumulative_uplift = calculate_uplift_curve(
        y_true, treatment, uplift_scores, n_bins
    )

    # Trapezoidal integration
    auuc = np.trapz(cumulative_uplift, percentiles)

    return float(auuc)


def calculate_qini_coefficient(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 100,
) -> float:
    """
    Calculate Qini coefficient.

    The Qini coefficient is the area between the Qini curve and the
    random baseline, normalized by the maximum possible area.

    Args:
        y_true: True outcome values
        treatment: Binary treatment indicator (0 or 1)
        uplift_scores: Predicted uplift scores (CATE)
        n_bins: Number of bins for integration

    Returns:
        Qini coefficient
    """
    percentiles, qini_values = calculate_qini_curve(
        y_true, treatment, uplift_scores, n_bins
    )

    # Area under model curve
    auc_model = np.trapz(qini_values, percentiles)

    # Random baseline: straight line from (0,0) to (1, final_qini)
    final_qini = qini_values[-1]
    auc_random = final_qini / 2  # Triangle area

    # Qini coefficient
    qini_coef = auc_model - auc_random

    return float(qini_coef)


def calculate_lift_at_k(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    k: float = 0.1,
) -> float:
    """
    Calculate uplift lift at top k%.

    Lift = (uplift in top k%) / (overall uplift)

    Args:
        y_true: True outcome values
        treatment: Binary treatment indicator
        uplift_scores: Predicted uplift scores
        k: Fraction of population to consider (default: 0.1 = 10%)

    Returns:
        Lift value (>1 means model is better than random)
    """
    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    uplift_scores = np.asarray(uplift_scores)

    n = len(y_true)
    top_k = int(np.ceil(k * n))

    # Sort by uplift scores descending
    sorted_idx = np.argsort(-uplift_scores)

    # Top k
    top_idx = sorted_idx[:top_k]
    y_top = y_true[top_idx]
    t_top = treatment[top_idx]

    # Calculate uplift in top k
    n_t_top = np.sum(t_top)
    n_c_top = len(t_top) - n_t_top

    if n_t_top > 0 and n_c_top > 0:
        uplift_top = np.mean(y_top[t_top == 1]) - np.mean(y_top[t_top == 0])
    else:
        return 0.0

    # Calculate overall uplift
    n_t_all = np.sum(treatment)
    n_c_all = n - n_t_all

    if n_t_all > 0 and n_c_all > 0:
        uplift_all = np.mean(y_true[treatment == 1]) - np.mean(y_true[treatment == 0])
    else:
        return 0.0

    if uplift_all == 0:
        return 0.0

    return float(uplift_top / uplift_all)


def evaluate_uplift_model(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 100,
) -> UpliftMetrics:
    """
    Comprehensive evaluation of uplift model.

    Args:
        y_true: True outcome values
        treatment: Binary treatment indicator
        uplift_scores: Predicted uplift scores (CATE)
        n_bins: Number of bins for curve calculations

    Returns:
        UpliftMetrics object with all metrics
    """
    logger.info("Evaluating uplift model", n_samples=len(y_true))

    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    uplift_scores = np.asarray(uplift_scores)

    # Calculate main metrics
    auuc = calculate_auuc(y_true, treatment, uplift_scores, n_bins)
    qini_coef = calculate_qini_coefficient(y_true, treatment, uplift_scores, n_bins)

    # Lift at different percentiles
    lift_10 = calculate_lift_at_k(y_true, treatment, uplift_scores, k=0.1)
    lift_20 = calculate_lift_at_k(y_true, treatment, uplift_scores, k=0.2)
    lift_50 = calculate_lift_at_k(y_true, treatment, uplift_scores, k=0.5)

    # Calculate ATE
    n_t = np.sum(treatment)
    n_c = len(treatment) - n_t
    ate = np.mean(y_true[treatment == 1]) - np.mean(y_true[treatment == 0])

    # Random model AUUC (baseline)
    random_scores = np.random.randn(len(y_true))
    auuc_random = calculate_auuc(y_true, treatment, random_scores, n_bins)

    # Perfect model AUUC (upper bound approximation)
    # For binary outcomes, perfect targeting = target only positive responders
    if len(np.unique(y_true)) == 2:
        # Binary outcome: perfect = target treated responders, avoid control responders
        perfect_scores = y_true * treatment - y_true * (1 - treatment)
    else:
        # Continuous: use true uplift as perfect scores
        # This is an approximation since we don't know true individual effects
        perfect_scores = uplift_scores  # Will be updated if we have oracle

    auuc_perfect = calculate_auuc(y_true, treatment, y_true * (2 * treatment - 1), n_bins)

    metrics = UpliftMetrics(
        auuc=auuc,
        qini_coefficient=qini_coef,
        auuc_random=auuc_random,
        auuc_perfect=auuc_perfect,
        lift_at_10=lift_10,
        lift_at_20=lift_20,
        lift_at_50=lift_50,
        ate=float(ate),
    )

    logger.info(
        "Uplift evaluation complete",
        auuc=auuc,
        qini=qini_coef,
        ate=ate,
    )

    return metrics


def plot_uplift_curves(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 100,
) -> dict:
    """
    Generate data for plotting uplift and Qini curves.

    Returns dictionary with curve data that can be used with matplotlib
    or any plotting library.

    Args:
        y_true: True outcome values
        treatment: Binary treatment indicator
        uplift_scores: Predicted uplift scores
        model_name: Name for the model (for legends)
        n_bins: Number of bins

    Returns:
        Dictionary with curve data
    """
    percentiles, uplift_curve = calculate_uplift_curve(
        y_true, treatment, uplift_scores, n_bins
    )

    _, qini_curve = calculate_qini_curve(
        y_true, treatment, uplift_scores, n_bins
    )

    # Random baseline
    random_scores = np.random.randn(len(y_true))
    _, random_uplift = calculate_uplift_curve(y_true, treatment, random_scores, n_bins)
    _, random_qini = calculate_qini_curve(y_true, treatment, random_scores, n_bins)

    return {
        "percentiles": percentiles.tolist(),
        "model_name": model_name,
        "uplift_curve": {
            "model": uplift_curve.tolist(),
            "random": random_uplift.tolist(),
        },
        "qini_curve": {
            "model": qini_curve.tolist(),
            "random": random_qini.tolist(),
        },
    }

