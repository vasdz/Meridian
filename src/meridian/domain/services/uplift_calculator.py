"""Uplift calculator domain service."""

import numpy as np

from meridian.core.logging import get_logger
from meridian.domain.models.uplift import ConfidenceInterval, UpliftPrediction

logger = get_logger(__name__)


class UpliftCalculator:
    """Domain service for uplift calculations."""

    def calculate_cate(
        self,
        y_treated: np.ndarray,
        y_control: np.ndarray,
    ) -> float:
        """
        Calculate Conditional Average Treatment Effect.

        Simple difference-in-means estimator.
        """
        return float(np.mean(y_treated) - np.mean(y_control))

    def calculate_confidence_interval(
        self,
        cate: float,
        y_treated: np.ndarray,
        y_control: np.ndarray,
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        """Calculate confidence interval for CATE."""
        from scipy import stats

        n_t = len(y_treated)
        n_c = len(y_control)

        var_t = np.var(y_treated, ddof=1)
        var_c = np.var(y_control, ddof=1)

        se = np.sqrt(var_t / n_t + var_c / n_c)

        z = stats.norm.ppf((1 + confidence_level) / 2)

        return ConfidenceInterval(
            lower=cate - z * se,
            upper=cate + z * se,
            level=confidence_level,
        )

    def rank_customers_by_uplift(
        self,
        predictions: list[UpliftPrediction],
        top_k: int | None = None,
    ) -> list[UpliftPrediction]:
        """Rank customers by uplift score (descending)."""
        sorted_predictions = sorted(
            predictions,
            key=lambda p: p.cate,
            reverse=True,
        )

        if top_k is not None:
            return sorted_predictions[:top_k]

        return sorted_predictions

    def calculate_qini(
        self,
        predictions: list[UpliftPrediction],
        actual_outcomes: list[float],
        treatments: list[int],
    ) -> float:
        """Calculate Qini coefficient."""
        # Sort by predicted CATE
        sorted_indices = np.argsort([p.cate for p in predictions])[::-1]

        outcomes = np.array(actual_outcomes)[sorted_indices]
        treatments_arr = np.array(treatments)[sorted_indices]

        n = len(outcomes)
        n_t = np.sum(treatments_arr)
        n_c = n - n_t

        cum_t = np.cumsum(treatments_arr * outcomes)
        cum_c = np.cumsum((1 - treatments_arr) * outcomes)

        qini = cum_t / n_t - cum_c / n_c

        return float(np.sum(qini) / n)

    def segment_by_cate(
        self,
        predictions: list[UpliftPrediction],
        thresholds: tuple[float, float] = (0.0, 0.1),
    ) -> dict[str, list[UpliftPrediction]]:
        """Segment customers based on CATE thresholds."""
        low_threshold, high_threshold = thresholds

        segments: dict[str, list[UpliftPrediction]] = {
            "sure_things": [],  # High baseline, low uplift
            "persuadables": [],  # High uplift
            "sleeping_dogs": [],  # Negative uplift
            "lost_causes": [],  # Low baseline, low uplift
        }

        for pred in predictions:
            if pred.cate > high_threshold:
                segments["persuadables"].append(pred)
            elif pred.cate < low_threshold:
                segments["sleeping_dogs"].append(pred)
            else:
                segments["sure_things"].append(pred)

        return segments
