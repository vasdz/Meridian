"""Uplift models package.

Provides meta-learners for heterogeneous treatment effect estimation:
- TLearner: Two-model approach
- SLearner: Single-model with treatment as feature
- XLearner: Cross-model with propensity weighting
- CausalForest: Tree-based causal inference

Metrics:
- AUUC (Area Under Uplift Curve)
- Qini coefficient
- Lift metrics
"""

from meridian.infrastructure.ml.uplift.meta_learner import MetaLearner
from meridian.infrastructure.ml.uplift.t_learner import TLearner
from meridian.infrastructure.ml.uplift.s_learner import SLearner
from meridian.infrastructure.ml.uplift.x_learner import XLearner
from meridian.infrastructure.ml.uplift.causal_forest import CausalForest
from meridian.infrastructure.ml.uplift.metrics import (
    UpliftMetrics,
    calculate_auuc,
    calculate_qini_coefficient,
    calculate_qini_curve,
    calculate_uplift_curve,
    calculate_lift_at_k,
    evaluate_uplift_model,
    plot_uplift_curves,
)

__all__ = [
    # Models
    "MetaLearner",
    "TLearner",
    "SLearner",
    "XLearner",
    "CausalForest",
    # Metrics
    "UpliftMetrics",
    "calculate_auuc",
    "calculate_qini_coefficient",
    "calculate_qini_curve",
    "calculate_uplift_curve",
    "calculate_lift_at_k",
    "evaluate_uplift_model",
    "plot_uplift_curves",
]


