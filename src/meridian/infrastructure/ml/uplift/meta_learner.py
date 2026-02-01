"""Meta-learner base class for uplift models."""

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from meridian.core.logging import get_logger
from meridian.infrastructure.ml.base import BaseMLModel

logger = get_logger(__name__)


class MetaLearner(BaseMLModel):
    """Base class for meta-learner uplift models."""

    def __init__(
        self,
        model_id: str = "meta_learner",
        base_learner: str = "lightgbm",
        **kwargs,
    ):
        super().__init__(model_id)
        self.base_learner = base_learner
        self.kwargs = kwargs

    def _get_base_learner(self) -> BaseEstimator:
        """Get base learner instance."""
        if self.base_learner == "lightgbm":
            try:
                from lightgbm import LGBMRegressor

                return LGBMRegressor(**self.kwargs)
            except ImportError:
                pass

        if self.base_learner == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(**self.kwargs)

        if self.base_learner == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**self.kwargs)

        # Default to simple regressor
        from sklearn.linear_model import Ridge

        return Ridge(**self.kwargs)

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fit the uplift model."""
        pass

    @abstractmethod
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE for samples."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict_cate."""
        return self.predict_cate(X)
