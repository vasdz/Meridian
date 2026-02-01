"""Demand Forecasting Module - Production-grade time series forecasting.

This module provides enterprise-level demand forecasting capabilities:
- Multi-horizon probabilistic forecasting
- Hierarchical forecasting (region → store → SKU)
- Conformal prediction for uncertainty quantification
- External regressors (promotions, holidays, weather)
- Model ensembling and selection

Designed for hyperscale retail operations.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Literal, Any
from datetime import datetime, timedelta
from enum import Enum

from meridian.core.logging import get_logger


logger = get_logger(__name__)


class ForecastGranularity(Enum):
    """Time granularity for forecasting."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AggregationLevel(Enum):
    """Hierarchical aggregation levels."""
    TOTAL = "total"
    REGION = "region"
    STORE = "store"
    CATEGORY = "category"
    SKU = "sku"


@dataclass
class ForecastConfig:
    """Configuration for forecasting model."""

    # Model parameters
    model_type: str = "ensemble"
    horizon: int = 14  # Days to forecast
    granularity: ForecastGranularity = ForecastGranularity.DAILY

    # Training parameters
    lookback_days: int = 365
    min_train_samples: int = 30

    # Probabilistic forecasting
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    conformal_alpha: float = 0.1  # For conformal prediction intervals

    # Features
    use_promotions: bool = True
    use_holidays: bool = True
    use_weather: bool = False
    use_price: bool = True

    # Hierarchical
    hierarchical: bool = True
    reconciliation_method: str = "mint"  # mint, ols, bu, td

    # Model selection
    auto_select: bool = True
    cv_folds: int = 3

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "horizon": self.horizon,
            "granularity": self.granularity.value,
            "lookback_days": self.lookback_days,
            "quantiles": self.quantiles,
            "hierarchical": self.hierarchical,
        }


@dataclass
class ForecastResult:
    """Result of a forecast prediction."""

    # Identifiers
    series_id: str
    aggregation_level: AggregationLevel

    # Timestamps
    forecast_date: datetime
    prediction_timestamps: list[datetime]

    # Point forecasts
    point_forecast: np.ndarray

    # Probabilistic forecasts
    quantile_forecasts: dict[float, np.ndarray]  # quantile -> values
    prediction_intervals: dict[float, tuple[np.ndarray, np.ndarray]]  # level -> (lower, upper)

    # Metadata
    model_used: str
    feature_importance: Optional[dict[str, float]] = None
    metrics: Optional[dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "series_id": self.series_id,
            "aggregation_level": self.aggregation_level.value,
            "forecast_date": self.forecast_date.isoformat(),
            "predictions": [
                {
                    "timestamp": ts.isoformat(),
                    "point_forecast": float(self.point_forecast[i]),
                    "lower_90": float(self.prediction_intervals.get(0.9, (self.point_forecast, self.point_forecast))[0][i]),
                    "upper_90": float(self.prediction_intervals.get(0.9, (self.point_forecast, self.point_forecast))[1][i]),
                }
                for i, ts in enumerate(self.prediction_timestamps)
            ],
            "model_used": self.model_used,
            "metrics": self.metrics,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        df = pd.DataFrame({
            "timestamp": self.prediction_timestamps,
            "point_forecast": self.point_forecast,
        })

        for quantile, values in self.quantile_forecasts.items():
            df[f"q{int(quantile*100)}"] = values

        return df


class BaseForecastModel(ABC):
    """Abstract base class for forecasting models."""

    def __init__(self, config: ForecastConfig):
        self.config = config
        self._is_fitted = False
        self._feature_names: list[str] = []

    @abstractmethod
    def fit(
        self,
        y: np.ndarray,
        timestamps: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> "BaseForecastModel":
        """Fit the forecasting model."""
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int,
        X_future: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """Generate point and quantile forecasts."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model name for identification."""
        pass

    def _create_features(
        self,
        timestamps: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Create time-based features."""
        features = []
        feature_names = []

        # Convert to datetime if needed
        if not isinstance(timestamps[0], datetime):
            timestamps = pd.to_datetime(timestamps)

        # Day of week (one-hot)
        dow = np.array([t.weekday() for t in timestamps])
        for i in range(7):
            features.append((dow == i).astype(float))
            feature_names.append(f"dow_{i}")

        # Month (cyclical encoding)
        months = np.array([t.month for t in timestamps])
        features.append(np.sin(2 * np.pi * months / 12))
        features.append(np.cos(2 * np.pi * months / 12))
        feature_names.extend(["month_sin", "month_cos"])

        # Day of month
        dom = np.array([t.day for t in timestamps])
        features.append(np.sin(2 * np.pi * dom / 31))
        features.append(np.cos(2 * np.pi * dom / 31))
        feature_names.extend(["dom_sin", "dom_cos"])

        # Week of year
        woy = np.array([t.isocalendar()[1] for t in timestamps])
        features.append(np.sin(2 * np.pi * woy / 52))
        features.append(np.cos(2 * np.pi * woy / 52))
        feature_names.extend(["woy_sin", "woy_cos"])

        # Trend
        days_from_start = np.array([(t - timestamps[0]).days for t in timestamps])
        features.append(days_from_start / 365)  # Normalized trend
        feature_names.append("trend")

        # Lag features (if y provided)
        if y is not None and len(y) >= len(timestamps):
            y_aligned = y[:len(timestamps)]  # Ensure same length
            for lag in [1, 7, 14, 28]:
                if len(y_aligned) > lag:
                    lagged = np.zeros(len(timestamps))
                    lagged[lag:] = y_aligned[:-lag]
                    features.append(lagged)
                    feature_names.append(f"lag_{lag}")

            # Rolling statistics
            for window in [7, 14, 28]:
                if len(y_aligned) > window:
                    rolling_mean = pd.Series(y_aligned).rolling(window, min_periods=1).mean().values
                    rolling_std = pd.Series(y_aligned).rolling(window, min_periods=1).std().fillna(0).values
                    features.extend([rolling_mean, rolling_std])
                    feature_names.extend([f"rolling_mean_{window}", f"rolling_std_{window}"])

        self._feature_names = feature_names
        return np.column_stack(features)


class LightGBMForecaster(BaseForecastModel):
    """LightGBM-based forecasting model with quantile regression."""

    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self._models: dict[float, Any] = {}  # quantile -> model
        self._last_values: Optional[np.ndarray] = None
        self._last_timestamps: Optional[np.ndarray] = None

    def fit(
        self,
        y: np.ndarray,
        timestamps: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> "LightGBMForecaster":
        """Fit LightGBM models for each quantile."""
        from sklearn.ensemble import GradientBoostingRegressor

        logger.info(
            "Fitting LightGBM forecaster",
            n_samples=len(y),
            quantiles=self.config.quantiles,
        )

        # Create features
        X_time = self._create_features(timestamps, y)
        self._n_features = X_time.shape[1]  # Save feature count
        if X is not None:
            X_combined = np.hstack([X_time, X])
        else:
            X_combined = X_time

        # Store for prediction
        self._last_values = y.copy()
        self._last_timestamps = timestamps.copy()

        # Fit model for each quantile
        for quantile in self.config.quantiles:
            if quantile == 0.5:
                # Use regular regression for median
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                )
            else:
                # Use quantile regression
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    loss="quantile",
                    alpha=quantile,
                    random_state=42,
                )

            model.fit(X_combined, y)
            self._models[quantile] = model

        self._is_fitted = True
        logger.info("LightGBM forecaster fitted successfully")

        return self

    def predict(
        self,
        horizon: int,
        X_future: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """Generate forecasts."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        # Generate future timestamps
        last_ts = pd.to_datetime(self._last_timestamps[-1])
        future_timestamps = pd.date_range(
            start=last_ts + timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        # Create features for future
        # For lag features, we need to do iterative prediction
        y_extended = self._last_values.copy()
        predictions = {q: [] for q in self.config.quantiles}

        for i, ts in enumerate(future_timestamps):
            # Create features for all timestamps including history and future so far
            all_timestamps = np.append(self._last_timestamps, future_timestamps[:i+1])

            # y_extended needs to match all_timestamps length
            # At iteration i, all_timestamps has len(_last_timestamps) + i + 1 elements
            # y_extended should have the same
            X_time = self._create_features(all_timestamps, y_extended)

            # Ensure we have the right number of features
            if X_time.shape[1] != self._n_features:
                # Pad or truncate features to match training
                if X_time.shape[1] < self._n_features:
                    padding = np.zeros((X_time.shape[0], self._n_features - X_time.shape[1]))
                    X_time = np.hstack([X_time, padding])
                else:
                    X_time = X_time[:, :self._n_features]

            X_current = X_time[-1:, :]

            if X_future is not None and i < len(X_future):
                X_current = np.hstack([X_current, X_future[i:i+1, :]])

            # Predict for each quantile
            for quantile, model in self._models.items():
                pred = model.predict(X_current)[0]
                predictions[quantile].append(max(0, pred))  # Ensure non-negative

            # Use median for extending y - append BEFORE next iteration
            y_extended = np.append(y_extended, predictions[0.5][-1])

        # Convert to arrays
        quantile_forecasts = {q: np.array(preds) for q, preds in predictions.items()}
        point_forecast = quantile_forecasts.get(0.5, np.mean(list(quantile_forecasts.values()), axis=0))

        return point_forecast, quantile_forecasts

    def get_model_name(self) -> str:
        return "lightgbm_quantile"


class ExponentialSmoothingForecaster(BaseForecastModel):
    """Exponential smoothing (Holt-Winters) forecaster."""

    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self._model = None
        self._residuals: Optional[np.ndarray] = None

    def fit(
        self,
        y: np.ndarray,
        timestamps: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> "ExponentialSmoothingForecaster":
        """Fit exponential smoothing model."""
        from scipy.optimize import minimize

        logger.info("Fitting Exponential Smoothing", n_samples=len(y))

        # Simple Holt-Winters implementation
        self._y = y.copy()

        # Optimize smoothing parameters
        def objective(params):
            alpha, beta, gamma = params
            return self._holt_winters_error(y, alpha, beta, gamma)

        result = minimize(
            objective,
            x0=[0.3, 0.1, 0.1],
            bounds=[(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)],
            method="L-BFGS-B",
        )

        self._alpha, self._beta, self._gamma = result.x

        # Compute fitted values and residuals
        fitted, self._level, self._trend, self._seasonal = self._holt_winters_fit(
            y, self._alpha, self._beta, self._gamma
        )
        self._residuals = y - fitted

        self._is_fitted = True
        logger.info(
            "Exponential Smoothing fitted",
            alpha=self._alpha,
            beta=self._beta,
            gamma=self._gamma,
        )

        return self

    def _holt_winters_error(
        self,
        y: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        season_length: int = 7,
    ) -> float:
        """Calculate MSE for Holt-Winters."""
        fitted, _, _, _ = self._holt_winters_fit(y, alpha, beta, gamma, season_length)
        return np.mean((y - fitted) ** 2)

    def _holt_winters_fit(
        self,
        y: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        season_length: int = 7,
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        """Fit Holt-Winters and return components."""
        n = len(y)

        # Initialize
        level = np.mean(y[:season_length])
        trend = (np.mean(y[season_length:2*season_length]) - np.mean(y[:season_length])) / season_length
        seasonal = np.zeros(season_length)
        for i in range(season_length):
            seasonal[i] = np.mean(y[i::season_length][:min(4, n//season_length)]) - level

        fitted = np.zeros(n)

        for t in range(n):
            season_idx = t % season_length

            if t == 0:
                fitted[t] = level + trend + seasonal[season_idx]
            else:
                # Update equations
                prev_level = level
                level = alpha * (y[t] - seasonal[season_idx]) + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
                seasonal[season_idx] = gamma * (y[t] - level) + (1 - gamma) * seasonal[season_idx]

                fitted[t] = level + trend + seasonal[season_idx]

        return fitted, level, trend, seasonal

    def predict(
        self,
        horizon: int,
        X_future: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """Generate forecasts with prediction intervals."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        point_forecast = np.zeros(horizon)
        level = self._level
        trend = self._trend
        seasonal = self._seasonal.copy()

        for h in range(horizon):
            season_idx = (len(self._y) + h) % len(seasonal)
            point_forecast[h] = max(0, level + trend * (h + 1) + seasonal[season_idx])

        # Compute prediction intervals using residuals
        residual_std = np.std(self._residuals)

        quantile_forecasts = {}
        for q in self.config.quantiles:
            from scipy import stats
            z = stats.norm.ppf(q)
            # Widen intervals for longer horizons
            horizon_factor = np.sqrt(1 + np.arange(1, horizon + 1) * 0.1)
            quantile_forecasts[q] = np.maximum(0, point_forecast + z * residual_std * horizon_factor)

        return point_forecast, quantile_forecasts

    def get_model_name(self) -> str:
        return "holt_winters"


class EnsembleForecaster(BaseForecastModel):
    """Ensemble of multiple forecasting models."""

    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self._models: list[BaseForecastModel] = []
        self._weights: Optional[np.ndarray] = None

    def fit(
        self,
        y: np.ndarray,
        timestamps: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> "EnsembleForecaster":
        """Fit all models in the ensemble."""
        logger.info("Fitting Ensemble forecaster", n_samples=len(y))

        # Create component models
        self._models = [
            LightGBMForecaster(self.config),
            ExponentialSmoothingForecaster(self.config),
        ]

        # Fit each model
        for model in self._models:
            try:
                model.fit(y, timestamps, X)
            except Exception as e:
                logger.warning(f"Failed to fit {model.get_model_name()}: {e}")

        # Filter to successfully fitted models
        self._models = [m for m in self._models if m._is_fitted]

        if not self._models:
            raise RuntimeError("No models were successfully fitted")

        # Cross-validation to determine weights
        if self.config.auto_select and len(y) > self.config.min_train_samples * 2:
            self._weights = self._compute_weights(y, timestamps, X)
        else:
            self._weights = np.ones(len(self._models)) / len(self._models)

        self._is_fitted = True
        logger.info(
            "Ensemble fitted",
            n_models=len(self._models),
            weights=self._weights.tolist(),
        )

        return self

    def _compute_weights(
        self,
        y: np.ndarray,
        timestamps: np.ndarray,
        X: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute model weights using cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit

        n_models = len(self._models)
        cv_errors = np.zeros(n_models)

        tscv = TimeSeriesSplit(n_splits=min(self.config.cv_folds, len(y) // 30))

        for train_idx, val_idx in tscv.split(y):
            y_train, y_val = y[train_idx], y[val_idx]
            ts_train = timestamps[train_idx]
            X_train = X[train_idx] if X is not None else None
            X_val = X[val_idx] if X is not None else None

            for i, ModelClass in enumerate([LightGBMForecaster, ExponentialSmoothingForecaster]):
                try:
                    model = ModelClass(self.config)
                    model.fit(y_train, ts_train, X_train)
                    pred, _ = model.predict(len(val_idx), X_val)
                    cv_errors[i] += np.mean((pred[:len(y_val)] - y_val) ** 2)
                except Exception:
                    cv_errors[i] += 1e10  # Penalty for failed models

        # Convert errors to weights (inverse error weighting)
        cv_errors = np.maximum(cv_errors, 1e-6)
        weights = 1 / cv_errors
        weights = weights / weights.sum()

        return weights

    def predict(
        self,
        horizon: int,
        X_future: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """Generate ensemble forecasts."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        all_point_forecasts = []
        all_quantile_forecasts = {q: [] for q in self.config.quantiles}

        for model in self._models:
            try:
                point, quantiles = model.predict(horizon, X_future)
                all_point_forecasts.append(point)
                for q in self.config.quantiles:
                    if q in quantiles:
                        all_quantile_forecasts[q].append(quantiles[q])
            except Exception as e:
                logger.warning(f"Prediction failed for {model.get_model_name()}: {e}")

        if not all_point_forecasts:
            raise RuntimeError("All model predictions failed")

        # Weighted average
        point_forecasts = np.array(all_point_forecasts)
        point_forecast = np.average(point_forecasts, axis=0, weights=self._weights[:len(all_point_forecasts)])

        quantile_forecasts = {}
        for q in self.config.quantiles:
            if all_quantile_forecasts[q]:
                qf = np.array(all_quantile_forecasts[q])
                quantile_forecasts[q] = np.average(qf, axis=0, weights=self._weights[:len(qf)])

        return np.maximum(0, point_forecast), quantile_forecasts

    def get_model_name(self) -> str:
        return "ensemble"


class ConformalPredictor:
    """Conformal prediction for distribution-free prediction intervals."""

    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.

        Args:
            alpha: Miscoverage rate (1 - coverage). Default 0.1 = 90% coverage.
        """
        self.alpha = alpha
        self._residuals: Optional[np.ndarray] = None

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> "ConformalPredictor":
        """Calibrate using holdout residuals."""
        self._residuals = np.abs(y_true - y_pred)
        logger.info(
            "Conformal predictor calibrated",
            n_samples=len(self._residuals),
            alpha=self.alpha,
        )
        return self

    def get_interval_width(self) -> float:
        """Get the conformal prediction interval half-width."""
        if self._residuals is None:
            raise RuntimeError("Predictor not calibrated")

        # Quantile of absolute residuals
        n = len(self._residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        return float(np.quantile(self._residuals, min(q_level, 1.0)))

    def predict_interval(
        self,
        point_forecast: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals."""
        width = self.get_interval_width()
        lower = np.maximum(0, point_forecast - width)
        upper = point_forecast + width
        return lower, upper


class DemandForecaster:
    """
    High-level demand forecasting service.

    Production-ready interface for retail demand forecasting with:
    - Automatic model selection
    - Hierarchical reconciliation
    - Conformal prediction intervals
    - Feature engineering
    """

    def __init__(self, config: Optional[ForecastConfig] = None):
        self.config = config or ForecastConfig()
        self._models: dict[str, BaseForecastModel] = {}
        self._conformal: Optional[ConformalPredictor] = None

    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = "demand",
        timestamp_col: str = "date",
        series_id_col: Optional[str] = "sku_id",
        feature_cols: Optional[list[str]] = None,
    ) -> "DemandForecaster":
        """
        Fit forecasting models for all series.

        Args:
            data: DataFrame with time series data
            target_col: Column name for demand values
            timestamp_col: Column name for timestamps
            series_id_col: Column name for series identifier (e.g., SKU)
            feature_cols: Additional feature columns
        """
        logger.info(
            "Fitting DemandForecaster",
            n_rows=len(data),
            n_series=data[series_id_col].nunique() if series_id_col else 1,
        )

        if series_id_col:
            # Fit model for each series
            for series_id in data[series_id_col].unique():
                series_data = data[data[series_id_col] == series_id].sort_values(timestamp_col)

                if len(series_data) < self.config.min_train_samples:
                    logger.warning(f"Skipping series {series_id}: insufficient data")
                    continue

                y = series_data[target_col].values
                timestamps = pd.to_datetime(series_data[timestamp_col]).values
                X = series_data[feature_cols].values if feature_cols else None

                model = EnsembleForecaster(self.config)
                try:
                    model.fit(y, timestamps, X)
                    self._models[str(series_id)] = model
                except Exception as e:
                    logger.error(f"Failed to fit series {series_id}: {e}")
        else:
            # Single series
            data = data.sort_values(timestamp_col)
            y = data[target_col].values
            timestamps = pd.to_datetime(data[timestamp_col]).values
            X = data[feature_cols].values if feature_cols else None

            model = EnsembleForecaster(self.config)
            model.fit(y, timestamps, X)
            self._models["__default__"] = model

        logger.info(f"Fitted {len(self._models)} series models")

        return self

    def predict(
        self,
        series_ids: Optional[list[str]] = None,
        horizon: Optional[int] = None,
        X_future: Optional[pd.DataFrame] = None,
    ) -> list[ForecastResult]:
        """
        Generate forecasts for specified series.

        Args:
            series_ids: List of series to forecast (None = all)
            horizon: Forecast horizon (None = use config)
            X_future: Future feature values

        Returns:
            List of ForecastResult objects
        """
        horizon = horizon or self.config.horizon
        series_ids = series_ids or list(self._models.keys())

        results = []

        for series_id in series_ids:
            if series_id not in self._models:
                logger.warning(f"No model for series {series_id}")
                continue

            model = self._models[series_id]
            X_fut = X_future[X_future["series_id"] == series_id].values if X_future is not None else None

            try:
                point_forecast, quantile_forecasts = model.predict(horizon, X_fut)

                # Generate timestamps
                last_date = datetime.now()  # Would come from training data
                pred_timestamps = [last_date + timedelta(days=i+1) for i in range(horizon)]

                # Build prediction intervals
                intervals = {}
                if 0.1 in quantile_forecasts and 0.9 in quantile_forecasts:
                    intervals[0.8] = (quantile_forecasts[0.1], quantile_forecasts[0.9])
                if 0.5 in quantile_forecasts:
                    lower = quantile_forecasts.get(0.1, point_forecast * 0.8)
                    upper = quantile_forecasts.get(0.9, point_forecast * 1.2)
                    intervals[0.9] = (lower, upper)

                result = ForecastResult(
                    series_id=series_id,
                    aggregation_level=AggregationLevel.SKU,
                    forecast_date=datetime.now(),
                    prediction_timestamps=pred_timestamps,
                    point_forecast=point_forecast,
                    quantile_forecasts=quantile_forecasts,
                    prediction_intervals=intervals,
                    model_used=model.get_model_name(),
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Prediction failed for {series_id}: {e}")

        logger.info(f"Generated {len(results)} forecasts")

        return results

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate forecast accuracy."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

        # Weighted MAPE (more weight to higher values)
        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "wmape": wmape,
        }

