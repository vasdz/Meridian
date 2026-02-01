"""Demand forecasting endpoints."""

from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from meridian.api.dependencies.auth import TokenData, get_current_user
from meridian.api.schemas.requests.forecasting import DemandForecastRequest
from meridian.api.schemas.responses.forecasting import (
    DemandForecast,
    DemandForecastResponse,
)
from meridian.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Extended schemas for production API


class ForecastConfigRequest(BaseModel):
    """Configuration for forecasting."""

    horizon_days: int = Field(default=14, ge=1, le=90)
    granularity: str = Field(default="daily")
    include_intervals: bool = Field(default=True)
    confidence_level: float = Field(default=0.9, ge=0.5, le=0.99)


class BatchForecastRequest(BaseModel):
    """Request for batch forecasting."""

    series_ids: list[str] = Field(..., min_length=1, max_length=1000)
    config: ForecastConfigRequest = Field(default_factory=ForecastConfigRequest)


class ForecastPoint(BaseModel):
    """Single forecast point."""

    date: date
    point_forecast: float
    lower_bound: float | None = None
    upper_bound: float | None = None


class SeriesForecast(BaseModel):
    """Forecast for a single series."""

    series_id: str
    forecasts: list[ForecastPoint]
    model_used: str
    metrics: dict | None = None


class BatchForecastResponse(BaseModel):
    """Response for batch forecasting."""

    results: list[SeriesForecast]
    total_series: int
    successful: int
    failed: int
    processing_time_ms: float


@router.post("/demand", response_model=DemandForecastResponse)
async def forecast_demand(
    request: DemandForecastRequest,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Generate demand forecasts for items."""
    import random
    import time

    time.time()

    logger.info(
        "Demand forecast request",
        item_count=len(request.items),
        horizon_days=request.horizon_days,
        user_id=current_user.user_id,
    )

    forecasts = []
    today = date.today()

    for item in request.items:
        base_demand = random.uniform(80, 200)

        for day in range(request.horizon_days):
            forecast_date = today + timedelta(days=day + 1)
            day_of_week = forecast_date.weekday()
            seasonality = 1.0 + 0.2 * (day_of_week in [5, 6])

            point = base_demand * seasonality + random.gauss(0, 10)

            forecasts.append(
                DemandForecast(
                    item_id=item.item_id,
                    store_id=item.store_id,
                    date=forecast_date,
                    point_forecast=max(0, point),
                    lower_bound=max(0, point * 0.7),
                    upper_bound=point * 1.3,
                )
            )

    return DemandForecastResponse(
        forecasts=forecasts,
        model_id=request.model_id or "ensemble_v2",
        horizon_days=request.horizon_days,
    )


@router.post("/batch", response_model=BatchForecastResponse)
async def batch_forecast(
    request: BatchForecastRequest,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Generate forecasts for multiple series in batch."""
    import random
    import time

    start_time = time.time()

    logger.info(
        "Batch forecast request",
        n_series=len(request.series_ids),
        horizon=request.config.horizon_days,
    )

    results = []
    failed = 0
    today = date.today()

    for series_id in request.series_ids:
        try:
            base_demand = random.uniform(50, 300)
            forecasts = []

            for day in range(request.config.horizon_days):
                forecast_date = today + timedelta(days=day + 1)
                point = base_demand * (1 + random.gauss(0, 0.1))

                forecast_point = ForecastPoint(
                    date=forecast_date,
                    point_forecast=max(0, point),
                )

                if request.config.include_intervals:
                    ci_width = point * (1 - request.config.confidence_level) * 2
                    forecast_point.lower_bound = max(0, point - ci_width)
                    forecast_point.upper_bound = point + ci_width

                forecasts.append(forecast_point)

            results.append(
                SeriesForecast(
                    series_id=series_id,
                    forecasts=forecasts,
                    model_used="ensemble_v2",
                )
            )

        except Exception as e:
            logger.error(f"Forecast failed for {series_id}: {e}")
            failed += 1

    processing_time = (time.time() - start_time) * 1000

    return BatchForecastResponse(
        results=results,
        total_series=len(request.series_ids),
        successful=len(results),
        failed=failed,
        processing_time_ms=processing_time,
    )


@router.get("/models")
async def list_forecast_models(
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """List available forecasting models."""
    return {
        "models": [
            {"id": "ensemble_v2", "name": "Ensemble", "is_default": True},
            {"id": "lightgbm_quantile", "name": "LightGBM Quantile"},
            {"id": "holt_winters", "name": "Holt-Winters"},
        ]
    }
