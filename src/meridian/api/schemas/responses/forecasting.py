"""Forecasting response schemas."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class DemandForecast(BaseModel):
    """Single demand forecast."""

    item_id: str
    store_id: Optional[str] = None
    date: date
    point_forecast: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class DemandForecastResponse(BaseModel):
    """Response schema for demand forecasts."""

    forecasts: list[DemandForecast]
    model_id: str
    horizon_days: int

