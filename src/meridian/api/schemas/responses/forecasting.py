"""Forecasting response schemas."""

from datetime import date

from pydantic import BaseModel


class DemandForecast(BaseModel):
    """Single demand forecast."""

    item_id: str
    store_id: str | None = None
    date: date
    point_forecast: float
    lower_bound: float | None = None
    upper_bound: float | None = None


class DemandForecastResponse(BaseModel):
    """Response schema for demand forecasts."""

    forecasts: list[DemandForecast]
    model_id: str
    horizon_days: int
