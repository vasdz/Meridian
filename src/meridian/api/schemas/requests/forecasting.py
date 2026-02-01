"""Forecasting request schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class ForecastItem(BaseModel):
    """Item to forecast."""

    item_id: str = Field(..., min_length=1, max_length=64)
    store_id: Optional[str] = Field(None, max_length=64)


class DemandForecastRequest(BaseModel):
    """Request schema for demand forecasting."""

    items: list[ForecastItem] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Items to forecast",
    )
    horizon_days: int = Field(
        default=14,
        ge=1,
        le=365,
        description="Forecast horizon in days",
    )
    model_id: Optional[str] = Field(
        None,
        description="ID of forecasting model to use",
    )
    quantiles: list[float] = Field(
        default=[0.1, 0.5, 0.9],
        description="Prediction quantiles",
    )

