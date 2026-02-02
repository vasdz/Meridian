"""Forecasting request schemas."""

from pydantic import BaseModel, Field, field_validator

from meridian.api.dependencies.security import validate_identifier


class ForecastItem(BaseModel):
    """Item to forecast."""

    item_id: str = Field(..., min_length=1, max_length=64)
    store_id: str | None = Field(None, max_length=64)

    @field_validator("item_id")
    @classmethod
    def _validate_item_id(cls, value: str) -> str:
        return validate_identifier(value, "item_id")

    @field_validator("store_id")
    @classmethod
    def _validate_store_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_identifier(value, "store_id")


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
    model_id: str | None = Field(
        None,
        description="ID of forecasting model to use",
    )
    quantiles: list[float] = Field(
        default=[0.1, 0.5, 0.9],
        description="Prediction quantiles",
    )

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_identifier(value, "model_id")

    @field_validator("quantiles")
    @classmethod
    def _validate_quantiles(cls, values: list[float]) -> list[float]:
        for value in values:
            if value <= 0 or value >= 1:
                raise ValueError("Quantiles must be between 0 and 1")
        return values
