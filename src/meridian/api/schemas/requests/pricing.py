"""Pricing request schemas."""

from pydantic import BaseModel, Field, field_validator

from meridian.api.dependencies.security import validate_identifier


class ProductPricing(BaseModel):
    """Product pricing information."""

    product_id: str = Field(..., min_length=1, max_length=64)
    current_price: float = Field(..., gt=0)
    cost: float | None = Field(None, ge=0)

    @field_validator("product_id")
    @classmethod
    def _validate_product_id(cls, value: str) -> str:
        return validate_identifier(value, "product_id")


class PriceOptimizationRequest(BaseModel):
    """Request schema for price optimization."""

    products: list[ProductPricing] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Products to optimize",
    )
    objective: str = Field(
        default="maximize_profit",
        pattern="^(maximize_profit|maximize_revenue)$",
        description="Optimization objective",
    )
    constraints: dict | None = Field(
        None,
        description="Optimization constraints",
    )
