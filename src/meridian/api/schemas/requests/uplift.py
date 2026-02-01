"""Uplift prediction request schemas."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class UpliftPredictionRequest(BaseModel):
    """Request schema for uplift prediction."""

    customer_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of customer IDs to predict",
    )
    model_id: str = Field(
        default="causal_forest_v1",
        description="ID of the uplift model to use",
    )
    return_confidence_intervals: bool = Field(
        default=False,
        description="Whether to return confidence intervals",
    )

    @field_validator("customer_ids")
    @classmethod
    def validate_customer_ids(cls, v):
        """Validate customer ID format."""
        import re
        for cid in v:
            if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", cid):
                raise ValueError(f"Invalid customer ID format: {cid}")
        return v


class UpliftBatchPredictionRequest(BaseModel):
    """Request schema for batch uplift prediction."""

    customer_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="List of customer IDs for batch prediction",
    )
    model_id: str = Field(
        default="causal_forest_v1",
        description="ID of the uplift model to use",
    )
    batch_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Batch size for processing",
    )

