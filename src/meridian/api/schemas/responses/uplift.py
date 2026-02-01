"""Uplift prediction response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class ConfidenceInterval(BaseModel):
    """Confidence interval."""

    lower: float
    upper: float


class UpliftPrediction(BaseModel):
    """Single uplift prediction."""

    customer_id: str
    cate: float = Field(..., description="Conditional Average Treatment Effect")
    confidence_interval: Optional[ConfidenceInterval] = None
    treatment_probability: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Probability that treatment effect is positive",
    )


class UpliftPredictionResponse(BaseModel):
    """Response schema for uplift predictions."""

    predictions: list[UpliftPrediction]
    model_id: str
    model_version: str

