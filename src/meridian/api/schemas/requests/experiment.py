"""Experiment request schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class ExperimentVariant(BaseModel):
    """Experiment variant definition."""

    name: str = Field(..., min_length=1, max_length=64)
    weight: float = Field(default=0.5, ge=0, le=1)
    description: Optional[str] = None


class CreateExperimentRequest(BaseModel):
    """Request schema for creating an experiment."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Experiment name",
    )
    hypothesis: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Experiment hypothesis",
    )
    variants: list[ExperimentVariant] = Field(
        default_factory=lambda: [
            ExperimentVariant(name="control", weight=0.5),
            ExperimentVariant(name="treatment", weight=0.5),
        ],
        description="Experiment variants",
    )
    primary_metric: str = Field(
        default="conversion_rate",
        description="Primary metric to optimize",
    )
    target_mde: Optional[float] = Field(
        None,
        gt=0,
        lt=1,
        description="Target minimum detectable effect",
    )


class UpdateExperimentRequest(BaseModel):
    """Request schema for updating an experiment."""

    name: Optional[str] = Field(None, min_length=1, max_length=128)
    hypothesis: Optional[str] = Field(None, min_length=10, max_length=1000)
    status: Optional[str] = Field(
        None,
        pattern="^(draft|running|completed|archived)$",
    )

