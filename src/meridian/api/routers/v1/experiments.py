"""A/B experiment management endpoints."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query

from meridian.api.dependencies.auth import get_current_user, TokenData
from meridian.api.schemas.requests.experiment import (
    CreateExperimentRequest,
    UpdateExperimentRequest,
)
from meridian.core.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()


@router.get("")
async def list_experiments(
    current_user: Annotated[TokenData, Depends(get_current_user)],
    status: Optional[str] = Query(None, pattern="^(draft|running|completed|archived)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all experiments with optional filtering."""
    logger.info(
        "List experiments",
        status=status,
        user_id=current_user.user_id,
    )

    # Placeholder data
    return {
        "experiments": [
            {
                "id": "exp-1",
                "name": "Promo Email Test",
                "status": "running",
                "created_at": "2024-01-01T00:00:00Z",
            },
        ],
        "total": 1,
        "limit": limit,
        "offset": offset,
    }


@router.post("", status_code=201)
async def create_experiment(
    request: CreateExperimentRequest,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Create a new A/B experiment."""
    logger.info(
        "Create experiment",
        name=request.name,
        user_id=current_user.user_id,
    )

    import uuid
    return {
        "id": str(uuid.uuid4()),
        "name": request.name,
        "hypothesis": request.hypothesis,
        "status": "draft",
        "variants": request.variants,
    }


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Get experiment details."""
    return {
        "id": experiment_id,
        "name": "Test Experiment",
        "status": "running",
    }


@router.post("/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Start an experiment."""
    logger.info("Starting experiment", experiment_id=experiment_id)
    return {"status": "running", "experiment_id": experiment_id}


@router.post("/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Stop an experiment."""
    logger.info("Stopping experiment", experiment_id=experiment_id)
    return {"status": "completed", "experiment_id": experiment_id}


@router.get("/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Get experiment analysis results."""
    return {
        "experiment_id": experiment_id,
        "lift": 0.05,
        "p_value": 0.03,
        "is_significant": True,
        "confidence_interval": {"lower": 0.02, "upper": 0.08},
    }


@router.post("/sample-size")
async def calculate_sample_size(
    current_user: Annotated[TokenData, Depends(get_current_user)],
    baseline_rate: float = Query(..., gt=0, lt=1),
    mde: float = Query(..., gt=0, lt=1),
    power: float = Query(0.8, gt=0, lt=1),
):
    """Calculate required sample size for an experiment."""
    # Simple approximation
    import math
    from scipy import stats

    z_alpha = stats.norm.ppf(0.975)
    z_power = stats.norm.ppf(power)

    p = baseline_rate
    effect = baseline_rate * mde

    n = 2 * ((z_alpha + z_power) ** 2) * p * (1 - p) / (effect ** 2)

    return {
        "required_sample_size": int(math.ceil(n)),
        "sample_size_per_variant": int(math.ceil(n / 2)),
        "baseline_rate": baseline_rate,
        "mde": mde,
        "power": power,
    }

