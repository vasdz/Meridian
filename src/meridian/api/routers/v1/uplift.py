"""Uplift prediction endpoints."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends

from meridian.api.dependencies.auth import get_current_user, TokenData
from meridian.api.dependencies.rate_limit import RateLimited
from meridian.api.schemas.requests.uplift import (
    UpliftPredictionRequest,
    UpliftBatchPredictionRequest,
)
from meridian.api.schemas.responses.uplift import (
    UpliftPredictionResponse,
    UpliftPrediction,
)
from meridian.core.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()


@router.post("/predict", response_model=UpliftPredictionResponse)
async def predict_uplift(
    request: UpliftPredictionRequest,
    current_user: Annotated[TokenData, Depends(get_current_user)],
    _rate_limit: RateLimited,
):
    """
    Predict uplift (CATE) for customers.

    Returns the Conditional Average Treatment Effect for each customer,
    indicating the expected incremental impact of treatment.
    """
    logger.info(
        "Uplift prediction request",
        customer_count=len(request.customer_ids),
        model_id=request.model_id,
        user_id=current_user.user_id,
    )

    # Placeholder predictions
    import random
    predictions = [
        UpliftPrediction(
            customer_id=cid,
            cate=random.uniform(-0.05, 0.15),
            confidence_interval={"lower": -0.02, "upper": 0.12}
            if request.return_confidence_intervals else None,
        )
        for cid in request.customer_ids
    ]

    return UpliftPredictionResponse(
        predictions=predictions,
        model_id=request.model_id,
        model_version="1.0.0",
    )


@router.post("/predict/batch", response_model=UpliftPredictionResponse)
async def predict_uplift_batch(
    request: UpliftBatchPredictionRequest,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Batch uplift prediction for large customer lists."""
    logger.info(
        "Batch uplift prediction request",
        customer_count=len(request.customer_ids),
        user_id=current_user.user_id,
    )

    # Process in batches
    predictions = []
    import random
    for cid in request.customer_ids:
        predictions.append(
            UpliftPrediction(
                customer_id=cid,
                cate=random.uniform(-0.05, 0.15),
            )
        )

    return UpliftPredictionResponse(
        predictions=predictions,
        model_id=request.model_id,
        model_version="1.0.0",
    )


@router.get("/models")
async def list_uplift_models(
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """List available uplift models."""
    return {
        "models": [
            {"id": "causal_forest_v1", "type": "causal_forest", "version": "1.0.0"},
            {"id": "x_learner_v1", "type": "x_learner", "version": "1.0.0"},
        ]
    }

