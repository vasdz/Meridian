"""Customer endpoints."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query

from meridian.api.dependencies.auth import get_current_user, TokenData
from meridian.core.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()


@router.get("")
async def list_customers(
    current_user: Annotated[TokenData, Depends(get_current_user)],
    segment: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List customers with optional filtering."""
    return {
        "customers": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{customer_id}")
async def get_customer(
    customer_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Get customer details."""
    return {
        "id": customer_id,
        "segment": "high_value",
        "features": {},
    }


@router.get("/{customer_id}/features")
async def get_customer_features(
    customer_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Get customer features for ML."""
    return {
        "customer_id": customer_id,
        "features": {
            "age": 35,
            "tenure_days": 365,
            "total_spend": 1500.0,
            "transaction_count": 25,
        },
    }


@router.get("/{customer_id}/predictions")
async def get_customer_predictions(
    customer_id: str,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """Get all predictions for a customer."""
    return {
        "customer_id": customer_id,
        "uplift": {"cate": 0.05, "model": "causal_forest_v1"},
        "churn_risk": 0.15,
        "ltv": 2500.0,
    }

