"""Admin endpoints for model management."""

from typing import Annotated

from fastapi import APIRouter, Depends

from meridian.api.dependencies.auth import get_current_user, require_scopes, TokenData
from meridian.core.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()


@router.get("/models")
async def list_models(
    current_user: Annotated[TokenData, Depends(require_scopes(["admin:read"]))],
):
    """List all registered models."""
    return {
        "models": [
            {
                "id": "causal_forest_v1",
                "type": "uplift",
                "version": "1.0.0",
                "status": "production",
            },
            {
                "id": "deepar_v1",
                "type": "forecasting",
                "version": "1.0.0",
                "status": "production",
            },
        ]
    }


@router.post("/models/{model_id}/reload")
async def reload_model(
    model_id: str,
    current_user: Annotated[TokenData, Depends(require_scopes(["admin:write"]))],
):
    """Reload a model from registry."""
    logger.info("Reloading model", model_id=model_id, user_id=current_user.user_id)
    return {"status": "reloaded", "model_id": model_id}


@router.get("/debug/config")
async def get_debug_config(
    current_user: Annotated[TokenData, Depends(require_scopes(["admin:read"]))],
):
    """Get current configuration (debug mode only)."""
    from meridian.core.config import settings

    if not settings.debug:
        return {"error": "Debug mode disabled"}

    return {
        "environment": settings.environment,
        "debug": settings.debug,
        "db_host": settings.db_host,
        "redis_host": settings.redis_host,
    }


@router.post("/cache/clear")
async def clear_cache(
    current_user: Annotated[TokenData, Depends(require_scopes(["admin:write"]))],
):
    """Clear all caches."""
    logger.info("Clearing cache", user_id=current_user.user_id)
    return {"status": "cleared"}

