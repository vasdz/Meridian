"""Dependency injection container."""

from typing import Annotated

from fastapi import Depends

from meridian.domain.services.experiment_design import ExperimentDesignService
from meridian.domain.services.pricing_optimizer import PricingOptimizer
from meridian.domain.services.uplift_calculator import UpliftCalculator
from meridian.infrastructure.cache.redis_cache import RedisCache
from meridian.infrastructure.ml.model_registry import ModelRegistry

# Singleton instances
_model_registry = None
_cache = None
_uplift_calculator = None
_experiment_design = None
_pricing_optimizer = None


def get_model_registry() -> ModelRegistry:
    """Get model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def get_cache() -> RedisCache:
    """Get cache instance."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache


def get_uplift_calculator() -> UpliftCalculator:
    """Get uplift calculator instance."""
    global _uplift_calculator
    if _uplift_calculator is None:
        _uplift_calculator = UpliftCalculator()
    return _uplift_calculator


def get_experiment_design_service() -> ExperimentDesignService:
    """Get experiment design service instance."""
    global _experiment_design
    if _experiment_design is None:
        _experiment_design = ExperimentDesignService()
    return _experiment_design


def get_pricing_optimizer() -> PricingOptimizer:
    """Get pricing optimizer instance."""
    global _pricing_optimizer
    if _pricing_optimizer is None:
        _pricing_optimizer = PricingOptimizer()
    return _pricing_optimizer


# Type aliases for dependency injection
ModelRegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
CacheDep = Annotated[RedisCache, Depends(get_cache)]
UpliftCalculatorDep = Annotated[UpliftCalculator, Depends(get_uplift_calculator)]
ExperimentDesignDep = Annotated[ExperimentDesignService, Depends(get_experiment_design_service)]
PricingOptimizerDep = Annotated[PricingOptimizer, Depends(get_pricing_optimizer)]
