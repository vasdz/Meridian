"""Domain exceptions."""

from typing import Any


class MeridianBaseError(Exception):
    """Base exception for all Meridian errors."""

    error_code: str = "MERIDIAN_ERROR"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(MeridianBaseError):
    """Input validation error."""

    error_code = "VALIDATION_ERROR"
    status_code = 422


class NotFoundError(MeridianBaseError):
    """Resource not found error."""

    error_code = "NOT_FOUND"
    status_code = 404


class AuthenticationError(MeridianBaseError):
    """Authentication failed error."""

    error_code = "AUTHENTICATION_ERROR"
    status_code = 401


class AuthorizationError(MeridianBaseError):
    """Authorization failed error."""

    error_code = "AUTHORIZATION_ERROR"
    status_code = 403


class RateLimitExceededError(MeridianBaseError):
    """Rate limit exceeded error."""

    error_code = "RATE_LIMIT_EXCEEDED"
    status_code = 429


class UpliftModelError(MeridianBaseError):
    """Uplift model error."""

    error_code = "UPLIFT_MODEL_ERROR"
    status_code = 500


class ForecastingError(MeridianBaseError):
    """Forecasting model error."""

    error_code = "FORECASTING_ERROR"
    status_code = 500


class PricingError(MeridianBaseError):
    """Pricing optimization error."""

    error_code = "PRICING_ERROR"
    status_code = 500


class ExperimentError(MeridianBaseError):
    """Experiment management error."""

    error_code = "EXPERIMENT_ERROR"
    status_code = 400


class FeatureStoreError(MeridianBaseError):
    """Feature store error."""

    error_code = "FEATURE_STORE_ERROR"
    status_code = 500


class DatabaseError(MeridianBaseError):
    """Database operation error."""

    error_code = "DATABASE_ERROR"
    status_code = 500


class CacheError(MeridianBaseError):
    """Cache operation error."""

    error_code = "CACHE_ERROR"
    status_code = 500


class ExternalServiceError(MeridianBaseError):
    """External service error."""

    error_code = "EXTERNAL_SERVICE_ERROR"
    status_code = 502
