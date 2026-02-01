"""Global exception handler middleware."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from meridian.core.exceptions import MeridianBaseError
from meridian.core.logging import get_logger, get_correlation_id


logger = get_logger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers."""

    @app.exception_handler(MeridianBaseError)
    async def meridian_exception_handler(
        request: Request,
        exc: MeridianBaseError,
    ) -> JSONResponse:
        """Handle Meridian domain exceptions."""
        logger.warning(
            "Domain exception",
            error_code=exc.error_code,
            message=exc.message,
            path=request.url.path,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request,
        exc: ValueError,
    ) -> JSONResponse:
        """Handle validation errors."""
        logger.warning(
            "Validation error",
            error=str(exc),
            path=request.url.path,
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(exc),
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions without leaking details."""
        correlation_id = get_correlation_id()

        # Log full error for debugging
        logger.exception(
            "Unhandled exception",
            error=str(exc),
            path=request.url.path,
            correlation_id=correlation_id,
        )

        # Return generic message to client
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

