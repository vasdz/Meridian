"""Audit logging middleware."""

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from meridian.core.logging import get_logger, get_correlation_id
from meridian.infrastructure.security.audit_logger import AuditLogger


logger = get_logger(__name__)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log all API requests for audit trail."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Extract request info
        request_id = request.headers.get("X-Request-ID", "")
        correlation_id = get_correlation_id()
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "")
        api_key = request.headers.get("X-API-Key", "")

        # Process request
        response = await call_next(request)

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Log request
        audit_data = {
            "request_id": request_id,
            "correlation_id": correlation_id,
            "client_ip": client_ip,
            "user_agent": user_agent[:512] if user_agent else None,
            "api_key_prefix": api_key[:20] if api_key else None,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params)[:500] if request.query_params else None,
            "status_code": response.status_code,
            "response_time_ms": response_time_ms,
        }

        # Log to structured logger
        logger.info(
            "API request",
            **audit_data,
        )

        # Log to database (async, fire-and-forget)
        # await AuditLogger.log_request(audit_data)

        return response

