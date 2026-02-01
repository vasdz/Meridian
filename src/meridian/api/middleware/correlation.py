"""Correlation ID middleware for request tracing."""

import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from meridian.core.logging import set_correlation_id


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Add correlation ID to all requests."""

    CORRELATION_ID_HEADER = "X-Correlation-ID"
    REQUEST_ID_HEADER = "X-Request-ID"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate correlation ID
        correlation_id = request.headers.get(
            self.CORRELATION_ID_HEADER,
            str(uuid.uuid4()),
        )

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Set in context for logging
        set_correlation_id(correlation_id)

        # Store in request state
        request.state.correlation_id = correlation_id
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add headers to response
        response.headers[self.CORRELATION_ID_HEADER] = correlation_id
        response.headers[self.REQUEST_ID_HEADER] = request_id

        return response

