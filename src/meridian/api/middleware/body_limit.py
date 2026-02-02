"""Request body size limiting middleware."""

from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from meridian.core.config import settings
from meridian.core.logging import get_logger

logger = get_logger(__name__)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies larger than configured limit."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > settings.max_request_body_bytes:
                    logger.warning(
                        "Request body too large",
                        content_length=length,
                        limit=settings.max_request_body_bytes,
                        path=request.url.path,
                    )
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large"},
                    )
            except ValueError:
                logger.warning("Invalid content-length header", value=content_length)

        return await call_next(request)

