"""Audit logging middleware."""

import json
import time
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from meridian.core.config import settings
from meridian.core.logging import get_correlation_id, get_logger

logger = get_logger(__name__)

REDACTED = "[REDACTED]"
SENSITIVE_QUERY_KEYS = {"email", "phone", "token", "password", "secret", "api_key"}
SENSITIVE_BODY_KEYS = {"email", "phone", "token", "password", "secret", "api_key"}
MAX_BODY_BYTES = 16 * 1024


def _is_json_content_type(content_type: str) -> bool:
    return "application/json" in content_type or content_type.endswith("+json")


def _redact_query_params(request: Request) -> str | None:
    if not request.query_params:
        return None

    redacted = []
    for key, value in request.query_params.multi_items():
        if key.lower() in SENSITIVE_QUERY_KEYS:
            redacted.append((key, REDACTED))
        else:
            redacted.append((key, value[:200]))

    return str(redacted)[:500]


def _redact_json(value: object) -> object:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_BODY_KEYS:
                redacted[key] = REDACTED
            else:
                redacted[key] = _redact_json(item)
        return redacted

    if isinstance(value, list):
        return [_redact_json(item) for item in value]

    if isinstance(value, str) and len(value) > 256:
        return value[:256]

    return value


async def _redact_request_body(request: Request) -> tuple[str | None, bool, bool]:
    content_type = request.headers.get("content-type", "")
    if not _is_json_content_type(content_type):
        return None, False, False

    body = await request.body()
    request._body = body  # Preserve body for downstream handlers.

    if not body:
        return None, False, False

    if len(body) > MAX_BODY_BYTES:
        return "[TRUNCATED]", True, False

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "[UNPARSABLE]", False, True

    return json.dumps(_redact_json(payload))[:1000], False, False


def _redact_response_body(response: Response) -> tuple[str | None, bool, bool]:
    content_type = response.headers.get("content-type", "")
    if not _is_json_content_type(content_type):
        return None, False, False

    body = getattr(response, "body", None)
    if not body:
        return None, False, False

    if len(body) > settings.max_response_body_bytes:
        return "[TRUNCATED]", True, False

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "[UNPARSABLE]", False, True

    return json.dumps(_redact_json(payload))[:1000], False, False


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log all API requests for audit trail."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        request_id = request.headers.get("X-Request-ID", "")
        correlation_id = get_correlation_id()
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "")
        api_key = request.headers.get("X-API-Key", "")
        user_id = getattr(request.state, "user_id", None)
        org_id = getattr(request.state, "org_id", None)
        app_id = getattr(request.state, "app_id", None)
        body_preview, body_truncated, body_parse_error = await _redact_request_body(request)
        content_type = request.headers.get("content-type", "")
        content_length = request.headers.get("content-length")

        response = await call_next(request)

        response_body = None
        response_truncated = False
        response_parse_error = False
        if settings.log_response_body:
            response_body, response_truncated, response_parse_error = _redact_response_body(response)

        response_time_ms = (time.time() - start_time) * 1000

        audit_data = {
            "request_id": request_id,
            "correlation_id": correlation_id,
            "user_id": user_id,
            "org_id": org_id,
            "app_id": app_id,
            "client_ip": client_ip,
            "user_agent": user_agent[:256] if user_agent else None,
            "api_key_prefix": api_key[:8] if api_key else None,
            "method": request.method,
            "path": request.url.path,
            "query_params": _redact_query_params(request),
            "body": body_preview,
            "body_truncated": body_truncated,
            "body_parse_error": body_parse_error,
            "content_type": content_type,
            "content_length": content_length,
            "response_body": response_body,
            "response_truncated": response_truncated,
            "response_parse_error": response_parse_error,
            "status_code": response.status_code,
            "response_time_ms": response_time_ms,
        }

        logger.info("API request", **audit_data)

        return response
