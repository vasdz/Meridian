"""Security dependencies - Input validation, SQLi protection."""

import re
from typing import Annotated

from fastapi import HTTPException, Query, status


# Patterns for detecting potentially malicious input
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)",
    r"(--|;|\/\*|\*\/)",
    r"(\bOR\b\s+\d+\s*=\s*\d+)",
    r"('|\"|`)",
]

XSS_PATTERNS = [
    r"<script[^>]*>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe",
    r"<object",
]


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize string input."""
    if not value:
        return value

    # Truncate
    value = value[:max_length]

    # Check for SQL injection
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid characters in input",
            )

    # Check for XSS
    for pattern in XSS_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid characters in input",
            )

    return value


def validate_customer_id(customer_id: str) -> str:
    """Validate customer ID format."""
    # Allow alphanumeric, hyphens, underscores
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", customer_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid customer ID format",
        )
    return customer_id


def validate_experiment_name(name: str) -> str:
    """Validate experiment name."""
    sanitized = sanitize_string(name, max_length=128)

    # Only allow alphanumeric, spaces, hyphens, underscores
    if not re.match(r"^[a-zA-Z0-9\s_-]{1,128}$", sanitized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid experiment name format",
        )

    return sanitized


# Query parameter validators
def validate_pagination(
    limit: Annotated[int, Query(ge=1, le=1000)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """Validate pagination parameters."""
    return {"limit": limit, "offset": offset}

