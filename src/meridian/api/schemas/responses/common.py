"""Common response schemas."""

from typing import Any, Optional

from pydantic import BaseModel


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = True
    message: Optional[str] = None


class ErrorDetail(BaseModel):
    """Error detail."""

    code: str
    message: str
    details: Optional[dict[str, Any]] = None
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""

    error: ErrorDetail


class PaginatedResponse(BaseModel):
    """Base for paginated responses."""

    total: int
    limit: int
    offset: int


class ValidationErrorDetail(BaseModel):
    """Validation error detail."""

    loc: list[str]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """Validation error response."""

    detail: list[ValidationErrorDetail]

