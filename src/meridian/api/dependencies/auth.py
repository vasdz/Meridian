"""Authentication dependencies - JWT and API Key validation."""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from meridian.core.config import settings
from meridian.core.logging import get_logger

logger = get_logger(__name__)

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


class TokenData:
    """Token payload data."""

    def __init__(self, user_id: str, scopes: list[str] = None):
        self.user_id = user_id
        self.scopes = scopes or []


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """Create JWT access token."""
    to_encode = data.copy()

    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> TokenData:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        user_id = payload.get("sub")
        scopes = payload.get("scopes", [])

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        return TokenData(user_id=user_id, scopes=scopes)

    except JWTError as e:
        logger.warning("JWT decode error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate token",
        )


async def validate_api_key(api_key: str) -> TokenData | None:
    """Validate API key against stored keys."""
    # In production, look up hashed key in database
    # For demo, accept keys with prefix "mk_"
    if api_key and api_key.startswith("mk_"):
        return TokenData(user_id="api-user", scopes=["api:read", "api:write"])
    return None


async def get_current_user(
    bearer: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> TokenData:
    """Get current authenticated user from JWT or API key."""
    # Try JWT first
    if bearer and bearer.credentials:
        return decode_access_token(bearer.credentials)

    # Try API key
    if api_key:
        token_data = await validate_api_key(api_key)
        if token_data:
            return token_data

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    bearer: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> TokenData | None:
    """Get current user if authenticated, None otherwise."""
    try:
        return await get_current_user(bearer, api_key)
    except HTTPException:
        return None


def require_scopes(required_scopes: list[str]):
    """Dependency to require specific scopes."""

    async def check_scopes(
        current_user: Annotated[TokenData, Depends(get_current_user)],
    ) -> TokenData:
        for scope in required_scopes:
            if scope not in current_user.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required scope: {scope}",
                )
        return current_user

    return check_scopes
