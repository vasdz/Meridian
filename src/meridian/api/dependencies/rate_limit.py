"""Rate limiting dependency."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from meridian.core.config import settings
from meridian.core.logging import get_logger
from meridian.infrastructure.cache.redis_cache import RedisCache

logger = get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter using Redis."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 100,
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._cache = RedisCache()

    async def check_rate_limit(self, key: str, limit: int | None = None) -> bool:
        """Check if request is within rate limit."""
        if not settings.rate_limit_enabled:
            return True

        rate_key = f"rate:{key}"
        effective_limit = limit or (self.requests_per_minute + self.burst_size)

        try:
            current = await self._cache.incr(rate_key)

            if current == 1:
                await self._cache.expire(rate_key, 60)

            return current <= effective_limit

        except Exception as e:
            logger.warning("Rate limit check failed", error=str(e))
            return True  # Fail open

    async def get_remaining(self, key: str, limit: int | None = None) -> int:
        """Get remaining requests."""
        rate_key = f"rate:{key}"
        effective_limit = limit or (self.requests_per_minute + self.burst_size)

        try:
            value = await self._cache.get(rate_key)
            if value is None:
                return effective_limit
            return max(0, effective_limit - int(value))
        except Exception:
            return effective_limit


# Default rate limiter instance
rate_limiter = RateLimiter(
    requests_per_minute=settings.rate_limit_requests_per_minute,
    burst_size=settings.rate_limit_burst_size,
)


async def check_rate_limit(request: Request) -> None:
    """Rate limiting dependency."""
    if not settings.rate_limit_enabled:
        return

    client_ip = request.client.host if request.client else "unknown"
    api_key = request.headers.get(settings.api_key_header, "")
    user_id = getattr(request.state, "user_id", "")

    endpoint_key = f"{request.method}:{request.url.path}"
    base_limit = settings.rate_limit_endpoint_limits.get(
        endpoint_key,
        settings.rate_limit_requests_per_minute,
    )
    burst_limit = settings.rate_limit_endpoint_burst.get(
        endpoint_key,
        settings.rate_limit_burst_size,
    )
    effective_limit = base_limit + burst_limit

    if user_id:
        key = f"user:{user_id}:{endpoint_key}"
    elif api_key:
        key = f"key:{api_key[:20]}:{endpoint_key}"
    else:
        key = f"ip:{client_ip}:{endpoint_key}"

    if not await rate_limiter.check_rate_limit(key, limit=effective_limit):
        remaining = await rate_limiter.get_remaining(key, limit=effective_limit)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(effective_limit),
                "X-RateLimit-Remaining": str(remaining),
                "Retry-After": "60",
            },
        )


# Dependency
RateLimited = Annotated[None, Depends(check_rate_limit)]
