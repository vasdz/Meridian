"""Rate limiting dependency."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from meridian.core.config import settings
from meridian.core.exceptions import RateLimitExceededError
from meridian.infrastructure.cache.redis_cache import RedisCache
from meridian.core.logging import get_logger


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

    async def check_rate_limit(self, key: str) -> bool:
        """Check if request is within rate limit."""
        if not settings.rate_limit_enabled:
            return True

        rate_key = f"rate:{key}"

        try:
            current = await self._cache.incr(rate_key)

            if current == 1:
                # First request, set expiration
                await self._cache.expire(rate_key, 60)

            return current <= self.requests_per_minute

        except Exception as e:
            logger.warning("Rate limit check failed", error=str(e))
            return True  # Fail open

    async def get_remaining(self, key: str) -> int:
        """Get remaining requests."""
        rate_key = f"rate:{key}"

        try:
            value = await self._cache.get(rate_key)
            if value is None:
                return self.requests_per_minute
            return max(0, self.requests_per_minute - int(value))
        except Exception:
            return self.requests_per_minute


# Default rate limiter instance
rate_limiter = RateLimiter(
    requests_per_minute=settings.rate_limit_requests_per_minute,
    burst_size=settings.rate_limit_burst_size,
)


async def check_rate_limit(request: Request) -> None:
    """Rate limiting dependency."""
    if not settings.rate_limit_enabled:
        return

    # Get client identifier (IP or API key)
    client_ip = request.client.host if request.client else "unknown"
    api_key = request.headers.get(settings.api_key_header, "")

    key = api_key[:20] if api_key else client_ip

    if not await rate_limiter.check_rate_limit(key):
        remaining = await rate_limiter.get_remaining(key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(settings.rate_limit_requests_per_minute),
                "X-RateLimit-Remaining": str(remaining),
                "Retry-After": "60",
            },
        )


# Dependency
RateLimited = Annotated[None, Depends(check_rate_limit)]

