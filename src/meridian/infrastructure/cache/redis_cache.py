"""Redis cache adapter."""

import json
from typing import Any

import redis.asyncio as redis

from meridian.core.config import settings
from meridian.core.logging import get_logger

logger = get_logger(__name__)

# Global Redis connection
_redis: redis.Redis | None = None


async def init_redis() -> None:
    """Initialize Redis connection."""
    global _redis

    if not settings.redis_enabled:
        logger.info("Redis disabled, skipping initialization")
        return

    try:
        _redis = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        # Test connection
        await _redis.ping()
        logger.info("Redis connection initialized", host=settings.redis_host)
    except Exception as e:
        logger.warning("Redis unavailable, caching disabled", error=str(e))
        _redis = None


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis

    if _redis:
        await _redis.close()
        logger.info("Redis connection closed")


async def check_redis_connection() -> bool:
    """Check if Redis connection is healthy."""
    try:
        if _redis is None:
            await init_redis()
        await _redis.ping()
        return True
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        return False


class RedisCache:
    """Redis cache adapter."""

    def __init__(self, prefix: str = "meridian"):
        self.prefix = prefix

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if _redis is None:
            return None

        try:
            full_key = self._make_key(key)
            value = await _redis.get(full_key)

            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None

        except Exception as e:
            logger.warning("Cache get error", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
    ) -> bool:
        """Set value in cache."""
        if _redis is None:
            return False

        try:
            full_key = self._make_key(key)

            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, str):
                value = str(value)

            await _redis.set(full_key, value, ex=ttl)
            return True

        except Exception as e:
            logger.warning("Cache set error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if _redis is None:
            return False

        try:
            full_key = self._make_key(key)
            await _redis.delete(full_key)
            return True
        except Exception as e:
            logger.warning("Cache delete error", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if _redis is None:
            return False

        try:
            full_key = self._make_key(key)
            return bool(await _redis.exists(full_key))
        except Exception:
            return False

    async def incr(self, key: str) -> int:
        """Increment a counter."""
        if _redis is None:
            return 0

        full_key = self._make_key(key)
        return await _redis.incr(full_key)

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on a key."""
        if _redis is None:
            return False

        full_key = self._make_key(key)
        return bool(await _redis.expire(full_key, ttl))

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if _redis is None:
            return 0

        full_pattern = self._make_key(pattern)
        keys = await _redis.keys(full_pattern)

        if keys:
            return await _redis.delete(*keys)
        return 0
