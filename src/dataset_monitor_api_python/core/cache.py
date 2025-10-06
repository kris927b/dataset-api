import asyncio
import time
from typing import Any, Dict, Optional
from ..core.config import settings


class Cache:
    """
    A simple in-memory cache with TTL support and asyncio.Lock for thread safety.
    """

    def __init__(self, ttl_seconds: Optional[int] = None):
        self._cache: Dict[str, tuple[Any, float]] = {}  # key -> (value, timestamp)
        self._lock = asyncio.Lock()
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else settings.CACHE_TTL_SECONDS

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache if it exists and hasn't expired."""
        async with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if self._is_expired(timestamp):
                del self._cache[key]
                return None
            return value

    async def set(self, key: str, value: Any) -> None:
        """Store a value in the cache with the current timestamp."""
        async with self._lock:
            self._cache[key] = (value, time.time())

    async def has(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        async with self._lock:
            if key not in self._cache:
                return False
            _, timestamp = self._cache[key]
            if self._is_expired(timestamp):
                del self._cache[key]
                return False
            return True

    async def delete(self, key: str) -> None:
        """Remove a key from the cache."""
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear all entries from the cache."""
        async with self._lock:
            self._cache.clear()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a timestamp has exceeded the TTL."""
        if self._ttl_seconds is None:
            return False
        return (time.time() - timestamp) > self._ttl_seconds

    async def cleanup_expired(self) -> None:
        """Remove all expired entries from the cache."""
        async with self._lock:
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if self._is_expired(timestamp)
            ]
            for key in expired_keys:
                del self._cache[key]


# Global cache instance for analysis results
analysis_cache = Cache()