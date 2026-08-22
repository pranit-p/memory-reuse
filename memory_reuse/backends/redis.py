"""Redis cache backend using redis.asyncio."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, cast

from memory_reuse.backends.base import AbstractBackend
from memory_reuse.exceptions import BackendConnectionError, BackendNotAvailableError

if TYPE_CHECKING:
    # Only imported at runtime when redis is available
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_MAX_CONNECTIONS = 20


class RedisBackend(AbstractBackend):
    """Cache backend that stores data in Redis.

    Requires the optional ``redis`` extra::

        pip install memory-reuse[redis]

    The connection is established lazily on the first operation.
    Connection errors are converted to :exc:`BackendConnectionError` so
    callers do not need to handle redis-specific exceptions.

    Args:
        url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
            Prefer reading this from the ``MEMORY_REUSE_REDIS_URL``
            environment variable rather than hardcoding it.
        max_connections: Maximum size of the underlying connection pool.
            Defaults to 20.

    Example::

        import os
        backend = RedisBackend(url=os.environ["MEMORY_REUSE_REDIS_URL"])
        await backend.set("key", b"value", ttl=300)
        data = await backend.get("key")
    """

    def __init__(self, url: str, max_connections: int = _MAX_CONNECTIONS) -> None:
        self._url = url
        self._max_connections = max_connections
        self._client: aioredis.Redis | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _get_client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        """Return the Redis client, creating it on first call.

        Returns:
            An initialised ``redis.asyncio.Redis`` client.

        Raises:
            BackendNotAvailableError: If the ``redis`` package is not installed.
            BackendConnectionError: If the connection attempt fails.
        """
        if self._client is not None:
            return self._client

        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise BackendNotAvailableError(
                "Redis backend requires 'redis' package. "
                "Install it with: pip install memory-reuse[redis]"
            ) from exc

        try:
            pool = aioredis.ConnectionPool.from_url(
                self._url,
                max_connections=self._max_connections,
                decode_responses=False,
            )
            self._client = aioredis.Redis(connection_pool=pool)
            # Validate connectivity
            await self._client.ping()
        except Exception as exc:
            self._client = None
            # Deliberately not logging self._url to avoid leaking credentials
            logger.error("RedisBackend: failed to connect to Redis server")
            raise BackendConnectionError(
                "Could not connect to Redis. Check your connection URL and network."
            ) from exc

        return self._client

    # ------------------------------------------------------------------
    # AbstractBackend implementation
    # ------------------------------------------------------------------

    async def get(self, key: str) -> bytes | None:
        """Retrieve raw bytes stored under ``key``.

        Args:
            key: Cache key.

        Returns:
            Stored bytes or ``None`` on miss.

        Raises:
            BackendConnectionError: On Redis connectivity failure.
        """
        client = await self._get_client()
        try:
            # The client is created with decode_responses=False, so values are
            # always raw bytes (or None on a miss). Cast to satisfy the type
            # checker, which infers the broader bytes | str | None union.
            value = await client.get(key)
            return cast("bytes | None", value)
        except Exception as exc:
            logger.error("RedisBackend: GET failed for key prefix '%s'", key[:8])
            raise BackendConnectionError("Redis GET operation failed") from exc

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        """Store ``value`` under ``key``.

        Args:
            key: Cache key.
            value: Bytes to store.
            ttl: Time-to-live in seconds. ``None`` means no expiry.

        Raises:
            BackendConnectionError: On Redis connectivity failure.
        """
        client = await self._get_client()
        try:
            if ttl is not None:
                await client.setex(key, ttl, value)
            else:
                await client.set(key, value)
        except Exception as exc:
            logger.error("RedisBackend: SET failed for key prefix '%s'", key[:8])
            raise BackendConnectionError("Redis SET operation failed") from exc

    async def delete(self, key: str) -> None:
        """Remove the entry for ``key``.

        Args:
            key: Cache key to remove.

        Raises:
            BackendConnectionError: On Redis connectivity failure.
        """
        client = await self._get_client()
        try:
            await client.delete(key)
        except Exception as exc:
            logger.error("RedisBackend: DELETE failed for key prefix '%s'", key[:8])
            raise BackendConnectionError("Redis DELETE operation failed") from exc

    async def exists(self, key: str) -> bool:
        """Check whether ``key`` exists in Redis.

        Args:
            key: Cache key.

        Returns:
            ``True`` if the key exists (and has not expired in Redis).

        Raises:
            BackendConnectionError: On Redis connectivity failure.
        """
        client = await self._get_client()
        try:
            return bool(await client.exists(key))
        except Exception as exc:
            logger.error("RedisBackend: EXISTS failed for key prefix '%s'", key[:8])
            raise BackendConnectionError("Redis EXISTS operation failed") from exc

    async def flush(self) -> None:
        """Delete all keys in the current Redis database.

        Warning:
            This calls ``FLUSHDB`` on the connected database. Use with care
            in shared Redis environments.

        Raises:
            BackendConnectionError: On Redis connectivity failure.
        """
        client = await self._get_client()
        try:
            await client.flushdb()
            logger.debug("RedisBackend: database flushed")
        except Exception as exc:
            raise BackendConnectionError("Redis FLUSHDB operation failed") from exc

    async def ping(self) -> bool:
        """Check Redis connectivity.

        Returns:
            ``True`` if Redis responds to PING, ``False`` on any error.
        """
        try:
            client = await self._get_client()
            return await client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        """Close the connection pool gracefully.

        Call this during application shutdown to release Redis connections.
        """
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None
            logger.debug("RedisBackend: connection pool closed")
