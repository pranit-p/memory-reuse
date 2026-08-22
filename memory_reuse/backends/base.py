"""Abstract base class for cache backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractBackend(ABC):
    """Interface that all cache backends must implement.

    Every method is a coroutine so that network-bound backends (e.g. Redis)
    can be awaited without blocking the event loop, while the in-memory
    backend simply returns immediately.

    Implementors should document their own connection-lifecycle behaviour
    (lazy vs eager connection, reconnect logic, etc.).
    """

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Retrieve the raw bytes stored under ``key``.

        Args:
            key: The cache key to look up.

        Returns:
            The stored bytes, or ``None`` if the key does not exist or has
            expired.
        """

    @abstractmethod
    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        """Store ``value`` under ``key``.

        Args:
            key: The cache key.
            value: Raw bytes to store (typically gzip-compressed JSON).
            ttl: Time-to-live in seconds. ``None`` means the entry never
                expires. A backend may ignore this if it does not support TTL.
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove the entry for ``key`` if it exists.

        Args:
            key: The cache key to remove. A no-op if the key does not exist.
        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check whether a non-expired entry exists for ``key``.

        Args:
            key: The cache key to check.

        Returns:
            ``True`` if the key exists and has not expired, ``False``
            otherwise.
        """

    @abstractmethod
    async def flush(self) -> None:
        """Delete all entries managed by this backend instance.

        Use with caution in production — this removes every cached value.
        """

    @abstractmethod
    async def ping(self) -> bool:
        """Check backend connectivity.

        Returns:
            ``True`` if the backend is reachable and functioning, ``False``
            otherwise. Should not raise.
        """
