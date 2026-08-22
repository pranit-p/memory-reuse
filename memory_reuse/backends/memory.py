"""In-memory cache backend with TTL support and LRU eviction."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

from memory_reuse.backends.base import AbstractBackend

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ENTRIES = 10_000


@dataclass
class _Entry:
    """Internal storage wrapper for a cached value with optional expiry."""

    value: bytes
    expires_at: float | None  # Unix timestamp, or None for no expiry


class InMemoryBackend(AbstractBackend):
    """Fully in-memory cache backend — no external dependencies required.

    Features:

    * **TTL** — entries are lazily expired on access.
    * **LRU eviction** — when ``max_entries`` is reached the least-recently-used
      entry is dropped to make room for the new one.
    * **Thread-safe** — an :class:`asyncio.Lock` serialises all mutations.

    Args:
        max_entries: Maximum number of entries to hold before LRU eviction
            kicks in. Defaults to 10 000.

    Example::

        backend = InMemoryBackend(max_entries=500)
        await backend.set("key", b"value", ttl=60)
        data = await backend.get("key")
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self._max_entries = max_entries
        # OrderedDict used as an ordered map: most-recently-used at the end
        self._store: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # AbstractBackend implementation
    # ------------------------------------------------------------------

    async def get(self, key: str) -> bytes | None:
        """Return stored bytes for ``key``, or ``None`` on miss/expiry.

        Args:
            key: Cache key to look up.

        Returns:
            Stored bytes or ``None``.
        """
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._store[key]
                logger.debug("InMemoryBackend: key expired, removed from store")
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL.

        If the store is at capacity, the LRU entry is evicted first.

        Args:
            key: Cache key.
            value: Bytes to store.
            ttl: Time-to-live in seconds. ``None`` means no expiry.
        """
        expires_at: float | None = None
        if ttl is not None:
            expires_at = time.monotonic() + ttl

        async with self._lock:
            if key in self._store:
                # Update in place and move to end
                self._store[key] = _Entry(value=value, expires_at=expires_at)
                self._store.move_to_end(key)
            else:
                if len(self._store) >= self._max_entries:
                    evicted_key, _ = self._store.popitem(last=False)
                    logger.debug("InMemoryBackend: LRU eviction triggered")
                self._store[key] = _Entry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        """Remove the entry for ``key``.

        Args:
            key: Cache key to remove. No-op if the key does not exist.
        """
        async with self._lock:
            self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        """Return ``True`` if ``key`` exists and has not expired.

        Args:
            key: Cache key to check.

        Returns:
            Boolean existence flag.
        """
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            if self._is_expired(entry):
                del self._store[key]
                return False
            return True

    async def flush(self) -> None:
        """Remove all entries from the store."""
        async with self._lock:
            self._store.clear()
        logger.debug("InMemoryBackend: store flushed")

    async def ping(self) -> bool:
        """Always returns ``True`` — the in-memory backend is always available.

        Returns:
            ``True``
        """
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_expired(self, entry: _Entry) -> bool:
        """Check whether an entry has passed its TTL.

        Args:
            entry: The :class:`_Entry` to inspect.

        Returns:
            ``True`` if the entry has an expiry time that is in the past.
        """
        if entry.expires_at is None:
            return False
        return time.monotonic() >= entry.expires_at

    @property
    def size(self) -> int:
        """Current number of entries in the store (including not-yet-evicted expired ones).

        Returns:
            Integer entry count.
        """
        return len(self._store)
