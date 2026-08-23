"""Exact-match cache for LLM call results."""

from __future__ import annotations

import logging
from typing import Any

from memory_reuse._utils import (
    build_cache_key,
    check_scope,
    deserialize_value,
    serialize_value,
)
from memory_reuse.backends.base import AbstractBackend
from memory_reuse.config import CacheConfig
from memory_reuse.stats import StatsTracker

logger = logging.getLogger(__name__)


class ExactCache:
    """Hash-based cache for exact-match lookups (e.g. LLM responses).

    Entries are keyed by the SHA-256 hash of the input ``key_parts`` list,
    scoped under the configured prefix, scope, and scope-ID.  Values are
    gzip-compressed JSON so large language-model responses are stored
    efficiently.

    This class is intentionally low-level; most users interact with it
    through the higher-level :class:`~memory_reuse.core.MemoryCache` API
    or the LangGraph decorators.

    Args:
        backend: The storage backend to use.
        config: Cache configuration.
        stats: Statistics tracker shared with the parent :class:`MemoryCache`.

    Example::

        cache = ExactCache(backend, config, stats)
        result = await cache.get(["prompt", "v1"], scope="global", scope_id=None)
        if result is None:
            result = call_llm(...)
            await cache.set(["prompt", "v1"], result, scope="global",
                            scope_id=None, ttl=3600)
    """

    def __init__(
        self,
        backend: AbstractBackend,
        config: CacheConfig,
        stats: StatsTracker,
    ) -> None:
        self._backend = backend
        self._config = config
        self._stats = stats

    async def get(
        self,
        key_parts: list,
        scope: str,
        scope_id: str | None,
    ) -> Any | None:
        """Look up a cached value by its key parts.

        Args:
            key_parts: Ordered list of values that together identify the
                cached item.  These are JSON-serialised and hashed.
            scope: Cache scope — ``"global"``, ``"user"``, or ``"session"``.
            scope_id: User ID or session ID.  Required for non-global scopes.

        Returns:
            The cached value, or ``None`` on a cache miss.

        Raises:
            ScopeViolationError: If ``scope`` requires a ``scope_id`` but
                none is provided.
        """
        self._check_scope(scope, scope_id)
        key = self._build_key(key_parts, scope, scope_id)

        try:
            raw = await self._backend.get(key)
        except Exception:
            self._stats.record_error()
            logger.debug("ExactCache: backend error on GET, treating as miss")
            return None

        if raw is None:
            self._stats.record_miss()
            logger.debug("ExactCache: MISS scope=%s", scope)
            return None

        self._stats.record_hit()
        logger.debug("ExactCache: HIT scope=%s", scope)
        return deserialize_value(raw)

    async def set(
        self,
        key_parts: list,
        value: Any,
        scope: str,
        scope_id: str | None,
        ttl: int | None = None,
    ) -> None:
        """Store a value in the cache.

        Args:
            key_parts: Ordered list of values identifying the cached item.
            value: The value to cache.  Must be JSON-serialisable.
            scope: Cache scope.
            scope_id: User or session identifier for non-global scopes.
            ttl: Time-to-live in seconds.  Falls back to
                :attr:`~memory_reuse.config.CacheConfig.default_ttl` when
                ``None``.

        Raises:
            ScopeViolationError: If ``scope`` requires a ``scope_id`` but
                none is provided.
        """
        self._check_scope(scope, scope_id)
        key = self._build_key(key_parts, scope, scope_id)
        effective_ttl = ttl if ttl is not None else self._config.default_ttl

        try:
            raw = serialize_value(value)
            await self._backend.set(key, raw, ttl=effective_ttl)
            logger.debug("ExactCache: SET scope=%s", scope)
        except Exception:
            self._stats.record_error()
            logger.debug("ExactCache: backend error on SET")

    async def invalidate(
        self,
        key_parts: list,
        scope: str,
        scope_id: str | None,
    ) -> None:
        """Remove a specific cache entry.

        Args:
            key_parts: Key parts that identify the entry to remove.
            scope: Cache scope.
            scope_id: User or session identifier for non-global scopes.

        Raises:
            ScopeViolationError: If ``scope`` requires a ``scope_id`` but
                none is provided.
        """
        self._check_scope(scope, scope_id)
        key = self._build_key(key_parts, scope, scope_id)
        try:
            await self._backend.delete(key)
            logger.debug("ExactCache: INVALIDATED scope=%s", scope)
        except Exception:
            self._stats.record_error()
            logger.debug("ExactCache: backend error on DELETE")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_key(self, key_parts: list, scope: str, scope_id: str | None) -> str:
        """Construct and validate the cache key.

        Args:
            key_parts: Parts to hash.
            scope: Scope string.
            scope_id: Scope identifier.

        Returns:
            The final cache key string.

        Raises:
            ValueError: If the resulting key exceeds ``max_key_size``.
        """
        key = build_cache_key(self._config.key_prefix, scope, scope_id, *key_parts)
        if len(key.encode("utf-8")) > self._config.max_key_size:
            raise ValueError(f"Cache key exceeds max_key_size={self._config.max_key_size} bytes")
        return key

    def _check_scope(self, scope: str, scope_id: str | None) -> None:
        """Raise :exc:`ScopeViolationError` when scope requires an ID but none given.

        Delegates to the shared :func:`memory_reuse._utils.check_scope` guard so
        that :exc:`~memory_reuse.exceptions.ScopeViolationError` behaviour is
        identical across every cache layer.

        Args:
            scope: Requested scope.
            scope_id: Provided scope identifier.

        Raises:
            ScopeViolationError: When ``scope`` is ``"user"`` or ``"session"``
                and ``scope_id`` is ``None`` or empty.
        """
        check_scope(scope, scope_id)
