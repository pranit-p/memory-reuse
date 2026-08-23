"""TTL-backed cache for tool/function call results."""

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
from memory_reuse.exceptions import InvalidTTLError
from memory_reuse.stats import StatsTracker

logger = logging.getLogger(__name__)


class ToolCache:
    """Cache for tool/function call results, keyed by tool name and arguments.

    Unlike :class:`~memory_reuse.cache.exact.ExactCache`, every ``set``
    operation **requires** an explicit TTL — tool outputs are typically
    time-sensitive (API responses, database queries) and should not be
    cached indefinitely.

    Args:
        backend: Storage backend.
        config: Cache configuration.
        stats: Shared statistics tracker.

    Example::

        tool_cache = ToolCache(backend, config, stats)
        result = await tool_cache.get("search_web", {"query": "AI news"},
                                       scope="user", scope_id="alice")
        if result is None:
            result = search_web(query="AI news")
            await tool_cache.set("search_web", {"query": "AI news"},
                                  result, scope="user", scope_id="alice", ttl=120)
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
        tool_name: str,
        args: dict,
        scope: str,
        scope_id: str | None,
    ) -> Any | None:
        """Look up a cached tool result.

        Args:
            tool_name: The name of the tool/function (used as part of the key).
            args: The arguments passed to the tool.  Must be JSON-serialisable.
            scope: Cache scope — ``"global"``, ``"user"``, or ``"session"``.
            scope_id: User ID or session ID for non-global scopes.

        Returns:
            The cached return value, or ``None`` on a miss.

        Raises:
            ScopeViolationError: If a non-global scope is requested without a
                ``scope_id``.
        """
        self._check_scope(scope, scope_id)
        key = self._build_key(tool_name, args, scope, scope_id)

        try:
            raw = await self._backend.get(key)
        except Exception:
            self._stats.record_error()
            logger.debug("ToolCache: backend error on GET, treating as miss")
            return None

        if raw is None:
            self._stats.record_miss()
            logger.debug("ToolCache: MISS tool=%s scope=%s", tool_name, scope)
            return None

        self._stats.record_hit()
        logger.debug("ToolCache: HIT tool=%s scope=%s", tool_name, scope)
        return deserialize_value(raw)

    async def set(
        self,
        tool_name: str,
        args: dict,
        value: Any,
        scope: str,
        scope_id: str | None,
        ttl: int,
    ) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool/function.
            args: Arguments the tool was called with.
            value: Return value to cache.  Must be JSON-serialisable.
            scope: Cache scope.
            scope_id: User or session identifier for non-global scopes.
            ttl: Time-to-live in **seconds**.  Must be a positive integer.

        Raises:
            InvalidTTLError: If ``ttl`` is not a positive integer.
            ScopeViolationError: If a non-global scope is requested without a
                ``scope_id``.
        """
        if not isinstance(ttl, int) or ttl <= 0:
            raise InvalidTTLError(f"ToolCache.set requires a positive integer TTL, got {ttl!r}")
        self._check_scope(scope, scope_id)
        key = self._build_key(tool_name, args, scope, scope_id)

        try:
            raw = serialize_value(value)
            await self._backend.set(key, raw, ttl=ttl)
            logger.debug("ToolCache: SET tool=%s scope=%s ttl=%d", tool_name, scope, ttl)
        except Exception:
            self._stats.record_error()
            logger.debug("ToolCache: backend error on SET tool=%s", tool_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_key(
        self,
        tool_name: str,
        args: dict,
        scope: str,
        scope_id: str | None,
    ) -> str:
        """Build a deterministic cache key for a tool call.

        Args:
            tool_name: Tool identifier.
            args: Call arguments.
            scope: Cache scope.
            scope_id: Scope identifier.

        Returns:
            Cache key string.
        """
        key = build_cache_key(self._config.key_prefix, scope, scope_id, "tool", tool_name, args)
        if len(key.encode("utf-8")) > self._config.max_key_size:
            raise ValueError(f"Cache key exceeds max_key_size={self._config.max_key_size} bytes")
        return key

    def _check_scope(self, scope: str, scope_id: str | None) -> None:
        """Guard against missing scope IDs.

        Delegates to the shared :func:`memory_reuse._utils.check_scope` guard so
        that :exc:`~memory_reuse.exceptions.ScopeViolationError` behaviour is
        identical across every cache layer.

        Args:
            scope: Requested scope.
            scope_id: Provided identifier.

        Raises:
            ScopeViolationError: When ``scope`` is non-global and ``scope_id``
                is absent.
        """
        check_scope(scope, scope_id)
