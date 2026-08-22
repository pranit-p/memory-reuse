"""Core MemoryCache class — the main entry point for memory-reuse."""

from __future__ import annotations

import logging
from typing import Any

from memory_reuse.backends.base import AbstractBackend
from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.cache.exact import ExactCache
from memory_reuse.cache.tool import ToolCache
from memory_reuse.config import CacheConfig
from memory_reuse.exceptions import BackendNotAvailableError
from memory_reuse.stats import CacheStats, StatsTracker

logger = logging.getLogger(__name__)


class MemoryCache:
    """High-level cache client for AI agent workloads.

    ``MemoryCache`` is the primary public interface. It wires together a
    storage backend, the exact-match LLM cache, the TTL-backed tool cache,
    and statistics tracking.

    Args:
        config: Cache configuration.  When omitted a default
            :class:`~memory_reuse.config.CacheConfig` is used (in-memory
            backend, 1-hour TTL, global scope).
        **kwargs: Keyword arguments forwarded to :class:`CacheConfig` when
            ``config`` is ``None``.  Allows quick construction::

                cache = MemoryCache(backend="redis",
                                    redis_url=os.environ["REDIS_URL"])

    Example::

        from memory_reuse import MemoryCache, CacheConfig

        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=600))
        cache.set_context(user_id="alice")

        result = await cache.exact.get(["my-prompt"], scope="user",
                                        scope_id="alice")
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = CacheConfig(**kwargs) if kwargs else CacheConfig()
        self._config = config
        self._context: dict[str, str | None] = {
            "user_id": None,
            "session_id": None,
            "tenant_id": None,
        }

        self._backend: AbstractBackend = self._create_backend()
        self._stats_tracker = StatsTracker()

        self.exact = ExactCache(self._backend, self._config, self._stats_tracker)
        self.tool = ToolCache(self._backend, self._config, self._stats_tracker)

        logger.debug(
            "MemoryCache initialised: backend=%s stats=%s",
            config.backend,
            config.enable_stats,
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> MemoryCache:
        """Create a :class:`MemoryCache` from ``MEMORY_REUSE_*`` environment variables.

        See :meth:`~memory_reuse.config.CacheConfig.from_env` for the full
        list of recognised variables.

        Returns:
            A configured :class:`MemoryCache` instance.

        Example::

            import os
            os.environ["MEMORY_REUSE_BACKEND"] = "redis"
            os.environ["MEMORY_REUSE_REDIS_URL"] = "redis://localhost:6379/0"
            cache = MemoryCache.from_env()
        """
        return cls(config=CacheConfig.from_env())

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def set_context(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Set the user/session context for scoped cache keys.

        Context values are used by the LangGraph decorators when no explicit
        scope ID is passed.  They do **not** affect calls that provide their
        own ``scope_id`` argument.

        Args:
            user_id: Identifier for the current user.
            session_id: Identifier for the current session.
            tenant_id: Identifier for the current tenant (future use).

        Example::

            cache.set_context(user_id="alice", session_id="sess-123")
        """
        if user_id is not None:
            self._context["user_id"] = user_id
        if session_id is not None:
            self._context["session_id"] = session_id
        if tenant_id is not None:
            self._context["tenant_id"] = tenant_id

    def clear_context(self) -> None:
        """Reset all context values to ``None``.

        Call this between requests in a shared server context to avoid
        leaking one user's context into another request.
        """
        self._context = {"user_id": None, "session_id": None, "tenant_id": None}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CacheStats:
        """Current cache statistics snapshot.

        Returns:
            A :class:`~memory_reuse.stats.CacheStats` dataclass.

        Example::

            print(cache.stats.hit_rate)
        """
        return self._stats_tracker.get_stats()

    def reset_stats(self) -> None:
        """Reset all hit/miss/error counters to zero."""
        self._stats_tracker.reset()

    # ------------------------------------------------------------------
    # Backend health
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Check whether the backend is reachable.

        Returns:
            ``True`` if the backend responds successfully.
        """
        return await self._backend.ping()

    async def close(self) -> None:
        """Close backend connections gracefully.

        Should be called on application shutdown to prevent resource leaks,
        especially when using the Redis backend.

        Example::

            async with asyncio.timeout(5):
                await cache.close()
        """
        if hasattr(self._backend, "close"):
            await self._backend.close()  # type: ignore[attr-defined]
        logger.debug("MemoryCache: backend connections closed")

    async def flush(self) -> None:
        """Flush **all** cached entries from the backend.

        Warning:
            This irreversibly removes every cached entry.  Use only in
            development or test environments.
        """
        await self._backend.flush()
        logger.debug("MemoryCache: backend flushed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_backend(self) -> AbstractBackend:
        """Instantiate the storage backend from configuration.

        Returns:
            An :class:`~memory_reuse.backends.base.AbstractBackend` instance.

        Raises:
            BackendNotAvailableError: For unknown backend identifiers.
        """
        match self._config.backend:
            case "memory":
                return InMemoryBackend()
            case "redis":
                from memory_reuse.backends.redis import RedisBackend

                if not self._config.redis_url:
                    raise BackendNotAvailableError(
                        "Redis backend requires 'redis_url' in CacheConfig or "
                        "the MEMORY_REUSE_REDIS_URL environment variable."
                    )
                return RedisBackend(url=self._config.redis_url)
            case _:
                raise BackendNotAvailableError(
                    f"Unknown backend '{self._config.backend}'. "
                    "Supported values: 'memory', 'redis'."
                )
