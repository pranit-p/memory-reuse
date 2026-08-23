"""Core MemoryCache class — the main entry point for memory-reuse."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from memory_reuse.backends.base import AbstractBackend
from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.cache.exact import ExactCache
from memory_reuse.cache.tool import ToolCache
from memory_reuse.config import CacheConfig
from memory_reuse.exceptions import BackendNotAvailableError
from memory_reuse.stats import CacheStats, StatsTracker

if TYPE_CHECKING:
    from memory_reuse.cache.semantic import SemanticCache
    from memory_reuse.vector.base import VectorIndex

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
        # Constructed only when semantic caching is enabled; ``None`` otherwise
        # so no embedding/vector dependency is imported for exact-only users.
        self.semantic: SemanticCache | None = self._maybe_build_semantic()

        logger.debug(
            "MemoryCache initialised: backend=%s stats=%s semantic=%s",
            config.backend,
            config.enable_stats,
            self.semantic is not None,
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
    # Combined exact + semantic API
    # ------------------------------------------------------------------

    async def lookup(
        self,
        key_parts: list,
        query_text: str,
        *,
        scope: str,
        scope_id: str | None,
        exact_only: bool = False,
        threshold: float | None = None,
    ) -> Any | None:
        """Look up a cached value, trying the exact cache before the semantic cache.

        The combined flow tries the fastest, cheapest path first: an exact
        hash-match is attempted before any embedding is computed.  Only when the
        exact cache misses — and semantic caching is enabled and not disabled for
        this call — is the query embedded and matched by similarity.

        Args:
            key_parts: Ordered list of values identifying the exact-cache entry.
            query_text: The natural-language query used for semantic matching.
            scope: Cache scope — ``"global"``, ``"user"``, or ``"session"``.
            scope_id: User or session identifier for non-global scopes.
            exact_only: When ``True``, the semantic cache is never consulted,
                forcing Phase 1 exact-only behaviour for this call site (for
                example a tool with side effects).
            threshold: Optional per-call similarity threshold overriding
                :attr:`~memory_reuse.config.CacheConfig.similarity_threshold`.

        Returns:
            The cached value on an exact or semantic hit, or ``None`` on a miss.

        Raises:
            ScopeViolationError: If ``scope`` requires a ``scope_id`` but none
                is provided.
        """
        exact_result = await self.exact.get(key_parts, scope=scope, scope_id=scope_id)
        if exact_result is not None:
            # Req 7.2 / 11.1: an exact hit returns immediately, never embedding.
            return exact_result

        if exact_only or self.semantic is None:
            # Req 7.5 / 9.4: disabled or exact-only behaves exactly like Phase 1.
            return None

        semantic_result = await self.semantic.get(
            query_text, scope=scope, scope_id=scope_id, threshold=threshold
        )
        if semantic_result is not None and self._config.store_exact_on_semantic_hit:
            # Req 7.4: promote the semantic hit to the exact cache so the next
            # identical request takes the faster exact path.
            await self.exact.set(key_parts, semantic_result, scope=scope, scope_id=scope_id)
        return semantic_result

    async def store(
        self,
        key_parts: list,
        query_text: str,
        value: Any,
        *,
        scope: str,
        scope_id: str | None,
        ttl: int | None = None,
        exact_only: bool = False,
    ) -> None:
        """Store a value in the exact cache and, when enabled, the semantic cache.

        The exact-match entry is always written so a subsequent identical request
        hits the faster exact path.  When semantic caching is enabled and not
        disabled for this call, the query's embedding is also stored so reworded
        but equivalent requests can match later.

        Args:
            key_parts: Ordered list of values identifying the exact-cache entry.
            query_text: The natural-language query whose embedding is stored.
            value: The value to cache. Must be JSON-serialisable.
            scope: Cache scope — ``"global"``, ``"user"``, or ``"session"``.
            scope_id: User or session identifier for non-global scopes.
            ttl: Time-to-live in seconds. Falls back to
                :attr:`~memory_reuse.config.CacheConfig.default_ttl` when
                ``None``.
            exact_only: When ``True``, only the exact-match entry is written and
                the semantic cache is left untouched.

        Raises:
            ScopeViolationError: If ``scope`` requires a ``scope_id`` but none
                is provided.
        """
        await self.exact.set(key_parts, value, scope=scope, scope_id=scope_id, ttl=ttl)

        if exact_only or self.semantic is None:
            return

        await self.semantic.set(query_text, value, scope=scope, scope_id=scope_id, ttl=ttl)

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

    def _maybe_build_semantic(self) -> SemanticCache | None:
        """Construct the semantic cache when enabled, else return ``None``.

        When :attr:`CacheConfig.semantic_enabled` is ``False`` this returns
        ``None`` immediately without importing any embedding or vector
        dependency, keeping the exact-only install path lightweight (Req 4).

        When enabled it lazily imports the embeddings factory and the vector
        index matching the configured backend, then wires up a
        :class:`~memory_reuse.cache.semantic.SemanticCache` sharing this
        cache's :class:`StatsTracker`.

        Returns:
            A configured :class:`SemanticCache`, or ``None`` when semantic
            caching is disabled.

        Raises:
            ConfigurationError: If ``semantic_enabled`` is set without a valid
                ``embedding_provider`` (surfaced by the embeddings factory).
            BackendNotAvailableError: For unknown backend identifiers.
        """
        if not self._config.semantic_enabled:
            return None

        # Imports are deferred so exact-only users never load these modules.
        from memory_reuse.cache.semantic import SemanticCache
        from memory_reuse.embeddings import create_embedder

        embedder = create_embedder(self._config)
        index = self._create_vector_index()

        return SemanticCache(
            index=index,
            embedder=embedder,
            config=self._config,
            stats=self._stats_tracker,
        )

    def _create_vector_index(self) -> VectorIndex:
        """Instantiate the vector index matching the configured backend.

        Returns:
            An :class:`~memory_reuse.vector.base.VectorIndex` implementation:
            an in-process index for the ``"memory"`` backend and a Redis-backed
            index for the ``"redis"`` backend.

        Raises:
            BackendNotAvailableError: For unknown backend identifiers, or when
                the Redis backend is selected without a ``redis_url``.
        """
        match self._config.backend:
            case "memory":
                from memory_reuse.vector.memory import InMemoryVectorIndex

                return InMemoryVectorIndex(
                    max_vectors_per_namespace=self._config.max_vectors_per_namespace
                )
            case "redis":
                from memory_reuse.vector.redis import RedisVectorIndex

                if not self._config.redis_url:
                    raise BackendNotAvailableError(
                        "Redis backend requires 'redis_url' in CacheConfig or "
                        "the MEMORY_REUSE_REDIS_URL environment variable."
                    )
                return RedisVectorIndex(url=self._config.redis_url)
            case _:
                raise BackendNotAvailableError(
                    f"Unknown backend '{self._config.backend}'. "
                    "Supported values: 'memory', 'redis'."
                )

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
