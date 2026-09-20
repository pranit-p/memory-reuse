"""Core MemoryCache class — the main entry point for memory-reuse."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from memory_reuse.analytics import AnalyticsSnapshot, AnalyticsTracker, CacheHitEvent
from memory_reuse.backends.base import AbstractBackend
from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.cache.exact import ExactCache
from memory_reuse.cache.tool import ToolCache
from memory_reuse.config import CacheConfig
from memory_reuse.exceptions import BackendNotAvailableError, ConfigurationError
from memory_reuse.execution import SingleFlight
from memory_reuse.optimizer import (
    EffectivenessAnalyzer,
    EffectivenessReport,
    OperationRecord,
    OperationTracker,
)
from memory_reuse.stats import CacheStats, StatsTracker

if TYPE_CHECKING:
    from memory_reuse.cache.semantic import SemanticCache
    from memory_reuse.integrations.langgraph import CachedGraph
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
        # Phase 5: savings analytics layered on the shared stats tracker. It
        # reads hit rate from ``_stats_tracker`` and never mutates its counters,
        # so exact-only / pre-Phase-5 behaviour is unchanged. Disabled with
        # stats so ``enable_stats=False`` zeroes analytics too.
        self._analytics = AnalyticsTracker(
            self._stats_tracker,
            pricing=self._config.pricing,
            enabled=self._config.enable_stats,
        )

        self.exact = ExactCache(self._backend, self._config, self._stats_tracker)
        self.tool = ToolCache(self._backend, self._config, self._stats_tracker)
        # Constructed only when semantic caching is enabled; ``None`` otherwise
        # so no embedding/vector dependency is imported for exact-only users.
        self.semantic: SemanticCache | None = self._maybe_build_semantic()

        # Phase 6: opt-in single-flight coalescing of concurrent misses. Built
        # only when enabled so exact-only / pre-Phase-6 users pay nothing. The
        # distributed lock requires the Redis backend.
        if self._config.distributed_lock and self._config.backend != "redis":
            raise ConfigurationError(
                "distributed_lock=True requires backend='redis'; "
                f"got backend={self._config.backend!r}."
            )
        self._single_flight: SingleFlight | None = (
            SingleFlight(self._backend, distributed=self._config.distributed_lock)
            if self._config.single_flight
            else None
        )

        # Phase 6: effectiveness analyzer. Records per-operation observations
        # (best-effort, bounded) and turns them into advisory recommendations.
        # Enabled alongside stats; it never mutates the stats/analytics counters.
        self._op_tracker = OperationTracker(
            enabled=self._config.enable_stats,
            max_operations=self._config.max_tracked_operations,
        )

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
    # Analytics (Phase 5)
    # ------------------------------------------------------------------

    @property
    def analytics(self) -> AnalyticsSnapshot:
        """Current savings-analytics snapshot (Phase 5).

        Returns an immutable :class:`~memory_reuse.analytics.AnalyticsSnapshot`
        with the current hit rate (read from the same counters as
        :attr:`stats`), plus tokens, cost, and latency saved. On a fresh cache —
        and whenever ``enable_stats`` is ``False`` — the snapshot is fully
        zeroed. Adding this accessor does not change :attr:`stats` or
        :class:`~memory_reuse.stats.CacheStats`.

        Returns:
            An :class:`~memory_reuse.analytics.AnalyticsSnapshot`.

        Example::

            print(cache.analytics.tokens_saved, cache.analytics.cost_saved)
        """
        return self._analytics.snapshot()

    def record_hit_event(self, event: CacheHitEvent) -> None:
        """Record the savings a cache hit avoided (Phase 5).

        Attribution is caller-supplied — the core caches do not know an
        operation's token/cost/latency — so integrations or application code
        call this on a hit to feed the analytics layer. Recording is best-effort
        and never fatal, and is a no-op when ``enable_stats`` is ``False``.

        Args:
            event: The :class:`~memory_reuse.analytics.CacheHitEvent` describing
                the tokens, cost, and/or latency the hit avoided.

        Example::

            cache.record_hit_event(CacheHitEvent(
                tokens_in=1200, tokens_out=400, latency_saved=0.4,
                operation="search_confluence"))
        """
        self._analytics.record_hit_event(event)

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
        serializer: Callable[[Any], Any] | None = None,
        deserializer: Callable[[Any], Any] | None = None,
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
            serializer: Optional per-call serializer override used when promoting
                a semantic hit to the exact cache. Falls back to the configured
                serializer when ``None``. Used by ``wrap_graph``'s automatic
                codec; most callers leave it ``None``.
            deserializer: Optional per-call deserializer override applied to the
                exact-cache read. Falls back to the configured deserializer when
                ``None``.

        Returns:
            The cached value on an exact or semantic hit, or ``None`` on a miss.

        Raises:
            ScopeViolationError: If ``scope`` requires a ``scope_id`` but none
                is provided.
        """
        effective_deser = deserializer if deserializer is not None else self._config.deserializer
        effective_ser = serializer if serializer is not None else self._config.serializer

        exact_result = await self.exact._get(  # noqa: SLF001
            key_parts, scope, scope_id, deserializer=effective_deser
        )
        if exact_result is not None:
            # Req 7.2 / 11.1: an exact hit returns immediately, never embedding.
            return exact_result

        if exact_only or self.semantic is None:
            # Req 7.5 / 9.4: disabled or exact-only behaves exactly like Phase 1.
            return None

        semantic_result = await self.semantic._get(  # noqa: SLF001
            query_text,
            scope,
            scope_id,
            threshold=threshold,
            deserializer=effective_deser,
        )
        if semantic_result is not None and self._config.store_exact_on_semantic_hit:
            # Req 7.4: promote the semantic hit to the exact cache so the next
            # identical request takes the faster exact path.
            await self.exact._set(  # noqa: SLF001
                key_parts, semantic_result, scope, scope_id, serializer=effective_ser
            )
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
        serializer: Callable[[Any], Any] | None = None,
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
            serializer: Optional per-call serializer override. Falls back to the
                configured serializer when ``None``. Used by ``wrap_graph``'s
                automatic codec; most callers leave it ``None``.

        Raises:
            ScopeViolationError: If ``scope`` requires a ``scope_id`` but none
                is provided.
        """
        effective_ser = serializer if serializer is not None else self._config.serializer
        await self.exact._set(  # noqa: SLF001
            key_parts, value, scope, scope_id, ttl=ttl, serializer=effective_ser
        )

        if exact_only or self.semantic is None:
            return

        await self.semantic._set(  # noqa: SLF001
            query_text, value, scope, scope_id, ttl=ttl, serializer=effective_ser
        )

    # ------------------------------------------------------------------
    # Single-flight coalescing (Phase 6)
    # ------------------------------------------------------------------

    async def get_or_compute(
        self,
        key_parts: list,
        compute: Callable[[], Any],
        *,
        scope: str,
        scope_id: str | None,
        ttl: int | None = None,
        query_text: str = "",
        exact_only: bool = False,
    ) -> Any:
        """Look up a value, computing it at most once across concurrent callers.

        On a hit the stored value is returned without running ``compute``. On a
        miss, when ``single_flight`` is enabled, concurrent calls for the same
        key coalesce so ``compute`` runs exactly once and its result is shared
        with every waiter (Phase 6, Req 1); otherwise this is a plain
        miss→compute→store. The computed value is stored via the combined
        :meth:`store` flow (exact, plus semantic when ``query_text`` is supplied
        and semantic is enabled).

        Args:
            key_parts: Ordered list of values identifying the entry.
            compute: A zero-argument callable (sync or async) producing the value
                on a miss.
            scope: Cache scope — ``"global"``, ``"user"``, or ``"session"``.
            scope_id: User or session identifier for non-global scopes.
            ttl: Time-to-live in seconds for the stored result.
            query_text: Optional natural-language query enabling semantic storage
                of the computed value.
            exact_only: When ``True``, never consult or write the semantic cache.

        Returns:
            The cached value on a hit, or the freshly computed value on a miss.

        Raises:
            ScopeViolationError: If ``scope`` requires a ``scope_id`` but none is
                provided.
        """

        async def _compute() -> Any:
            result = compute()
            if inspect.isawaitable(result):
                result = await result
            return result

        async def _load() -> Any:
            return await self.lookup(
                key_parts,
                query_text,
                scope=scope,
                scope_id=scope_id,
                exact_only=exact_only,
            )

        async def _store(value: Any) -> None:
            await self.store(
                key_parts,
                query_text,
                value,
                scope=scope,
                scope_id=scope_id,
                ttl=ttl,
                exact_only=exact_only,
            )

        if self._single_flight is None:
            # No coalescing: plain miss → compute → store, matching Phase 5.
            existing = await _load()
            if existing is not None:
                return existing
            value = await _compute()
            await _store(value)
            return value

        # Single-flight keys on the derived exact-cache key so concurrent callers
        # for the same logical entry coalesce.
        key = self.exact._build_key(key_parts, scope, scope_id)  # noqa: SLF001
        return await self._single_flight.run(key, compute=_compute, load=_load, store=_store)

    # ------------------------------------------------------------------
    # Effectiveness analysis (Phase 6)
    # ------------------------------------------------------------------

    def record_operation(
        self,
        operation: str,
        *,
        hit: bool,
        cost_saved: float | None = None,
        latency_saved: float | None = None,
        input_hash: str | None = None,
        cached: bool = False,
        volatile: bool = False,
    ) -> None:
        """Record a per-operation observation for the effectiveness analyzer (Phase 6).

        Best-effort and never fatal; a no-op when ``enable_stats`` is ``False``.
        The ``cached`` and ``volatile`` flags are caller-supplied signals the
        recommendation rules use — the analyzer cannot infer a tool's caching
        status or data-freshness on its own.

        Args:
            operation: Label identifying the operation (tool/node/graph name).
            hit: ``True`` when served from cache, ``False`` on a miss.
            cost_saved: Optional attributed cost for this observation.
            latency_saved: Optional attributed latency (seconds).
            input_hash: Optional stable fingerprint of the inputs, used to
                estimate the repeat ratio.
            cached: Whether the operation is already configured to be cached.
            volatile: Whether the operation returns rapidly-changing data.
        """
        self._op_tracker.record(
            OperationRecord(
                operation=operation,
                hit=hit,
                cost_saved=cost_saved,
                latency_saved=latency_saved,
                input_hash=input_hash,
                cached=cached,
                volatile=volatile,
            )
        )

    def analyze(self) -> EffectivenessReport:
        """Return per-operation statistics and advisory cache recommendations (Phase 6).

        The recommendations are advisory only — reading them never changes cache
        behaviour. On a cache with no recorded observations this returns an empty
        report rather than raising.

        Returns:
            An :class:`~memory_reuse.optimizer.EffectivenessReport`.
        """
        return EffectivenessAnalyzer(self._op_tracker).analyze()

    # ------------------------------------------------------------------
    # Graph-level cache (Phase 3)
    # ------------------------------------------------------------------

    def wrap_graph(
        self,
        graph: Any,
        *,
        semantic: bool = False,
        similarity_threshold: float | None = None,
        ttl: int | None = None,
        scope: Literal["global", "user", "session"] | None = None,
        key_fields: list[str] | None = None,
        exact_only: bool = False,
        graph_id: str | None = None,
        serialize_messages: str | bool = "auto",
    ) -> CachedGraph:
        """Wrap a compiled LangGraph graph so entire runs can be served from cache.

        On invoke the wrapper checks a graph-level cache keyed on the initial
        input state; on a hit it returns the stored final result without running
        any node, and on a miss it runs the real graph and stores the result.

        LangGraph is imported lazily here; the core package imports fine without
        it (Req 7).

        Args:
            graph: A compiled LangGraph graph exposing ``invoke`` / ``ainvoke``.
            semantic: Enable semantic (similarity) matching for the whole-run
                key. Requires ``semantic_enabled=True`` on the config to take
                effect.
            similarity_threshold: Per-wrapper threshold overriding the config
                default for graph-level semantic lookups.
            ttl: TTL for stored final results; falls back to
                :attr:`~memory_reuse.config.CacheConfig.default_ttl`.
            scope: ``"global"``, ``"user"``, or ``"session"``; falls back to
                :attr:`~memory_reuse.config.CacheConfig.default_scope`.
            key_fields: Subset of the initial input state used to derive the key
                and semantic query text.
            exact_only: Force exact-only matching, never consulting the semantic
                cache regardless of config.
            graph_id: Stable identifier for this graph included in the key so
                different wrapped graphs never collide. Defaults to a value
                derived from the graph object.
            serialize_messages: Controls automatic faithful serialization of the
                wrapped graph's results so a cache **hit** replays real LangChain
                message objects (with ``.content``), identical to a **miss** —
                plug-and-play, no codec wiring. ``"auto"`` (the default) enables
                it when ``langchain-core`` is importable and silently falls back
                to the default JSON serialization when it is not. ``True`` forces
                it on (raising a named error if ``langchain-core`` is absent).
                ``False`` disables it. When a serializer/deserializer is already
                configured on the cache, that explicit choice always wins and
                this option is ignored. The codec applies only to this wrapped
                graph's store/load, never to the cache's exact/tool/semantic
                layers for other callers.

        Returns:
            A :class:`~memory_reuse.integrations.langgraph.CachedGraph` wrapper.

        Raises:
            BackendNotAvailableError: If LangGraph is not installed, naming the
                extra to install.

        Warning:
            Graph-level caching replays a full stored result and is unsuitable
            for runs whose side effects must occur on every invocation. Use
            ``bypass_cache=True`` / ``no_store=True`` per call, or leave such
            graphs unwrapped.
        """
        from memory_reuse.integrations.langgraph import (
            CachedGraph,
            _require_langgraph,
            _resolve_graph_id,
        )

        _require_langgraph()

        resolved_scope = scope if scope is not None else self._config.default_scope
        resolved_graph_id = _resolve_graph_id(graph, graph_id)
        graph_serializer, graph_deserializer = self._resolve_graph_codec(serialize_messages)

        return CachedGraph(
            self,
            graph,
            semantic=semantic,
            similarity_threshold=similarity_threshold,
            ttl=ttl,
            scope=resolved_scope,
            key_fields=key_fields,
            exact_only=exact_only,
            graph_id=resolved_graph_id,
            serializer=graph_serializer,
            deserializer=graph_deserializer,
        )

    def _resolve_graph_codec(
        self, serialize_messages: str | bool
    ) -> tuple[Callable[[Any], Any] | None, Callable[[Any], Any] | None]:
        """Resolve the (serializer, deserializer) a wrapped graph should use (Req 10).

        An explicit serializer configured on the cache always wins (Req 10.6).
        Otherwise ``serialize_messages`` decides: ``False`` → the default (no
        override); ``"auto"`` → the LangChain codec when importable, else the
        default silently (Req 10.3); ``True`` → the LangChain codec, re-raising
        the named error if LangChain is absent so the explicit intent is honoured.
        """
        # Req 10.6: an explicit config codec wins; the graph adds no override.
        if self._config.serializer is not None or self._config.deserializer is not None:
            return None, None

        if serialize_messages is False:
            return None, None

        from memory_reuse.integrations.langchain_serde import langchain_message_codec

        if serialize_messages == "auto":
            try:
                return langchain_message_codec()
            except BackendNotAvailableError:
                # LangChain not installed: silently keep default serialization.
                return None, None
        # serialize_messages is True (or any other truthy explicit value):
        # honour it, surfacing the named error if LangChain is missing.
        return langchain_message_codec()

    async def invalidate_node(
        self,
        node: Callable | str,
        state: Any = None,
        *,
        scope: Literal["global", "user", "session"] = "global",
        scope_id: str | None = None,
        key_fields: list[str] | None = None,
    ) -> None:
        """Invalidate a cached node output for a specific node and input state.

        Reconstructs the same key parts ``cached_node`` uses
        (``[node_qualname, key_data]``) and deletes the matching exact entry via
        :meth:`~memory_reuse.cache.exact.ExactCache.invalidate`. Completes
        without error when no entry exists (Req 13.6).

        Args:
            node: The decorated node callable (its ``__qualname__`` is used) or
                an explicit node identifier string.
            state: The node input state used to reconstruct the key. Uses the
                full state, or the ``key_fields`` subset when provided.
            scope: ``"global"``, ``"user"``, or ``"session"``.
            scope_id: Explicit scope identifier. When omitted for a non-global
                scope it is resolved from ``state`` then the cache context.
            key_fields: Subset of the input state forming the key, matching the
                node's ``cached_node`` configuration.

        Raises:
            ScopeViolationError: If a non-global scope cannot be resolved.
        """
        from memory_reuse.integrations.langgraph import (
            _extract_scope_id,
            _get_context_scope_id,
        )

        node_id = node if isinstance(node, str) else node.__qualname__

        resolved_scope_id = scope_id
        if scope != "global" and not resolved_scope_id:
            resolved_scope_id = _extract_scope_id(scope, {}, state)
        if scope != "global" and not resolved_scope_id:
            resolved_scope_id = _get_context_scope_id(self, scope)

        key_data: Any
        if key_fields is not None and isinstance(state, dict):
            key_data = {f: state.get(f) for f in key_fields}
        elif isinstance(state, dict):
            key_data = state
        else:
            key_data = str(state)

        key_parts = [node_id, key_data]
        await self.exact.invalidate(key_parts, scope=scope, scope_id=resolved_scope_id)

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
            case "agentcore":
                from memory_reuse.backends.agentcore import (
                    AgentCoreBackend,
                    AgentCoreSettings,
                )

                settings = AgentCoreSettings(
                    region=self._config.agentcore_region,
                    memory_id=self._config.agentcore_memory_id,
                )
                # AgentCoreBackend.__init__ guards its own boto3 import and
                # raises BackendNotAvailableError naming the extra when absent.
                return AgentCoreBackend(settings)
            case _:
                raise BackendNotAvailableError(
                    f"Unknown backend '{self._config.backend}'. "
                    "Supported values: 'memory', 'redis', 'agentcore'."
                )
