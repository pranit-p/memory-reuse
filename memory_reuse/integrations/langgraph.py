"""LangGraph integration: ``cached_node`` / ``cached_tool`` decorators and the
graph-level execution cache (``CachedGraph`` / ``wrap_graph``).

LangGraph itself is an optional dependency. The core package imports this module
without LangGraph installed: LangGraph is only ever imported lazily inside
:func:`_require_langgraph` (called from ``wrap_graph``). ``CachedGraph``
duck-types on ``invoke`` / ``ainvoke`` and never references a LangGraph type.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from memory_reuse._utils import serialize_value
from memory_reuse.exceptions import BackendNotAvailableError, ScopeViolationError

if TYPE_CHECKING:
    from memory_reuse.core import MemoryCache

logger = logging.getLogger(__name__)

_SCOPE_TYPE = Literal["global", "user", "session"]


def _extract_scope_id(scope: str, kwargs: dict, state: Any = None) -> str | None:
    """Extract the appropriate ID for the given scope from kwargs or state.

    Looks for ``user_id`` / ``session_id`` in ``kwargs`` first, then in the
    ``state`` dict (if provided).

    Args:
        scope: ``"global"``, ``"user"``, or ``"session"``.
        kwargs: Keyword arguments from the decorated function call.
        state: Optional dict-like state object (e.g. LangGraph state).

    Returns:
        The scope ID string, or ``None`` if not found.
    """
    if scope == "global":
        return None

    key = "user_id" if scope == "user" else "session_id"

    # Check explicit kwargs first
    if key in kwargs:
        return str(kwargs[key])

    # Check LangGraph state dict
    if isinstance(state, dict) and key in state:
        return str(state[key])

    return None


def cached_node(
    cache: MemoryCache,
    *,
    scope: _SCOPE_TYPE = "global",
    ttl: int | None = None,
    key_fields: list[str] | None = None,
    semantic: bool = False,
    exact_only: bool = False,
) -> Callable:
    """Decorator for LangGraph nodes — caches the full node output.

    The cache key is derived from the input state (or a subset defined by
    ``key_fields``).  Both sync and async node functions are supported.

    This is the supported entry point for **node-level output caching** and,
    by extension, **node skip-detection**. The node-level contract is:

    * ``Node_Cache_Key = [func.__qualname__, key_data]`` where ``key_data`` is
      the full state dict, or ``{f: state.get(f) for f in key_fields}`` when
      ``key_fields`` is set.
    * A hit returns the stored output and **skips the node body**; a miss
      executes the body and stores the result under that key.
    * ``semantic=True`` routes through the combined exact-then-semantic
      :meth:`~memory_reuse.core.MemoryCache.lookup` / ``store`` flow;
      ``semantic=False`` (or ``exact_only=True``) uses exact-only matching.
    * Scope resolution (state → context) and
      :exc:`~memory_reuse.exceptions.ScopeViolationError` behaviour are shared
      with the other cache layers.

    Skip-detection within a graph run is the emergent behaviour of decorating
    each node with ``cached_node``: every decorated node performs its own
    lookup on its current input, so only nodes whose inputs changed re-execute.
    Use :meth:`~memory_reuse.core.MemoryCache.invalidate_node` to force a node
    to re-run when its upstream state is known to have changed.

    Args:
        cache: The :class:`~memory_reuse.core.MemoryCache` instance to use.
        scope: Cache scope.  For ``"user"`` or ``"session"`` scope the
            decorated function or the LangGraph state must expose a
            ``user_id`` / ``session_id`` key.
        ttl: Time-to-live in seconds.  Overrides the default TTL from
            :class:`~memory_reuse.config.CacheConfig`.
        key_fields: If provided, only these fields from the input state are
            included when computing the cache key.  Useful to ignore
            ephemeral state fields.
        semantic: When ``True``, route through the combined exact-then-semantic
            flow (:meth:`~memory_reuse.core.MemoryCache.lookup` /
            :meth:`~memory_reuse.core.MemoryCache.store`) so a reworded but
            equivalent state can hit the cache.  The semantic query text is
            derived from the ``key_fields`` subset (or full state) rendered to a
            string.  Requires ``semantic_enabled=True`` on the cache config to
            have any effect; otherwise behaviour is identical to Phase 1.
        exact_only: When ``True``, force Phase 1 exact-only behaviour for this
            call site even when the cache has semantic matching enabled.  Useful
            for nodes whose correctness depends on exact input.

    Returns:
        The decorator function.

    Raises:
        ScopeViolationError: At call time, if ``scope`` requires a scope ID
            that cannot be found.

    Example::

        @cached_node(cache, scope="user", key_fields=["messages"])
        async def summarise(state: dict) -> dict:
            return {"summary": llm.invoke(state["messages"])}
    """

    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            state = args[0] if args else kwargs.get("state", {})

            scope_id = _extract_scope_id(scope, kwargs, state)
            if scope != "global" and not scope_id:
                # Also check cache context
                scope_id = _get_context_scope_id(cache, scope)
            if scope != "global" and not scope_id:
                raise ScopeViolationError(
                    f"cached_node: scope='{scope}' but no {scope}_id found in "
                    "state, kwargs, or MemoryCache context."
                )

            # Build key parts from state
            key_data: Any
            if key_fields is not None:
                key_data = {f: state.get(f) for f in key_fields if isinstance(state, dict)}
            else:
                key_data = state if isinstance(state, dict) else str(state)

            key_parts = [func.__qualname__, key_data]

            if semantic:
                query_text = str(key_data)
                cached = await cache.lookup(
                    key_parts,
                    query_text,
                    scope=scope,
                    scope_id=scope_id,
                    exact_only=exact_only,
                )
            else:
                query_text = ""
                cached = await cache.exact.get(key_parts, scope=scope, scope_id=scope_id)
            if cached is not None:
                logger.debug("cached_node: HIT func=%s scope=%s", func.__qualname__, scope)
                return cached

            logger.debug("cached_node: MISS func=%s scope=%s", func.__qualname__, scope)
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, functools.partial(func, *args, **kwargs)
                )

            if semantic:
                await cache.store(
                    key_parts,
                    query_text,
                    result,
                    scope=scope,
                    scope_id=scope_id,
                    ttl=ttl,
                    exact_only=exact_only,
                )
            else:
                await cache.exact.set(key_parts, result, scope=scope, scope_id=scope_id, ttl=ttl)
            return result

        # For sync functions we still return the async_wrapper so that callers
        # in an async context (LangGraph, pytest-asyncio) can simply ``await``
        # it.  If a truly synchronous call site is needed, the caller should
        # use ``asyncio.run(decorated_func(...))``.
        return async_wrapper

    return decorator


def cached_tool(
    cache: MemoryCache,
    *,
    scope: _SCOPE_TYPE = "global",
    ttl: int = 300,
    semantic: bool = False,
    exact_only: bool = False,
) -> Callable:
    """Decorator for tool/function calls — caches the return value.

    The cache key is derived from the function's qualified name and all of
    its arguments.  Both sync and async callables are supported.

    Args:
        cache: The :class:`~memory_reuse.core.MemoryCache` instance to use.
        scope: Cache scope.
        ttl: Time-to-live in seconds.  Defaults to 300 (5 minutes).
        semantic: When ``True``, route through the combined exact-then-semantic
            flow (:meth:`~memory_reuse.core.MemoryCache.lookup` /
            :meth:`~memory_reuse.core.MemoryCache.store`) so a reworded but
            equivalent call can hit the cache.  The semantic query text is
            derived from the string form of the bound arguments.  Requires
            ``semantic_enabled=True`` on the cache config to have any effect;
            otherwise behaviour is identical to Phase 1.
        exact_only: When ``True``, force Phase 1 exact-only behaviour for this
            call site even when the cache has semantic matching enabled.  Useful
            for tools with side effects or destructive operations.

    Returns:
        The decorator function.

    Raises:
        ScopeViolationError: At call time, if ``scope`` requires a scope ID
            that cannot be found.

    Example::

        @cached_tool(cache, scope="global", ttl=600)
        async def fetch_weather(city: str) -> dict:
            return weather_api.get(city)
    """

    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build a dict of all arguments (positional + keyword)
            sig = inspect.signature(func)
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                args_dict: dict = dict(bound.arguments)
            except TypeError:
                args_dict = {"args": list(args), "kwargs": kwargs}

            scope_id = _extract_scope_id(scope, args_dict)
            if scope != "global" and not scope_id:
                scope_id = _get_context_scope_id(cache, scope)
            if scope != "global" and not scope_id:
                raise ScopeViolationError(
                    f"cached_tool: scope='{scope}' but no {scope}_id found in "
                    "function arguments or MemoryCache context."
                )

            tool_name = func.__qualname__

            if semantic:
                # The exact-cache key mirrors ToolCache's ``[tool_name, args]``
                # shape so exact hits behave identically to the non-semantic
                # path, while the query text is derived from the bound args.
                key_parts = [tool_name, args_dict]
                query_text = str(args_dict)
                cached = await cache.lookup(
                    key_parts,
                    query_text,
                    scope=scope,
                    scope_id=scope_id,
                    exact_only=exact_only,
                )
            else:
                cached = await cache.tool.get(tool_name, args_dict, scope=scope, scope_id=scope_id)
            if cached is not None:
                logger.debug("cached_tool: HIT func=%s scope=%s", tool_name, scope)
                return cached

            logger.debug("cached_tool: MISS func=%s scope=%s", tool_name, scope)
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, functools.partial(func, *args, **kwargs)
                )

            if semantic:
                await cache.store(
                    [tool_name, args_dict],
                    str(args_dict),
                    result,
                    scope=scope,
                    scope_id=scope_id,
                    ttl=ttl,
                    exact_only=exact_only,
                )
            else:
                await cache.tool.set(
                    tool_name, args_dict, result, scope=scope, scope_id=scope_id, ttl=ttl
                )
            return result

        # Always return the async wrapper so the decorator works correctly in
        # async runtimes (LangGraph, pytest-asyncio).  Sync callers should use
        # ``asyncio.run(decorated_func(...))``.
        return async_wrapper

    return decorator


def _get_context_scope_id(cache: MemoryCache, scope: str) -> str | None:
    """Read the scope ID from the MemoryCache context.

    Args:
        cache: The :class:`~memory_reuse.core.MemoryCache` instance.
        scope: ``"user"`` or ``"session"``.

    Returns:
        The scope ID string, or ``None`` if not set.
    """
    if scope == "user":
        return cache._context.get("user_id")
    if scope == "session":
        return cache._context.get("session_id")
    return None


# ----------------------------------------------------------------------------
# Phase 3: Graph-level execution cache
# ----------------------------------------------------------------------------


def _require_langgraph() -> None:
    """Assert that LangGraph is importable, else raise a clear, named error.

    The graph-level cache keeps LangGraph optional: this is the single place
    where LangGraph is imported, and it is called lazily from
    :meth:`memory_reuse.core.MemoryCache.wrap_graph`. On a missing dependency
    it raises :exc:`~memory_reuse.exceptions.BackendNotAvailableError` naming
    the extra to install rather than surfacing a raw ``ImportError`` traceback.

    Raises:
        BackendNotAvailableError: If LangGraph is not installed.
    """
    try:
        import langgraph  # noqa: F401,PLC0415  (availability check only)
    except ImportError as exc:
        raise BackendNotAvailableError(
            "wrap_graph requires LangGraph. Install it with "
            '`pip install "memory-reuse[langgraph]"`.'
        ) from exc


def _resolve_graph_id(graph: Any, graph_id: str | None) -> str:
    """Resolve a stable identifier for a compiled graph (Req 3.4).

    Resolution order: an explicit ``graph_id`` argument, then a stable
    ``name`` attribute on the graph if present, then ``type(graph).__qualname__``.

    Args:
        graph: The compiled graph being wrapped.
        graph_id: Explicit identifier passed to ``wrap_graph``, or ``None``.

    Returns:
        A stable identifier string.
    """
    if graph_id is not None:
        return graph_id
    name = getattr(graph, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(graph).__qualname__


def _derive_graph_key(
    state: Any,
    key_fields: list[str] | None,
    graph_id: str,
) -> tuple[list, str]:
    """Derive the whole-run exact key parts and semantic query text (Req 3).

    Args:
        state: The initial input state passed to ``invoke`` / ``ainvoke``.
        key_fields: Optional subset of state fields forming the key.
        graph_id: Stable identifier for the wrapped graph.

    Returns:
        A ``(key_parts, query_text)`` tuple where ``key_parts`` is
        ``["graph", graph_id, key_data]`` and ``query_text`` is
        ``str(key_data)``.
    """
    key_data: Any
    if key_fields is not None and isinstance(state, dict):
        key_data = {f: state.get(f) for f in key_fields}
    elif isinstance(state, dict):
        key_data = state
    else:
        key_data = str(state)

    key_parts = ["graph", graph_id, key_data]
    query_text = str(key_data)
    return key_parts, query_text


class CachedGraph:
    """Caching wrapper around a compiled LangGraph graph.

    Exposes ``invoke`` and ``ainvoke`` with the same signatures as the wrapped
    graph, transparently applying whole-run caching. On a hit the stored final
    result is returned with zero nodes executed; on a miss the wrapped graph
    runs and its final state is stored.

    Constructed via :meth:`memory_reuse.core.MemoryCache.wrap_graph`; not
    intended to be instantiated directly.

    Args:
        cache: The parent :class:`~memory_reuse.core.MemoryCache`.
        graph: The wrapped compiled graph (duck-typed on ``invoke``/``ainvoke``).
        semantic: Whether semantic (similarity) matching is enabled.
        similarity_threshold: Per-wrapper threshold override for semantic lookups.
        ttl: TTL applied to stored final results.
        scope: ``"global"``, ``"user"``, or ``"session"``.
        key_fields: Subset of the initial input state forming the key.
        exact_only: Force exact-only matching regardless of ``semantic``.
        graph_id: Stable identifier included in the key.
    """

    def __init__(
        self,
        cache: MemoryCache,
        graph: Any,
        *,
        semantic: bool,
        similarity_threshold: float | None,
        ttl: int | None,
        scope: str,
        key_fields: list[str] | None,
        exact_only: bool,
        graph_id: str,
    ) -> None:
        self._cache = cache
        self._graph = graph
        self._semantic = semantic
        self._similarity_threshold = similarity_threshold
        self._ttl = ttl
        self._scope = scope
        self._key_fields = key_fields
        self._exact_only = exact_only
        self._graph_id = graph_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ainvoke(
        self,
        state: Any,
        *args: Any,
        bypass_cache: bool = False,
        no_store: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Async whole-run cache path (Req 2).

        Performs the cache lookup before executing any node; on a hit returns
        the stored final result, on a miss runs the wrapped graph's ``ainvoke``
        and (unless ``no_store``) stores the result.

        Args:
            state: Initial input state.
            *args: Forwarded unchanged to the wrapped graph.
            bypass_cache: When ``True``, skip the lookup and always run.
            no_store: When ``True``, run but do not store the result.
            **kwargs: Forwarded unchanged to the wrapped graph.

        Returns:
            The final result, from cache on a hit or from the run on a miss.
        """
        scope_id = self._resolve_scope_id(state)
        key_parts, query_text = _derive_graph_key(state, self._key_fields, self._graph_id)

        if not bypass_cache:
            cached = await self._cache.lookup(
                key_parts,
                query_text,
                scope=self._scope,
                scope_id=scope_id,
                exact_only=self._exact_only or not self._semantic,
                threshold=self._similarity_threshold,
            )
            if cached is not None:
                logger.debug("CachedGraph: HIT graph=%s scope=%s", self._graph_id, self._scope)
                return cached

        logger.debug("CachedGraph: MISS graph=%s scope=%s", self._graph_id, self._scope)
        result = await self._graph.ainvoke(state, *args, **kwargs)

        if not no_store:
            self._check_serialisable(result)
            await self._cache.store(
                key_parts,
                query_text,
                result,
                scope=self._scope,
                scope_id=scope_id,
                ttl=self._ttl,
                exact_only=self._exact_only or not self._semantic,
            )
        return result

    def invoke(
        self,
        state: Any,
        *args: Any,
        bypass_cache: bool = False,
        no_store: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Synchronous whole-run cache path bridging to the async lookup/store.

        Mirrors :meth:`ainvoke` but runs the async cache primitives on an event
        loop and calls the wrapped graph's synchronous ``invoke`` inside a miss.

        Args:
            state: Initial input state.
            *args: Forwarded unchanged to the wrapped graph.
            bypass_cache: When ``True``, skip the lookup and always run.
            no_store: When ``True``, run but do not store the result.
            **kwargs: Forwarded unchanged to the wrapped graph.

        Returns:
            The final result, from cache on a hit or from the run on a miss.
        """
        scope_id = self._resolve_scope_id(state)
        key_parts, query_text = _derive_graph_key(state, self._key_fields, self._graph_id)

        if not bypass_cache:
            cached = self._run_coro(
                self._cache.lookup(
                    key_parts,
                    query_text,
                    scope=self._scope,
                    scope_id=scope_id,
                    exact_only=self._exact_only or not self._semantic,
                    threshold=self._similarity_threshold,
                )
            )
            if cached is not None:
                logger.debug("CachedGraph: HIT graph=%s scope=%s", self._graph_id, self._scope)
                return cached

        logger.debug("CachedGraph: MISS graph=%s scope=%s", self._graph_id, self._scope)
        result = self._graph.invoke(state, *args, **kwargs)

        if not no_store:
            self._check_serialisable(result)
            self._run_coro(
                self._cache.store(
                    key_parts,
                    query_text,
                    result,
                    scope=self._scope,
                    scope_id=scope_id,
                    ttl=self._ttl,
                    exact_only=self._exact_only or not self._semantic,
                )
            )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_scope_id(self, state: Any) -> str | None:
        """Resolve the scope id from state, then context (Req 5.4, 5.5)."""
        if self._scope == "global":
            return None
        scope_id = _extract_scope_id(self._scope, {}, state)
        if not scope_id:
            scope_id = _get_context_scope_id(self._cache, self._scope)
        if not scope_id:
            raise ScopeViolationError(
                f"wrap_graph: scope='{self._scope}' but no {self._scope}_id found "
                "in state or MemoryCache context."
            )
        return scope_id

    @staticmethod
    def _check_serialisable(final_result: Any) -> None:
        """Validate the final result serialises before storing (Req 8.3).

        Raises a clear :exc:`ValueError` on failure so no partial or corrupt
        entry is written, leaving ``ExactCache``'s best-effort behaviour for
        other callers unchanged.

        Args:
            final_result: The value about to be stored.

        Raises:
            ValueError: If the value cannot be serialised for storage.
        """
        try:
            serialize_value(final_result)
        except Exception as exc:
            raise ValueError(
                f"CachedGraph: final result is not serialisable for caching: {exc}"
            ) from exc

    @staticmethod
    def _run_coro(coro: Any) -> Any:
        """Run a coroutine to completion from a sync context.

        Uses :func:`asyncio.run` when no loop is running, and a
        run-in-executor bridge (a fresh loop on a worker thread) when called
        from within a running loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures  # noqa: PLC0415

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
