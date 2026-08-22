"""LangGraph integration: ``cached_node`` and ``cached_tool`` decorators."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from memory_reuse.exceptions import ScopeViolationError

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
) -> Callable:
    """Decorator for LangGraph nodes — caches the full node output.

    The cache key is derived from the input state (or a subset defined by
    ``key_fields``).  Both sync and async node functions are supported.

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
) -> Callable:
    """Decorator for tool/function calls — caches the return value.

    The cache key is derived from the function's qualified name and all of
    its arguments.  Both sync and async callables are supported.

    Args:
        cache: The :class:`~memory_reuse.core.MemoryCache` instance to use.
        scope: Cache scope.
        ttl: Time-to-live in seconds.  Defaults to 300 (5 minutes).

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
