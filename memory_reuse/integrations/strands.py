"""Strands Agents integration: a ``cached_tool`` decorator mirroring the
LangGraph one.

Strands is an optional dependency. The core package imports this module without
Strands installed: Strands is only ever imported lazily inside
:func:`_require_strands`, which is called from within the returned wrapper on
first invocation (not at import and not at decoration time).

The decorator does not re-implement any cache logic. After the dependency guard
it delegates to :func:`memory_reuse.integrations.langgraph.cached_tool`, so
keying, sync/async support, scope resolution (with
:exc:`~memory_reuse.exceptions.ScopeViolationError`), TTL expiry, exact-vs-
semantic routing, and error-propagation-without-store are identical by
construction (Req 6.1-6.12, 6.12).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from memory_reuse.exceptions import BackendNotAvailableError
from memory_reuse.integrations.langgraph import cached_tool as _langgraph_cached_tool

if TYPE_CHECKING:
    from memory_reuse.core import MemoryCache

_SCOPE_TYPE = Literal["global", "user", "session"]


def _require_strands() -> None:
    """Assert that Strands is importable, else raise a clear, named error.

    The Strands integration keeps Strands optional: this is the single place
    where Strands is imported, and it is called lazily from within the wrapper
    returned by :func:`cached_tool` (on first invocation, not at import). On a
    missing dependency it raises
    :exc:`~memory_reuse.exceptions.BackendNotAvailableError` naming the extra to
    install rather than surfacing a raw ``ImportError`` traceback.

    Raises:
        BackendNotAvailableError: If Strands is not installed.
    """
    try:
        import strands  # noqa: F401,PLC0415  (availability check only)
    except ImportError as exc:
        raise BackendNotAvailableError(
            "The Strands integration requires Strands. Install it with "
            '`pip install "memory-reuse[strands]"`.'
        ) from exc


def cached_tool(
    cache: MemoryCache,
    *,
    scope: _SCOPE_TYPE = "global",
    ttl: int = 300,
    semantic: bool = False,
    exact_only: bool = False,
) -> Callable:
    """Decorator for Strands tool calls — caches the return value.

    Thin parity wrapper over
    :func:`memory_reuse.integrations.langgraph.cached_tool`. After the Strands
    dependency guard runs (on first invocation), the call delegates to the
    shared decorator machinery, so the cache key
    (``[tool_qualname, bound_args]``), sync/async support, scope resolution with
    :exc:`~memory_reuse.exceptions.ScopeViolationError`, TTL expiry, exact-vs-
    semantic routing, and error propagation without storing on a raised body are
    all identical to the LangGraph decorator by construction (Req 6.1-6.12).

    Args:
        cache: The :class:`~memory_reuse.core.MemoryCache` instance to use.
        scope: Cache scope — ``"global"``, ``"user"``, or ``"session"``.
        ttl: Time-to-live in seconds. Defaults to 300 (5 minutes).
        semantic: When ``True``, route through the combined exact-then-semantic
            ``lookup`` / ``store`` flow on the cache.
        exact_only: When ``True``, force exact-only matching even when the cache
            has semantic matching enabled.

    Returns:
        The decorator function.

    Raises:
        BackendNotAvailableError: On first invocation, if Strands is not
            installed. The message names ``pip install "memory-reuse[strands]"``.
        ScopeViolationError: At call time, if ``scope`` requires a scope ID that
            cannot be resolved.

    Example::

        @cached_tool(cache, scope="global", ttl=600)
        async def fetch_weather(city: str) -> dict:
            return weather_api.get(city)
    """
    base = _langgraph_cached_tool(
        cache,
        scope=scope,
        ttl=ttl,
        semantic=semantic,
        exact_only=exact_only,
    )

    def decorator(func: Callable) -> Callable:
        delegated = base(func)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _require_strands()
            return await delegated(*args, **kwargs)

        return async_wrapper

    return decorator
