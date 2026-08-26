"""CrewAI integration: a ``cached_tool`` decorator mirroring the LangGraph
surface.

CrewAI itself is an optional dependency. The core package imports this module
without CrewAI installed: CrewAI is only ever imported lazily inside
:func:`_require_crewai` (called from the returned wrapper), so a bare
``import memory_reuse`` never touches it.

The decorator reuses the shared caching machinery
(:func:`memory_reuse.integrations.langgraph.cached_tool`) so keying, sync/async
support, scope resolution and :exc:`~memory_reuse.exceptions.ScopeViolationError`
behaviour, TTL expiry, exact-vs-semantic routing, and
error-propagation-without-store are identical by construction. The only
CrewAI-specific addition is a decoration-time guard that rejects the
``exact_only=True`` + ``semantic=True`` conflict rather than silently ignoring
one argument.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from memory_reuse.exceptions import BackendNotAvailableError, ConfigurationError
from memory_reuse.integrations.langgraph import cached_tool as _base_cached_tool

if TYPE_CHECKING:
    from memory_reuse.core import MemoryCache

_SCOPE_TYPE = Literal["global", "user", "session"]


def _require_crewai() -> None:
    """Assert that CrewAI is importable, else raise a clear, named error.

    The CrewAI integration keeps CrewAI optional: this is the single place where
    CrewAI is imported, and it is called lazily from inside the wrapper returned
    by :func:`cached_tool`. On a missing dependency it raises
    :exc:`~memory_reuse.exceptions.BackendNotAvailableError` naming the extra to
    install rather than surfacing a raw ``ImportError`` traceback.

    Raises:
        BackendNotAvailableError: If CrewAI is not installed.
    """
    try:
        import crewai  # noqa: F401,PLC0415  (availability check only)
    except ImportError as exc:
        raise BackendNotAvailableError(
            "The CrewAI integration requires CrewAI. Install it with "
            '`pip install "memory-reuse[crewai]"`.'
        ) from exc


def cached_tool(
    cache: MemoryCache,
    *,
    scope: _SCOPE_TYPE = "global",
    ttl: int = 300,
    semantic: bool = False,
    exact_only: bool = False,
) -> Callable:
    """Decorator for CrewAI tool calls — caches the return value.

    Mirrors :func:`memory_reuse.integrations.langgraph.cached_tool` exactly:
    the cache key is derived from the tool's qualified name and all of its
    arguments, both sync and async callables are supported, scope resolution and
    :exc:`~memory_reuse.exceptions.ScopeViolationError` behaviour, TTL expiry,
    exact-vs-semantic routing, and error-propagation-without-store are shared by
    delegating to the base decorator.

    CrewAI adds one decoration-time guard: supplying both ``exact_only=True`` and
    ``semantic=True`` is contradictory, so it is rejected with a
    :exc:`~memory_reuse.exceptions.ConfigurationError` rather than silently
    ignoring one argument (Req 7.8). CrewAI is imported lazily inside the
    returned wrapper, so a missing dependency surfaces as a clear
    :exc:`~memory_reuse.exceptions.BackendNotAvailableError` naming the extra.

    Args:
        cache: The :class:`~memory_reuse.core.MemoryCache` instance to use.
        scope: Cache scope (``"global"``, ``"user"``, or ``"session"``).
        ttl: Time-to-live in seconds. Defaults to 300 (5 minutes).
        semantic: When ``True``, route through the combined exact-then-semantic
            flow so a reworded but equivalent call can hit the cache.
        exact_only: When ``True``, force exact-only behaviour for this call site
            even when the cache has semantic matching enabled.

    Returns:
        The decorator function.

    Raises:
        ConfigurationError: At decoration time, if both ``exact_only`` and
            ``semantic`` are ``True``.
        ScopeViolationError: At call time, if ``scope`` requires a scope ID that
            cannot be found.
        BackendNotAvailableError: At call time, if CrewAI is not installed.

    Example::

        @cached_tool(cache, scope="global", ttl=600)
        async def fetch_weather(city: str) -> dict:
            return weather_api.get(city)
    """
    if exact_only and semantic:
        raise ConfigurationError(
            "cached_tool: exact_only=True and semantic=True are mutually "
            "exclusive; set at most one of them."
        )

    base_decorator = _base_cached_tool(
        cache,
        scope=scope,
        ttl=ttl,
        semantic=semantic,
        exact_only=exact_only,
    )

    def decorator(func: Callable) -> Callable:
        wrapped = base_decorator(func)

        @functools.wraps(func)
        async def async_wrapper(*args: object, **kwargs: object) -> object:
            _require_crewai()
            return await wrapped(*args, **kwargs)

        return async_wrapper

    return decorator
