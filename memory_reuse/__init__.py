"""memory-reuse — execution cache layer for AI agents.

Reduces LLM and tool call costs by caching results with exact-match hashing,
TTL support, multi-scope isolation (global / user / session), and first-class
LangGraph integration.

Quick start::

    from memory_reuse import MemoryCache, CacheConfig

    cache = MemoryCache()                 # in-memory backend, 1-hour TTL
    cache.set_context(user_id="alice")

    # LangGraph decorator
    from memory_reuse.integrations import cached_tool

    @cached_tool(cache, scope="user", ttl=300)
    async def search(query: str) -> list[str]:
        ...
"""

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.exceptions import (
    AgentMemoryError,
    BackendConnectionError,
    BackendNotAvailableError,
    InvalidTTLError,
    ScopeViolationError,
)
from memory_reuse.stats import CacheStats

__all__ = [
    "MemoryCache",
    "CacheConfig",
    "CacheStats",
    "AgentMemoryError",
    "BackendConnectionError",
    "BackendNotAvailableError",
    "ScopeViolationError",
    "InvalidTTLError",
]

__version__ = "0.1.0"
