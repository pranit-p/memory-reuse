# Backends & scopes

## Backend options

| Backend | Extra required | Persistence | Notes |
|---------|---------------|-------------|-------|
| `memory` | none | in-process only | LRU eviction, TTL support |
| `redis` | `[redis]` | yes | connection pool, lazy connect |
| `agentcore` | `[agentcore]` | yes | managed AWS store, shared across microVMs |

Configure via code or environment variables:

```bash
export MEMORY_REUSE_BACKEND=redis
export MEMORY_REUSE_REDIS_URL=redis://localhost:6379/0
export MEMORY_REUSE_DEFAULT_TTL=600
export MEMORY_REUSE_DEFAULT_SCOPE=user
```

```python
from memory_reuse import MemoryCache

cache = MemoryCache.from_env()
```

## AgentCore backend (AWS)

The `agentcore` backend targets the managed AWS AgentCore store. Because storage
is the AgentCore service rather than process memory, a value cached in one
AgentCore microVM is served to requests handled by another — the cross-microVM
isolation problem where each VM otherwise starts with an empty in-memory cache.
When selected, it backs **both** the exact cache and the tool cache. Values
round-trip as raw bytes, TTLs greater than zero expire lazily on read, and a
missing or zero TTL persists until the entry is deleted or flushed.

Requires the `agentcore` extra:

```bash
pip install "memory-reuse[agentcore]"
```

Select it in code with `CacheConfig`:

```python
from memory_reuse import MemoryCache, CacheConfig

cache = MemoryCache(CacheConfig(
    backend="agentcore",
    agentcore_region="us-east-1",
    agentcore_memory_id="mem-123",
))
```

Or via environment variables read by `MemoryCache.from_env()`:

```bash
export MEMORY_REUSE_BACKEND=agentcore
export MEMORY_REUSE_AGENTCORE_REGION=us-east-1
export MEMORY_REUSE_AGENTCORE_MEMORY_ID=mem-123
```

```python
cache = MemoryCache.from_env()
```

!!! warning "Dependency and settings guards"
    Selecting `backend="agentcore"` without the extra installed raises
    `BackendNotAvailableError` naming `pip install "memory-reuse[agentcore]"`.
    Selecting it via `from_env()` without `MEMORY_REUSE_AGENTCORE_REGION` or
    `MEMORY_REUSE_AGENTCORE_MEMORY_ID` raises `ConfigurationError` naming the
    missing setting, and no cache is constructed. Any other `backend` value
    raises `ConfigurationError` identifying the unrecognized value.

## Multi-scope support

Every cache entry lives in a scope so one user's data is never served to
another:

```python
cache.set_context(user_id="alice", session_id="sess-001")

# User-scoped: alice cannot see bob's cache
await cache.exact.get(["key"], scope="user", scope_id="alice")

# Session-scoped: isolated per conversation
await cache.tool.get("search", args, scope="session", scope_id="sess-001")

# Global: shared across all users — safe for public, stateless data
await cache.exact.get(["key"], scope="global", scope_id=None)
```

!!! warning "Scope safety"
    Using `scope="user"` without a `user_id` raises `ScopeViolationError` to
    prevent accidental cross-user data leaks. The same guard applies to the
    semantic cache.
