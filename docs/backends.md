# Backends & scopes

## Backend options

| Backend | Extra required | Persistence | Notes |
|---------|---------------|-------------|-------|
| `memory` | none | in-process only | LRU eviction, TTL support |
| `redis` | `[redis]` | yes | connection pool, lazy connect |

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
