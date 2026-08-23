# Quick start

Wrap any function with a decorator — on a cache hit the real call is skipped.

```python
from memory_reuse import MemoryCache, CacheConfig
from memory_reuse.integrations import cached_tool

cache = MemoryCache()                          # in-memory backend, 1-hour TTL

@cached_tool(cache, scope="global", ttl=300)   # cache for 5 minutes
async def search_web(query: str) -> list[str]:
    return await my_search_api(query)          # only called on cache miss
```

That's it. The first call to `search_web("...")` runs the real function; an
identical call within the TTL returns the stored result instantly.

## Manual get/set

You can also use the cache primitives directly, without a decorator:

```python
from memory_reuse import MemoryCache

cache = MemoryCache()

result = await cache.exact.get(["gpt-4", prompt], scope="global", scope_id=None)
if result is None:
    result = await llm.ainvoke(prompt)
    await cache.exact.set(
        ["gpt-4", prompt], result, scope="global", scope_id=None, ttl=3600
    )
```

## Next steps

- [Usage patterns](usage.md) — LangGraph nodes, tools, and LiteLLM wrappers.
- [Backends & scopes](backends.md) — Redis, and per-user / per-session isolation.
- [Semantic cache](semantic-cache.md) — let reworded queries hit the cache.
