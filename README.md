# memory-reuse

[![CI](https://github.com/pranit-p/memory-reuse/actions/workflows/test.yml/badge.svg)](https://github.com/pranit-p/memory-reuse/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/memory-reuse.svg)](https://pypi.org/project/memory-reuse/)
[![Python](https://img.shields.io/pypi/pyversions/memory-reuse.svg)](https://pypi.org/project/memory-reuse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An execution cache layer for AI agents that cuts LLM and tool call costs by avoiding redundant computation. Drop it into any Python agent or LangGraph workflow with a single decorator.

- **Framework-agnostic** — LangGraph, LiteLLM, or any plain Python function.
- **Zero required dependencies** — the core runs on the standard library alone.
- **Safe by default** — per-user / per-session scoping prevents cross-user cache leaks.
- **Typed** — ships with `py.typed`, fully type-hinted.

---

## How it works

When your agent calls an LLM or a tool, `memory-reuse` hashes the inputs and
checks the cache first. On a hit it returns the stored result instantly — no
tokens spent, no API call made. On a miss it runs the real call and stores the
result for next time.

```
request ──► hash inputs ──► cache lookup
                              ├── HIT  ──► return cached result (0 cost)
                              └── MISS ──► run LLM/tool ──► store ──► return
```

> **Current scope (v0.1):** exact-match caching — identical inputs hit the
> cache. Semantic caching (similar-but-not-identical inputs) is on the
> [roadmap](#roadmap).

---

## Install

Works with both **pip** and **uv** — pick whichever you use.

**pip**
```bash
pip install memory-reuse
```

**uv**
```bash
uv add memory-reuse
```

### Optional extras

| Extra | What it adds | pip | uv |
|---|---|---|---|
| `redis` | Redis backend support | `pip install memory-reuse[redis]` | `uv add memory-reuse[redis]` |
| `litellm` | LiteLLM cached wrappers | `pip install memory-reuse[litellm]` | `uv add memory-reuse[litellm]` |
| `all` | Everything above | `pip install memory-reuse[all]` | `uv add memory-reuse[all]` |

> **Note:** `uv` is a fast Python package manager. If you don't have it yet:
> `pip install uv` or see [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/)

---

## Quick start

```python
from memory_reuse import MemoryCache, CacheConfig
from memory_reuse.integrations import cached_tool

cache = MemoryCache()                          # in-memory backend, 1-hour TTL

@cached_tool(cache, scope="global", ttl=300)   # cache for 5 minutes
async def search_web(query: str) -> list[str]:
    return await my_search_api(query)          # only called on cache miss
```

---

## Usage patterns

### 1 — Basic exact cache (LLM responses)

```python
from memory_reuse import MemoryCache

cache = MemoryCache()

# Manual get/set
result = await cache.exact.get(["gpt-4", prompt], scope="global", scope_id=None)
if result is None:
    result = await llm.ainvoke(prompt)
    await cache.exact.set(["gpt-4", prompt], result, scope="global",
                          scope_id=None, ttl=3600)
```

### 2 — LangGraph node caching

```python
from memory_reuse.integrations import cached_node

@cached_node(cache, scope="user", key_fields=["messages"])
async def summarise(state: dict) -> dict:
    summary = await llm.ainvoke(state["messages"])
    return {"summary": summary}
```

The decorator reads `user_id` from the state dict automatically, or from
`cache.set_context(user_id=...)`.

### 3 — LangGraph tool caching

```python
from memory_reuse.integrations import cached_tool

@cached_tool(cache, scope="session", ttl=120)
async def fetch_user_profile(user_id: str) -> dict:
    return await db.get_user(user_id)
```

### 4 — LiteLLM (works with OpenAI, Claude, Bedrock, Groq, Ollama, and 100+ more)

```python
from memory_reuse.integrations import cached_litellm_completion, cached_litellm_embedding

# Completion — same prompt + model = cache hit, 0 tokens used
response = await cached_litellm_completion(
    cache,
    model="gpt-4o-mini",                  # swap for any LiteLLM model string
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    ttl=3600,
    scope="global",
)

# Embeddings — deterministic, safe to cache for 24 hours
embeddings = await cached_litellm_embedding(
    cache,
    model="text-embedding-3-small",
    input=["What is machine learning?"],
)
```

---

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
cache = MemoryCache.from_env()
```

---

## Multi-scope support

```python
cache.set_context(user_id="alice", session_id="sess-001")

# User-scoped: alice cannot see bob's cache
await cache.exact.get(["key"], scope="user", scope_id="alice")

# Session-scoped: isolated per conversation
await cache.tool.get("search", args, scope="session", scope_id="sess-001")

# Global: shared across all users — safe for public, stateless data
await cache.exact.get(["key"], scope="global", scope_id=None)
```

Using `scope="user"` without a `user_id` raises `ScopeViolationError` to
prevent accidental cross-user data leaks.

---

## Cache statistics

```python
stats = cache.stats
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Hits: {stats.hits}  Misses: {stats.misses}")
print(stats.to_dict())
```

---

## Examples

Runnable examples live in [`examples/`](examples/):

- `basic_exact_cache.py` — the cache primitives with no framework.
- `langgraph_agent_example.py` — cached nodes and tools in a LangGraph-style flow.
- `langgraph_math_agent.py` — a real ReAct agent with a **calculator** and a
  **web-search** tool, calling an LLM via LiteLLM.

```bash
export API_KEY="your-groq-key"          # example uses Groq via LiteLLM
python examples/langgraph_math_agent.py
```

---

## Roadmap

| Phase | Feature | Status |
|---|---|---|
| 1 | Exact cache (LLM + tool), Redis backend, LangGraph + LiteLLM | ✅ Shipped in v0.1 |
| 2 | Semantic cache (embedding similarity, configurable threshold) | Planned |
| 3 | Graph-level and node-level execution reuse | Planned |
| 4 | Analytics dashboard, more framework integrations | Planned |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
tests, and code-style guidelines.

---

## License

[MIT](LICENSE) © memory-reuse Contributors
