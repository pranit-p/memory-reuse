# Usage patterns

## 1 — Basic exact cache (LLM responses)

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

## 2 — LangGraph node caching

```python
from memory_reuse.integrations import cached_node

@cached_node(cache, scope="user", key_fields=["messages"])
async def summarise(state: dict) -> dict:
    summary = await llm.ainvoke(state["messages"])
    return {"summary": summary}
```

The decorator reads `user_id` from the state dict automatically, or from
`cache.set_context(user_id=...)`.

## 3 — LangGraph / framework tool caching

```python
from memory_reuse.integrations import cached_tool

@cached_tool(cache, scope="session", ttl=120)
async def fetch_user_profile(user_id: str) -> dict:
    return await db.get_user(user_id)
```

## 4 — LiteLLM (OpenAI, Claude, Bedrock, Groq, Ollama, and 100+ more)

```python
from memory_reuse.integrations import (
    cached_litellm_completion,
    cached_litellm_embedding,
)

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

!!! tip "Semantic on decorators"
    `cached_node`, `cached_tool`, and `cached_litellm_completion` all accept
    `semantic=True` (and `exact_only=True`) to route through the combined
    exact-then-semantic flow. See the [Semantic cache](semantic-cache.md) guide.

## 5 — Strands & CrewAI tool caching

The Strands and CrewAI frameworks each get a `cached_tool` decorator with the
same signature and behaviour as the LangGraph one above:

```python
# Strands — pip install "memory-reuse[strands]"
from memory_reuse.integrations.strands import cached_tool

@cached_tool(cache, scope="global", ttl=600)
async def fetch_weather(city: str) -> dict:
    return await weather_api.get(city)

# CrewAI — pip install "memory-reuse[crewai]"
from memory_reuse.integrations.crewai import cached_tool

@cached_tool(cache, scope="session", ttl=300)
async def fetch_user_profile(user_id: str) -> dict:
    return await db.get_user(user_id)
```

See the [Framework integrations](framework-integrations.md) guide for the full
walkthrough, including the CrewAI `exact_only` + `semantic` guard and the
missing-dependency errors.

## 6 — Graph-level & node-level cache (`wrap_graph`)

Wrap a compiled LangGraph graph so an entire run can be served from cache, or
decorate individual nodes so each is skipped on a hit. On a graph-level hit the
stored final result is replayed with **zero nodes executed**.

```python
from memory_reuse import MemoryCache

cache = MemoryCache()
graph = build_graph().compile()
cached_graph = cache.wrap_graph(graph, scope="user", key_fields=["question"])

result = await cached_graph.ainvoke({"question": "...", "user_id": "alice"})
```

Requires the `langgraph` extra (`pip install "memory-reuse[langgraph]"`). See
the dedicated [Graph-level & node-level cache](graph-level-cache.md) guide for
the full walkthrough: `wrap_graph` options, key derivation, semantic matching,
per-call `bypass_cache` / `no_store` controls, `cached_node` skip-detection,
and `invalidate_node`.
