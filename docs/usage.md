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
