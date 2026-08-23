# Architecture

This page explains **what memory-reuse does**, **how it's built**, **where you'd
use it**, and **how the pieces fit together**.

## What problem it solves

AI agents repeat work constantly — the same prompt, the same tool call, the same
question phrased slightly differently. Each repeat costs tokens, money, and
latency. memory-reuse sits between your agent and the expensive call and returns
a stored result whenever it safely can.

```mermaid
flowchart LR
    A[Agent / app] -->|LLM or tool call| B{memory-reuse}
    B -->|HIT| C[Cached result<br/>0 tokens, ~ms]
    B -->|MISS| D[Real LLM / tool]
    D --> E[Store result] --> B
    C --> A
    D --> A
```

## The two-layer lookup

A lookup always tries the **exact cache first** (a hash match — fast and free).
Only on an exact miss, and only if semantic caching is enabled, does it embed
the query and look for a *similar* previous request. An exact hit never computes
an embedding, so identical repeats stay as cheap as a plain cache.

```mermaid
flowchart TD
    Q[lookup query] --> EX{Exact hash<br/>match?}
    EX -->|yes| HIT1[Return cached value]
    EX -->|no| SEM{Semantic<br/>enabled?}
    SEM -->|no| MISS[Return None → miss]
    SEM -->|yes| EMB[Embed query]
    EMB --> SEARCH[Vector search<br/>within scope]
    SEARCH --> TH{Best score ≥<br/>threshold?}
    TH -->|no| MISS
    TH -->|yes| EXOPT{extract_answer?}
    EXOPT -->|no| HIT2[Return stored value]
    EXOPT -->|yes| NARROW[Return best-matching<br/>sentence]
```

## Component architecture

`MemoryCache` is the front door. It wires together a storage backend, the exact
and tool caches, and — when enabled — the semantic cache (an embedding provider
plus a vector index). Everything shares one stats tracker.

```mermaid
flowchart TB
    subgraph Client
        MC[MemoryCache]
    end

    subgraph Caches
        EX[ExactCache]
        TL[ToolCache]
        SM[SemanticCache]
    end

    subgraph Storage
        BK[(Backend:<br/>in-memory / Redis)]
        VI[(VectorIndex:<br/>in-memory / Redis)]
    end

    subgraph Embeddings
        EP[EmbeddingProvider:<br/>local / openai / litellm]
    end

    ST[StatsTracker]

    MC --> EX
    MC --> TL
    MC --> SM
    EX --> BK
    TL --> BK
    SM --> EP
    SM --> VI
    MC -. records .-> ST
    EX -. records .-> ST
    SM -. records .-> ST
```

Each layer is swappable behind an interface:

- **Backend** (`AbstractBackend`) — `InMemoryBackend` or `RedisBackend`.
- **VectorIndex** — `InMemoryVectorIndex` (brute-force cosine) or
  `RedisVectorIndex` (Redis Stack KNN with an in-process fallback).
- **EmbeddingProvider** — `LocalEmbedder`, `OpenAIEmbedder`, or `LiteLLMEmbedder`,
  all lazily importing their heavy dependency only when selected.

## Scope isolation

Every entry lives in a **namespace** derived from its scope, so one user's
cached data is never served to another. The same rule applies to both the exact
and semantic caches.

```mermaid
flowchart LR
    G[scope: global] --> NG[namespace: global]
    U["scope: user, id=alice"] --> NU["namespace: user:alice"]
    S["scope: session, id=s1"] --> NS["namespace: session:s1"]
    NG --- X[(isolated stores)]
    NU --- X
    NS --- X
```

## Where you can use it

memory-reuse is framework-agnostic — anywhere you make repeatable LLM or tool
calls:

| Use case | What to cache | Scope | Semantic? |
|---|---|---|---|
| Chatbot / FAQ agent | LLM answers to user questions | `user` or `session` | Yes — reworded questions repeat |
| RAG / docs Q&A | answers to knowledge queries | `global` | Yes |
| Tool calls (search, weather, DB reads) | tool return values | `global` or `user` | Usually no — depends on exact args |
| Deterministic tools (math, formatting) | computed results | `global` | No — `exact_only` |
| Embedding pipelines | embedding vectors | `global` | No (cache the embedding call itself) |
| Multi-user SaaS agent | per-user answers | `user` | Optional |

Rule of thumb: **semantic caching for natural-language questions**, **exact
caching for commands and anything correctness-critical** (see
[Semantic cache → Choosing where to use it](semantic-cache.md#choosing-where-to-use-semantic-matching)).

## How to use it — at a glance

=== "Decorator (simplest)"
    ```python
    from memory_reuse import MemoryCache
    from memory_reuse.integrations import cached_tool

    cache = MemoryCache()

    @cached_tool(cache, scope="global", ttl=300)
    async def search_web(query: str) -> list[str]:
        return await my_search_api(query)   # only runs on a miss
    ```

=== "Manual exact cache"
    ```python
    result = await cache.exact.get(["gpt-4", prompt], scope="global", scope_id=None)
    if result is None:
        result = await llm.ainvoke(prompt)
        await cache.exact.set(["gpt-4", prompt], result, scope="global",
                              scope_id=None, ttl=3600)
    ```

=== "Combined exact + semantic"
    ```python
    from memory_reuse import MemoryCache, CacheConfig

    cache = MemoryCache(CacheConfig(
        semantic_enabled=True,
        embedding_provider="local",
    ))

    answer = await cache.lookup(key_parts, query_text=question,
                                scope="user", scope_id="alice")
    if answer is None:
        answer = await run_agent(question)
        await cache.store(key_parts, query_text=question, value=answer,
                          scope="user", scope_id="alice")
    ```

For deeper guides see [Quick start](quickstart.md),
[Usage patterns](usage.md), and the [Semantic cache](semantic-cache.md).
