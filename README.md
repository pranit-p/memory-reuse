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

When your agent calls an LLM or a tool, `memory-reuse` checks the cache first.
On a hit it returns the stored result instantly — no tokens spent, no API call
made. On a miss it runs the real call and stores the result for next time:

```
request ──► cache lookup
              ├── HIT  ──► return stored result   (0 tokens, ~ms)
              └── MISS ──► run LLM / tool ──► store ──► return
```

The lookup always tries the fastest, cheapest path first. An **exact** hash
match is attempted before anything else; only on an exact miss — and only if the
semantic cache is enabled — is the query embedded and matched by similarity. So
identical repeats never pay for an embedding.

### The three cache types

memory-reuse supports three kinds of lookup. Use one, or combine them.

| Type | How it works | Best for |
|---|---|---|
| **Exact cache** | Hashes the input and looks for an identical match. | Repeated identical LLM prompts. |
| **Tool cache** | Hashes a tool name + its arguments, with a TTL, and looks for an identical match. | API calls, DB queries, search — anything with expiry. |
| **Semantic cache** | Embeds the input and finds the closest previous request by cosine similarity. | Same intent phrased differently ("reworded" questions). |

```
Exact:     "What is order 123 status?"  ==  "What is order 123 status?"
           same string  →  instant hash match

Tool:      fetch_order(order_id="123")  called 5 min ago
           same function + args  →  cached result returned (until TTL expires)

Semantic:  "What is order 123 status?"  ≈  "Where is my order 123?"
           different words, same intent  →  embedding-similarity match
```

> **Exact/tool are on by default; semantic is opt-in.** Enabling the
> [semantic cache](#semantic-cache) is what lets reworded-but-equivalent
> requests reuse a stored answer. See the
> [Architecture guide](https://pranit-p.github.io/memory-reuse/architecture/)
> for the full component and data-flow diagrams.

> **Node-level and graph-level caching are here.** Beyond the three cache types
> above, memory-reuse can skip whole LangGraph nodes or replay an entire agent
> run from cache. See [Graph-level execution cache](#graph-level-execution-cache-wrap_graph).

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
| `semantic` | Semantic cache with **API** embeddings (OpenAI / LiteLLM) — no torch | `pip install memory-reuse[semantic]` | `uv add memory-reuse[semantic]` |
| `semantic-local` | Semantic cache with **local** embeddings (sentence-transformers, pulls in torch) | `pip install memory-reuse[semantic-local]` | `uv add memory-reuse[semantic-local]` |
| `strands` | Strands Agents `cached_tool` integration | `pip install memory-reuse[strands]` | `uv add memory-reuse[strands]` |
| `crewai` | CrewAI `cached_tool` integration | `pip install memory-reuse[crewai]` | `uv add memory-reuse[crewai]` |
| `agentcore` | AWS AgentCore shared backend | `pip install memory-reuse[agentcore]` | `uv add memory-reuse[agentcore]` |
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

### 5 — Strands / CrewAI tool caching

Strands and CrewAI each get a `cached_tool` decorator that mirrors the LangGraph
one — same signature `cached_tool(cache, *, scope="global", ttl=300,
semantic=False, exact_only=False)`, same keying, scope resolution, TTL expiry,
and exact-vs-semantic routing. They delegate to the shared caching machinery, so
behaviour is identical across frameworks.

**Strands** — requires `pip install "memory-reuse[strands]"`:

```python
from memory_reuse.integrations.strands import cached_tool

@cached_tool(cache, scope="global", ttl=600)
async def fetch_weather(city: str) -> dict:
    return await weather_api.get(city)
```

**CrewAI** — requires `pip install "memory-reuse[crewai]"`:

```python
from memory_reuse.integrations.crewai import cached_tool

@cached_tool(cache, scope="session", ttl=300)
async def fetch_user_profile(user_id: str) -> dict:
    return await db.get_user(user_id)
```

CrewAI adds one decoration-time guard: passing both `exact_only=True` and
`semantic=True` is contradictory, so it raises `ConfigurationError` rather than
silently ignoring one argument. Using either integration without its dependency
installed raises `BackendNotAvailableError` naming the extra to install.

---

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
cache = MemoryCache.from_env()
```

### AgentCore backend (AWS)

The AgentCore backend targets the managed AWS AgentCore store, so a value cached
in one AgentCore microVM is served to requests handled by another — solving the
cross-microVM isolation problem where each VM starts with an empty in-memory
cache. Requires `pip install "memory-reuse[agentcore]"`.

Select it in code:

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

Selecting `backend="agentcore"` without the dependency installed raises
`BackendNotAvailableError` naming the extra; selecting it via `from_env()`
without `MEMORY_REUSE_AGENTCORE_REGION` or `MEMORY_REUSE_AGENTCORE_MEMORY_ID`
raises `ConfigurationError` naming the missing setting.

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

## Semantic cache

Exact caching only hits when inputs are *identical*. The semantic cache also
serves a cached result when a new query is **meaningfully similar** to a stored
one — so "What is 128 multiplied by 47?" can reuse the answer to "What is 128
times 47?". This lifts hit rates for natural-language workloads (chatbots, FAQ
agents, docs Q&A) where the same intent is phrased many ways.

### Choosing an embedding provider

The semantic cache turns text into vectors using one of three interchangeable
providers, selected by `embedding_provider`:

| Provider | `embedding_provider` | Runs | Install | Notes |
|---|---|---|---|---|
| **OpenAI** | `"openai"` | OpenAI API | `pip install memory-reuse[semantic]` | Hosted, paid per call. No torch. |
| **LiteLLM** | `"litellm"` | 100+ backends (Bedrock, Cohere, …) | `pip install memory-reuse[semantic]` | Model string picks the backend. No torch. |
| **Local** | `"local"` | Your machine | see below | sentence-transformers; private, no per-call cost; pulls in torch. |

There are just two install commands to remember. The `semantic` extra covers
**both API providers** (it bundles the small `openai` and `litellm` clients,
plus `numpy` as a cosine-similarity speedup) and installs **no torch**:

```bash
pip install "memory-reuse[semantic]"     # OpenAI + LiteLLM embeddings, lightweight
```

**Local embeddings** need `sentence-transformers`, which depends on **PyTorch**.
On a CPU-only machine (no NVIDIA GPU) install the **CPU torch wheel first** to
avoid a ~2 GB GPU/CUDA download — the CPU build is ~200 MB:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "memory-reuse[semantic-local]"
```

With a GPU you can skip the first line and just `pip install
"memory-reuse[semantic-local]"`. Either way, the model weights themselves
(e.g. `all-MiniLM-L6-v2`, ~90 MB) download from Hugging Face on first use and
are then cached on disk for offline reuse.

> **Quieter / fully offline runs.** memory-reuse already silences the Hugging
> Face log chatter and the per-embedding progress bar. Once the model is
> cached, you can additionally skip Hugging Face's cache-validation HTTP checks
> by exporting `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1`) in your own
> process. Do this only in your application — set after the first (downloading)
> run, never inside a shared library.

### Enabling it

Semantic caching is **off by default** — existing exact-only code is unchanged.
Turn it on via `CacheConfig` by setting `semantic_enabled=True` and choosing an
embedding provider (`local`, `openai`, or `litellm`):

```python
from memory_reuse import MemoryCache, CacheConfig

cache = MemoryCache(CacheConfig(
    backend="memory",
    semantic_enabled=True,
    embedding_provider="local",              # local sentence-transformers model
    embedding_model="all-MiniLM-L6-v2",
    similarity_threshold=0.95,               # how close is "close enough"
))

# Exact-first, then semantic. An exact hit never computes an embedding.
result = await cache.lookup(
    ["qa", "What is 128 multiplied by 47?"],
    query_text="What is 128 multiplied by 47?",
    scope="global", scope_id=None,
)
if result is None:
    result = await run_llm(...)              # only on a miss
    await cache.store(
        ["qa", "What is 128 times 47?"],
        query_text="What is 128 times 47?",
        value=result, scope="global", scope_id=None,
    )
```

Use `cache.lookup(...)` / `cache.store(...)` for the combined exact-then-semantic
flow. The exact cache is always tried first, so a semantic embedding is only
computed on an exact miss (no extra cost when an exact hit is available).

### Three ways to set the threshold

The `similarity_threshold` is a float in `[0.0, 1.0]`; a higher value demands a
closer match. You can set it three ways, from lowest to highest precedence:

1. **Config field** — the instance-wide default:

   ```python
   CacheConfig(semantic_enabled=True, embedding_provider="local",
               similarity_threshold=0.92)
   ```

2. **Environment variable** — read by `MemoryCache.from_env()`:

   ```bash
   export MEMORY_REUSE_SEMANTIC_ENABLED=true
   export MEMORY_REUSE_EMBEDDING_PROVIDER=local
   export MEMORY_REUSE_SIMILARITY_THRESHOLD=0.90
   ```

   ```python
   cache = MemoryCache.from_env()
   ```

3. **Per-call override** — passed to a single `lookup`, taking precedence over
   the config/env value for that call only:

   ```python
   await cache.lookup(key_parts, query_text="...", scope="global",
                      scope_id=None, threshold=0.98)
   ```

### Returning just the relevant answer (`extract_answer`)

A semantic hit returns the **whole** stored answer by default. If you asked
"Tell me about Python" and later ask "Who created Python?", the second query
matches the first and returns the entire paragraph — even though only one
sentence answers it.

Set `extract_answer=True` to have the cache return **only the sentence(s) that
best match the new question**:

```python
cache = MemoryCache(CacheConfig(
    semantic_enabled=True,
    embedding_provider="local",
    extract_answer=True,          # narrow the stored answer to the best sentence
    extract_min_similarity=0.5,   # confidence a sentence needs to be picked
))
```

Now "Who created Python?" returns just *"Python is a high-level, interpreted
programming language created by Guido van Rossum and first released in 1991."*
instead of the full paragraph.

How it works and its limits:

- **Purely extractive, no LLM.** It splits the stored answer into sentences,
  embeds each with the same model, and returns the sentence closest to the
  query. It never calls an LLM and never fabricates — it can only return text
  already present in the stored answer.
- **Falls back to the full answer** when no sentence clears
  `extract_min_similarity`, so you never get an empty result.
- **String answers only.** Non-string values (dicts, numbers) and
  single-sentence answers are returned unchanged.
- **Best-effort, not QA.** It returns a whole real sentence, so it can't reshape
  text into a crisp answer the way a model would. It is off by default.

### Choosing where to use semantic matching

Semantic matching compares *meaning*, so it shines for natural-language
questions where the same intent is phrased many ways — chatbots, FAQ agents,
docs Q&A, search. That is exactly where it saves the most.

It's a similarity match, though, so keep it to reads and questions rather than
correctness-critical commands. Two prompts can look close yet mean opposite
things — "cancel order 123" vs "confirm order 123" — so use the exact cache for
anything whose result depends on precise wording, especially actions with side
effects. memory-reuse gives you two simple levers to stay on the safe side:

- **Tune the threshold.** The default (`0.95`) favours precision — matches only
  fire when queries are very close. Raise it if you ever see a wrong match;
  lower it to trade some precision for a higher hit rate.
- **Opt a call out with `exact_only=True`.** For a sensitive call site, skip the
  semantic cache entirely regardless of the global config:

  ```python
  await cache.lookup(key_parts, query_text="...", scope="global",
                     scope_id=None, exact_only=True)   # exact match only
  ```

Used this way — similarity for questions, exact for commands — semantic caching
is both safe and a big hit-rate win.

### Latency and cost tradeoff

Enabling semantic caching adds an **embedding computation on every exact miss**.
That embedding costs time (local model inference or an API round-trip) and, for
hosted providers, money. The win is fewer full LLM calls when reworded queries
match; the cost is the embedding overhead on misses. Enable it when your
workload has many differently-worded but equivalent requests, so the saved LLM
calls outweigh the embedding cost. An exact hit short-circuits before any
embedding, so identical repeats stay as cheap as Phase 1.

---

## Graph-level execution cache (`wrap_graph`)

Wrap a compiled LangGraph graph so an entire run can be served from cache. On a
hit the stored final result is replayed with **zero nodes executed**; on a miss
the real graph runs and its final state is stored.

Requires the `langgraph` extra: `pip install "memory-reuse[langgraph]"`.

```python
from memory_reuse import MemoryCache

cache = MemoryCache()
graph = build_graph().compile()          # your compiled LangGraph graph

cached_graph = cache.wrap_graph(
    graph,
    scope="user",                         # global | user | session
    key_fields=["question"],              # ignore ephemeral state fields
    ttl=3600,
)

# Same signatures as the wrapped graph, plus per-call cache controls.
result = await cached_graph.ainvoke({"question": "How do I reset my password?",
                                     "user_id": "alice"})
result = cached_graph.invoke({"question": "...", "user_id": "alice"})  # sync too
```

**Semantic matching.** Enable `semantic=True` (with `semantic_enabled=True` on
the config) so reworded but equivalent questions reuse a stored run. A
per-wrapper `similarity_threshold` overrides the config default.

```python
cached_graph = cache.wrap_graph(graph, semantic=True, similarity_threshold=0.92)
```

**Per-call controls.**

- `bypass_cache=True` — always run the graph, skip the lookup.
- `no_store=True` — run the graph but do not store the result.

**Node-level invalidation.** Invalidate a single cached node output when its
upstream state is known to have changed. Safe and idempotent when no entry
exists.

```python
await cache.invalidate_node(summarise, {"messages": [...]}, scope="user",
                            scope_id="alice", key_fields=["messages"])
```

> **Side effects.** Graph-level caching replays a full stored result. It is
> **unsuitable for runs whose side effects must occur on every invocation**
> (writes, emails, payments). Use `bypass_cache` / `no_store` for those, or
> leave the graph unwrapped.

See the [graph-level cache guide](https://pranit-p.github.io/memory-reuse/usage/#5-graph-level-execution-cache-wrap_graph)
for the full walkthrough.

---

## Configuration reference

All options live on `CacheConfig`. Every field can also be set from an
environment variable (read by `MemoryCache.from_env()`) where noted.

| Field | Type / accepted values | Default | Env var | Description |
|---|---|---|---|---|
| `backend` | `"memory"` \| `"redis"` \| `"agentcore"` | `"memory"` | `MEMORY_REUSE_BACKEND` | Storage backend. `redis` needs the `[redis]` extra; `agentcore` needs the `[agentcore]` extra. |
| `redis_url` | `str` \| `None` | `None` | `MEMORY_REUSE_REDIS_URL` | Redis connection URL. Required when `backend="redis"`. |
| `agentcore_region` | `str` \| `None` | `None` | `MEMORY_REUSE_AGENTCORE_REGION` | AWS region hosting the AgentCore store. Required when `backend="agentcore"`. |
| `agentcore_memory_id` | `str` \| `None` | `None` | `MEMORY_REUSE_AGENTCORE_MEMORY_ID` | AgentCore memory / store resource id. Required when `backend="agentcore"`. |
| `default_ttl` | `int > 0` \| `None` | `3600` | `MEMORY_REUSE_DEFAULT_TTL` (int or `"none"`) | Default entry TTL in seconds. `None` never expires. |
| `default_scope` | `"global"` \| `"user"` \| `"session"` | `"global"` | `MEMORY_REUSE_DEFAULT_SCOPE` | Scope used when none is passed explicitly. |
| `key_prefix` | `str` | `"memreuse"` | `MEMORY_REUSE_KEY_PREFIX` | Prefix prepended to every cache key. |
| `max_key_size` | `int > 0` | `512` | — | Max cache-key length in bytes. |
| `enable_stats` | `bool` | `True` | `MEMORY_REUSE_ENABLE_STATS` (`true`/`false`) | Track hit/miss/error counters. |
| `semantic_enabled` | `bool` | `False` | `MEMORY_REUSE_SEMANTIC_ENABLED` (`true`/`false`) | Turn on the semantic cache. Requires `embedding_provider`. |
| `similarity_threshold` | `float` in `[0.0, 1.0]` | `0.95` | `MEMORY_REUSE_SIMILARITY_THRESHOLD` | Minimum similarity to count as a match. Higher = stricter. |
| `embedding_provider` | `"openai"` \| `"local"` \| `"litellm"` \| `None` | `None` | `MEMORY_REUSE_EMBEDDING_PROVIDER` | Which embedding backend to use. Required when `semantic_enabled=True`. |
| `embedding_model` | `str` \| `None` | `None` (provider default) | `MEMORY_REUSE_EMBEDDING_MODEL` | Model name passed to the provider. |
| `max_vectors_per_namespace` | `int > 0` | `10000` | — | Per-scope vector cap before LRU eviction. |
| `store_exact_on_semantic_hit` | `bool` | `True` | — | On a semantic hit, also write an exact entry so the next identical request takes the faster exact path. |
| `extract_answer` | `bool` | `False` | — | Return only the best-matching sentence(s) of a string answer on a semantic hit (extractive, no LLM). |
| `extract_min_similarity` | `float` in `[0.0, 1.0]` | `0.5` | — | Confidence a sentence needs before `extract_answer` returns it instead of the full answer. |

Invalid values raise at construction time: an out-of-range `similarity_threshold`
or `extract_min_similarity` raises `ConfigurationError`; a non-positive
`default_ttl` raises `InvalidTTLError`; enabling `semantic_enabled` without an
`embedding_provider` raises `ConfigurationError`.

---

## Cache statistics

```python
stats = cache.stats
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Hits: {stats.hits}  Misses: {stats.misses}")
print(f"Exact hits: {stats.exact_hits}  Semantic hits: {stats.semantic_hits}")
print(stats.to_dict())
```

`hits` always equals `exact_hits + semantic_hits`, so you can see how many of
your hits came from the faster exact path versus semantic matching.

---

## Examples

Runnable examples live in [`examples/`](examples/):

- `basic_exact_cache.py` — the cache primitives with no framework.
- `langgraph_agent_example.py` — cached nodes and tools in a LangGraph-style flow.
- `langgraph_math_agent.py` — a real ReAct agent with a **calculator** and a
  **web-search** tool, calling an LLM via LiteLLM.
- `semantic_cache_demo.py` — a reworded query hitting the **semantic cache**
  via the combined `lookup`/`store` flow (offline, no model download).
- `semantic_agent.py` — a real ReAct agent (**calculator** + **web search**)
  whose LLM calls run through the **semantic cache** with a local embedding
  model, so reworded questions reuse cached answers.
- `framework_tool_caching.py` — the **Strands** and **CrewAI** `cached_tool`
  decorators: store-and-replay round trip plus the CrewAI `exact_only`+`semantic`
  guard (offline, no framework install needed).
- `agentcore_backend.py` — the **AWS AgentCore** shared backend: cross-microVM
  cache sharing, byte round-trip, and TTL/connectivity semantics against an
  in-process fake service (offline).

```bash
export API_KEY="your-groq-key"          # example uses Groq via LiteLLM
python examples/langgraph_math_agent.py
```

---

## Roadmap

| Phase | Feature | Status |
|---|---|---|
| 1 | Exact cache (LLM + tool), Redis backend, LangGraph + LiteLLM | ✅ Shipped in v0.1 |
| 2 | Semantic cache (embedding similarity, threshold control, answer extraction) | ✅ Shipped in v0.2 |
| 3 | Graph-level and node-level execution reuse (`wrap_graph`, node skipping, `invalidate_node`) | ✅ Shipped in v0.3 |
| 4 | Framework integrations (Strands, CrewAI) and the AWS AgentCore shared backend | ✅ Shipped in v0.4 |
| 5 | Analytics dashboard, cost estimation, Prometheus + OpenTelemetry export | Planned |

---

## Documentation

Full documentation — guides plus an auto-generated API reference — is published
at **[pranit-p.github.io/memory-reuse](https://pranit-p.github.io/memory-reuse/)**.

Build and preview it locally with the `docs` extra:

```bash
pip install -e ".[docs]"
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # produce the static site in ./site
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
tests, and code-style guidelines, and [CONTRIBUTORS.md](CONTRIBUTORS.md) for the
list of people who have helped build this project.

---

## License

[MIT](LICENSE) © Pranit Pawar
