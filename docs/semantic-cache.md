# Semantic cache

Exact caching only hits when inputs are *identical*. The semantic cache also
serves a cached result when a new query is **meaningfully similar** to a stored
one — so "What is 128 multiplied by 47?" can reuse the answer to "What is 128
times 47?". This lifts hit rates for natural-language workloads (chatbots, FAQ
agents, docs Q&A) where the same intent is phrased many ways.

## Choosing an embedding provider

The semantic cache turns text into vectors using one of three interchangeable
providers, selected by `embedding_provider`:

| Provider | `embedding_provider` | Runs | Install | Notes |
|---|---|---|---|---|
| **OpenAI** | `"openai"` | OpenAI API | `pip install "memory-reuse[semantic]"` | Hosted, paid per call. No torch. |
| **LiteLLM** | `"litellm"` | 100+ backends (Bedrock, Cohere, …) | `pip install "memory-reuse[semantic]"` | Model string picks the backend. No torch. |
| **Local** | `"local"` | Your machine | `pip install "memory-reuse[semantic-local]"` | sentence-transformers; private, no per-call cost; pulls in torch. |

See [Install](install.md#local-embeddings-and-pytorch) for the CPU-only torch
note when using the local provider.

## Enabling it

Semantic caching is **off by default** — existing exact-only code is unchanged.
Turn it on via `CacheConfig` by setting `semantic_enabled=True` and choosing an
embedding provider:

```python
from memory_reuse import MemoryCache, CacheConfig

cache = MemoryCache(CacheConfig(
    backend="memory",
    semantic_enabled=True,
    embedding_provider="local",              # or "openai" / "litellm"
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

Use `cache.lookup(...)` / `cache.store(...)` for the combined
exact-then-semantic flow. The exact cache is always tried first, so a semantic
embedding is only computed on an exact miss.

## Three ways to set the threshold

`similarity_threshold` is a float in `[0.0, 1.0]`; higher demands a closer
match. Set it three ways, from lowest to highest precedence:

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
3. **Per-call override** — passed to a single `lookup`, highest precedence:
   ```python
   await cache.lookup(key_parts, query_text="...", scope="global",
                      scope_id=None, threshold=0.98)
   ```

## Returning just the relevant answer (`extract_answer`)

A semantic hit returns the **whole** stored answer by default. If you asked
"Tell me about Python" and later ask "Who created Python?", the second query
matches the first and returns the entire paragraph — even though only one
sentence answers it.

Set `extract_answer=True` to return **only the sentence(s) that best match the
new question**:

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
  query. It never calls an LLM and never fabricates.
- **Falls back to the full answer** when no sentence clears
  `extract_min_similarity`, so you never get an empty result.
- **String answers only.** Non-string values and single-sentence answers are
  returned unchanged.
- **Best-effort, not QA.** It returns a whole real sentence, so it can't reshape
  text into a crisp answer the way a model would. Off by default.

## Choosing where to use semantic matching

Semantic matching compares *meaning*, so it shines for natural-language
questions where the same intent is phrased many ways.

It's a similarity match, though, so keep it to reads and questions rather than
correctness-critical commands. Two prompts can look close yet mean opposite
things — "cancel order 123" vs "confirm order 123" — so use the exact cache for
anything whose result depends on precise wording, especially actions with side
effects. Two levers keep you on the safe side:

- **Tune the threshold.** The default (`0.95`) favours precision. Raise it if
  you ever see a wrong match; lower it for a higher hit rate.
- **Opt a call out with `exact_only=True`:**
  ```python
  await cache.lookup(key_parts, query_text="...", scope="global",
                     scope_id=None, exact_only=True)   # exact match only
  ```

Used this way — similarity for questions, exact for commands — semantic caching
is both safe and a big hit-rate win.

## Latency and cost tradeoff

Enabling semantic caching adds an **embedding computation on every exact miss**.
That costs time (local inference or an API round-trip) and, for hosted
providers, money. The win is fewer full LLM calls when reworded queries match.
Enable it when your workload has many differently-worded but equivalent
requests. An exact hit short-circuits before any embedding, so identical
repeats stay as cheap as exact-only caching.
