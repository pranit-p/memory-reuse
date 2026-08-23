# memory-reuse

An execution cache layer for AI agents that cuts LLM and tool call costs by
avoiding redundant computation. Drop it into any Python agent or LangGraph
workflow with a single decorator.

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

!!! tip "Exact vs semantic"
    By default `memory-reuse` does **exact-match** caching — identical inputs
    hit the cache. Enabling the optional
    [semantic cache](semantic-cache.md) also serves cached results for
    *similar-but-not-identical* inputs (reworded questions) using embedding
    similarity.

---

## Where to next

- New here? Start with [Install](install.md) and the [Quick start](quickstart.md).
- Want reworded queries to hit the cache? See the [Semantic cache](semantic-cache.md).
- Looking for a specific option? Jump to the [Configuration reference](configuration.md).
- Prefer reading code? Browse the [Examples](examples.md).
