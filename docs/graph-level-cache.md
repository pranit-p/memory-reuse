# Graph-level & node-level cache

Beyond caching individual LLM responses and tool calls, memory-reuse can cache
an **entire agent run** or **individual graph nodes**. On a hit the stored
result is replayed with zero work; on a miss the real graph (or node) runs and
its output is stored.

This capability is opt-in and additive — existing exact/semantic/tool caching
and the `cached_node` / `cached_tool` decorators are unchanged. It requires the
`langgraph` extra:

```bash
pip install "memory-reuse[langgraph]"
```

LangGraph stays optional: it is only imported lazily inside `wrap_graph` and the
node paths, so the core package still imports without it.

---

## Graph-level cache (`wrap_graph`)

Wrap a compiled LangGraph graph so an entire run can be served from cache. On a
hit the stored final result is replayed with **zero nodes executed**; on a miss
the real graph runs and its final state is stored.

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

The returned `CachedGraph` mirrors the wrapped graph's `invoke` / `ainvoke`
signatures and forwards any extra `*args` / `**kwargs` unchanged.

### `wrap_graph` options

| Option | Default | Meaning |
|---|---|---|
| `semantic` | `False` | Enable semantic matching for reworded-but-equivalent runs. |
| `similarity_threshold` | config default | Per-wrapper threshold for semantic matches. |
| `ttl` | config default | Time-to-live (seconds) for stored runs. |
| `scope` | config default | Isolation scope: `global`, `user`, or `session`. |
| `key_fields` | `None` | Only these input-state fields determine the cache key. |
| `exact_only` | `False` | Force exact-only lookups/stores (never embed). |
| `graph_id` | derived | Explicit identifier so distinct graphs never collide. |

### How the cache key is derived

The whole-run key is `["graph", graph_id, key_data]`, where:

- `key_data` is the `key_fields` subset of the input state when `key_fields` is
  set (and the state is a dict), otherwise the full dict state, otherwise
  `str(state)` for non-dict states.
- `graph_id` resolves from the explicit `graph_id` argument, then a stable graph
  attribute (e.g. `graph.name`), then `type(graph).__qualname__`.

Distinct graph identifiers never share entries, and only the selected input
fields affect the key — so ephemeral fields (timestamps, request ids) can be
excluded via `key_fields`.

---

## Semantic matching

Enable `semantic=True` (with `semantic_enabled=True` on the config) so reworded
but equivalent questions reuse a stored run. A per-wrapper
`similarity_threshold` overrides the config default; an exact hit always
short-circuits before any embedding is computed.

```python
cached_graph = cache.wrap_graph(graph, semantic=True, similarity_threshold=0.92)
```

With `semantic=False` the wrapper behaves exactly like the exact cache path.

---

## Per-call controls

Both `invoke` and `ainvoke` accept two cache-control keywords (consumed by the
wrapper, never forwarded to the graph):

- `bypass_cache=True` — always run the graph, skip the lookup.
- `no_store=True` — run the graph but do not store the result.

```python
await cached_graph.ainvoke(state, bypass_cache=True)   # force a fresh run
await cached_graph.ainvoke(state, no_store=True)        # run without caching
```

---

## Node-level cache (`cached_node`)

Decorating a graph node with `cached_node` caches that node's full output keyed
on its input state. A hit skips the node body entirely; decorating each node of
a graph yields per-node skip-detection — only nodes without a matching cached
output execute.

```python
from memory_reuse import MemoryCache

cache = MemoryCache()

@cached_node(cache, scope="user", key_fields=["messages"])
async def summarise(state: dict) -> dict:
    ...  # expensive LLM call; skipped on a cache hit
```

The node cache key is `[func.__qualname__, key_data]`, isolated per scope. The
`cached_node` signature (`scope`, `ttl`, `key_fields`, `semantic`, `exact_only`)
is unchanged from earlier releases.

---

## Node-level invalidation

Invalidate a single cached node output when its upstream state is known to have
changed. It is safe and idempotent — invalidating a non-existent entry (or the
same entry twice) completes without error — and scope-isolated, so invalidating
one scope leaves other scopes' entries intact.

```python
await cache.invalidate_node(
    summarise,                      # the node callable, or an explicit id string
    {"messages": [...]},
    scope="user",
    scope_id="alice",
    key_fields=["messages"],
)
```

---

## Side effects

!!! warning "Graph-level caching replays a full stored result"
    It is **unsuitable for runs whose side effects must occur on every
    invocation** (writes, emails, payments). Use `bypass_cache` / `no_store`
    for those calls, or leave the graph unwrapped. Node-level caching has the
    same caveat per node.

---

## API reference

See [Integrations](api/integrations.md) for the full `wrap_graph`,
`CachedGraph`, `cached_node`, and `invalidate_node` signatures.
