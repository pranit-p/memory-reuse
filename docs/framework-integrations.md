# Framework integrations (Strands & CrewAI)

Beyond LangGraph and LiteLLM, memory-reuse ships `cached_tool` decorators for
the **Strands Agents** and **CrewAI** frameworks. Both mirror the LangGraph
[`cached_tool`](usage.md#3-langgraph-framework-tool-caching) surface exactly —
same signature, same keying, same scope resolution, TTL expiry, and
exact-vs-semantic routing. Under the hood each delegates to the shared caching
machinery, so behaviour is identical across frameworks by construction; no cache
logic is re-implemented.

Both decorators share this signature:

```python
cached_tool(cache, *, scope="global", ttl=300, semantic=False, exact_only=False)
```

| Parameter | Default | Meaning |
|---|---|---|
| `scope` | `"global"` | Scope isolation — `"global"`, `"user"`, or `"session"`. |
| `ttl` | `300` | Time-to-live in seconds for a stored result. |
| `semantic` | `False` | Route through the combined exact-then-semantic flow. |
| `exact_only` | `False` | Force exact-only matching even when semantic is enabled. |

A decorated tool returns the stored result without executing its body on a cache
hit, runs and stores the result on a miss, and re-executes once the TTL elapses.
If the tool body raises, the exception propagates and nothing is stored. A scope
that requires a scope ID which cannot be resolved raises `ScopeViolationError`
before the body runs. See [Backends & scopes](backends.md) for scope details.

## Strands

Requires the `strands` extra:

```bash
pip install "memory-reuse[strands]"
```

```python
from memory_reuse import MemoryCache
from memory_reuse.integrations.strands import cached_tool

cache = MemoryCache()

@cached_tool(cache, scope="global", ttl=600)
async def fetch_weather(city: str) -> dict:
    return await weather_api.get(city)
```

## CrewAI

Requires the `crewai` extra:

```bash
pip install "memory-reuse[crewai]"
```

```python
from memory_reuse import MemoryCache
from memory_reuse.integrations.crewai import cached_tool

cache = MemoryCache()

@cached_tool(cache, scope="session", ttl=300)
async def fetch_user_profile(user_id: str) -> dict:
    return await db.get_user(user_id)
```

CrewAI adds one decoration-time guard: `exact_only=True` and `semantic=True` are
contradictory, so supplying both raises `ConfigurationError` rather than silently
ignoring one argument.

!!! warning "Missing-dependency errors"
    Using either integration without its dependency installed raises
    `BackendNotAvailableError` whose message names the extra to install
    (`pip install "memory-reuse[strands]"` or
    `pip install "memory-reuse[crewai]"`) rather than surfacing a raw
    `ImportError`. The core package imports fine without either installed.

!!! tip "Semantic matching"
    Set `semantic=True` (with `semantic_enabled=True` on the cache config) to let
    reworded-but-equivalent calls reuse a stored result. See the
    [Semantic cache](semantic-cache.md) guide.
