# Configuration reference

All options live on [`CacheConfig`](api/core.md#memory_reuse.config.CacheConfig).
Every field can also be set from an environment variable (read by
`MemoryCache.from_env()`) where noted.

| Field | Type / accepted values | Default | Env var | Description |
|---|---|---|---|---|
| `backend` | `"memory"` \| `"redis"` | `"memory"` | `MEMORY_REUSE_BACKEND` | Storage backend. `redis` needs the `[redis]` extra. |
| `redis_url` | `str` \| `None` | `None` | `MEMORY_REUSE_REDIS_URL` | Redis connection URL. Required when `backend="redis"`. |
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

## Validation

Invalid values raise at construction time:

- an out-of-range `similarity_threshold` or `extract_min_similarity` raises
  `ConfigurationError`;
- a non-positive `default_ttl` raises `InvalidTTLError`;
- enabling `semantic_enabled` without an `embedding_provider` raises
  `ConfigurationError`.

## From environment

```python
from memory_reuse import MemoryCache, CacheConfig

# Build a config purely from MEMORY_REUSE_* variables:
config = CacheConfig.from_env()
cache = MemoryCache(config)

# Or in one step:
cache = MemoryCache.from_env()
```
