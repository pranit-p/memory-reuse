# Cache statistics

Every cache tracks hit/miss/error counters (unless `enable_stats=False`).

```python
stats = cache.stats
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Hits: {stats.hits}  Misses: {stats.misses}")
print(f"Exact hits: {stats.exact_hits}  Semantic hits: {stats.semantic_hits}")
print(stats.to_dict())
```

`hits` always equals `exact_hits + semantic_hits`, so you can see how many of
your hits came from the faster exact path versus semantic matching.

Reset the counters at any time:

```python
cache.reset_stats()
```

See [`CacheStats`](api/core.md#memory_reuse.stats.CacheStats) for the full
field list.
