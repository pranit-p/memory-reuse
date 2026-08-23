# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Phase 3: graph-level and node-level execution reuse.

---

## [0.2.0] — 2026-08-23

Phase 2 — semantic cache. Backward compatible: all Phase 1 exact-cache and
decorator APIs are unchanged, and every new capability is off by default.

### Added
- **Phase 2 — semantic cache.** Opt-in embedding-based similarity matching so
  reworded but equivalent requests reuse a cached result instead of triggering
  a new LLM call. Off by default; existing exact-only code is unchanged.
- `SemanticCache` — similarity-based cache backed by a pluggable `VectorIndex`,
  filtering expired entries on read and enforcing scope isolation.
- `MemoryCache.lookup` / `MemoryCache.store` — combined exact-first, then
  semantic flow; an exact hit never computes an embedding.
- `EmbeddingProvider` interface with `OpenAIEmbedder`, `LocalEmbedder`
  (sentence-transformers), and `LiteLLMEmbedder` implementations, plus a
  `create_embedder` factory selecting one from config.
- `VectorIndex` abstraction with `InMemoryVectorIndex` (brute-force cosine,
  LRU eviction, namespace isolation) and `RedisVectorIndex` (Redis Stack KNN
  with an in-process fallback).
- `cosine_similarity` helper normalised to `[0.0, 1.0]`.
- `CacheConfig` fields: `semantic_enabled`, `similarity_threshold` (default
  `0.95`), `embedding_provider`, `embedding_model`, `max_vectors_per_namespace`,
  and `store_exact_on_semantic_hit`, with validation and `from_env` parsing of
  `MEMORY_REUSE_SEMANTIC_ENABLED`, `MEMORY_REUSE_SIMILARITY_THRESHOLD`,
  `MEMORY_REUSE_EMBEDDING_PROVIDER`, and `MEMORY_REUSE_EMBEDDING_MODEL`.
  Also adds `extract_answer` and `extract_min_similarity` (see below).
- `CacheStats` gains `exact_hits` and `semantic_hits` counters; `hits` remains
  their sum and `hit_rate` is unchanged.
- Opt-in answer extraction: `CacheConfig.extract_answer` (off by default) makes
  a semantic hit on a string answer return only the sentence(s) that best match
  the query, using the same embedding model — a purely extractive step with no
  LLM call and no fabrication. `extract_min_similarity` sets how confident the
  best sentence must be before it is returned instead of the full answer.
- Optional extras for semantic caching: `semantic` bundles the API embedding
  path (`numpy`, `openai`, `litellm`) with no torch, and `semantic-local` adds
  `sentence-transformers` for local embeddings. `all` includes `semantic-local`.
  Two commands cover every provider — `[semantic]` for API embeddings,
  `[semantic-local]` for local — and CPU-only users can install a CPU torch
  wheel before pulling in the local provider.
- `semantic=True` (and `exact_only`) parameters on `cached_node`,
  `cached_tool`, and `cached_litellm_completion`.
- New exceptions: `EmbeddingProviderError`, `ProviderMismatchError`, and
  `ConfigurationError`.
- Examples: `examples/semantic_cache_demo.py` (offline reworded-query demo) and
  `examples/semantic_agent.py` (a real ReAct agent with calculator + web search,
  per-user scoping, and semantic caching via a local model).

---

## [0.1.0] — 2026-08-22

First public release. Phase 1 — exact caching.

### Added
- `MemoryCache` — high-level client wiring together backends, caches, and stats.
- `ExactCache` — SHA-256 hash-keyed cache for LLM responses with gzip compression.
- `ToolCache` — TTL-enforced cache for tool/function call results.
- `InMemoryBackend` — zero-dependency in-process backend with LRU eviction and TTL support.
- `RedisBackend` — async Redis backend with connection pooling (optional `[redis]` extra).
- `CacheConfig` — dataclass-based configuration with `from_env()` factory reading
  `MEMORY_REUSE_*` environment variables.
- Multi-scope support: `global`, `user`, `session`.
- `ScopeViolationError` — raised when user-scoped data would be cached without a `user_id`.
- `CacheStats` / `StatsTracker` — hit/miss/error counters with a `hit_rate` property.
- `cached_node` decorator — LangGraph node output caching.
- `cached_tool` decorator — function/tool return-value caching (any framework).
- `cached_litellm_completion` / `cached_litellm_embedding` — cached wrappers for LiteLLM,
  supporting OpenAI, Anthropic, AWS Bedrock, Groq, Ollama, and 100+ providers
  (optional `[litellm]` extra).
- All decorators and wrappers support both sync and async callables.
- Shipped type hints (`py.typed`, PEP 561).
- Examples: basic exact cache, LangGraph agent, and a real LangGraph agent with a
  calculator and web-search tool.

[Unreleased]: https://github.com/pranit-p/memory-reuse/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pranit-p/memory-reuse/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pranit-p/memory-reuse/releases/tag/v0.1.0
