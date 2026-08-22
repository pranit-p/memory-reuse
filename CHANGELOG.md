# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Phase 2: semantic cache (embedding-based similarity matching).
- Phase 3: graph-level and node-level execution reuse.

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

[Unreleased]: https://github.com/pranit-p/memory-reuse/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pranit-p/memory-reuse/releases/tag/v0.1.0
