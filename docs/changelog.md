# Changelog

The full, authoritative changelog lives in
[`CHANGELOG.md`](https://github.com/pranit-p/memory-reuse/blob/main/CHANGELOG.md)
in the repository. Highlights below.

## 0.2.0 — Phase 2: semantic cache

Backward compatible — all Phase 1 exact-cache and decorator APIs are unchanged,
and every new capability is off by default.

- **Semantic cache** — opt-in embedding-based similarity matching so reworded
  but equivalent requests reuse a cached result instead of triggering a new LLM
  call.
- `MemoryCache.lookup` / `MemoryCache.store` — combined exact-first, then
  semantic flow; an exact hit never computes an embedding.
- Three embedding providers (`local`, `openai`, `litellm`) behind a common
  interface, selected by config.
- In-memory and Redis vector indexes with namespace isolation and expiry safety.
- `extract_answer` — return only the best-matching sentence(s) of a string
  answer on a semantic hit (extractive, no LLM).
- `CacheStats` gains `exact_hits` / `semantic_hits`.
- New extras: `semantic` (API embeddings, no torch) and `semantic-local`
  (local embeddings).

## 0.1.0 — Phase 1: exact caching

First public release: `MemoryCache`, exact and tool caches, in-memory and Redis
backends, multi-scope isolation, LangGraph + LiteLLM integrations, and shipped
type hints.
