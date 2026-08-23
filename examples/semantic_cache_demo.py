"""Example: Semantic cache — a reworded query reuses a cached result.

Exact caching only hits when inputs are *identical*. The semantic cache also
serves a cached result when a new query is *meaningfully similar* to a stored
one, so "What is 128 multiplied by 47?" can reuse the answer stored for
"What is 128 times 47?".

This demo stays fully offline: it injects a tiny deterministic ``FakeEmbedder``
so no model is downloaded and no API is called. To use a real local model
instead, install the extra and configure the ``local`` provider::

    pip install memory-reuse[semantic]

    cache = MemoryCache(CacheConfig(
        backend="memory",
        semantic_enabled=True,
        embedding_provider="local",
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.95,
    ))
    # ...and delete the ``cache.semantic._embedder = ...`` injection below.

Run this example:

    python examples/semantic_cache_demo.py
"""

from __future__ import annotations

import asyncio

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse.embeddings.base import EmbeddingProvider


class FakeEmbedder(EmbeddingProvider):
    """Deterministic, offline embedder mapping known phrasings to one vector.

    Any text listed in ``vectors`` embeds to its shared vector, so different
    phrasings of the same intent become a semantic match. Unknown text embeds
    to a distinct vector, so it never matches by accident.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    @property
    def identity(self) -> str:
        return "fake:demo"

    @property
    def dimension(self) -> int:
        return 3

    async def embed(self, text: str) -> list[float]:
        if text in self._vectors:
            return list(self._vectors[text])
        # Distinct fallback vector so unregistered text never matches.
        return [float(len(text) % 7), float(sum(map(ord, text)) % 11), 1.0]


async def fake_llm_call(prompt: str) -> str:
    """Simulate an expensive LLM API call."""
    print(f"  [LLM] Calling API for: {prompt!r}")
    await asyncio.sleep(0.1)  # Simulate network latency
    return "6016"  # 128 * 47


async def main() -> None:
    # Two different phrasings of the same question that should share an answer.
    original = "What is 128 times 47?"
    reworded = "What is 128 multiplied by 47?"

    # Both phrasings embed to the same vector, so they are a semantic match.
    shared_vector = [1.0, 0.0, 0.0]
    embedder = FakeEmbedder({original: shared_vector, reworded: shared_vector})

    # Enable semantic caching. ``embedding_provider="local"`` satisfies config
    # validation; we then inject the offline FakeEmbedder so nothing is
    # downloaded. In a real app you would rely on the configured provider.
    config = CacheConfig(
        backend="memory",
        semantic_enabled=True,
        embedding_provider="local",
        similarity_threshold=0.95,
    )
    cache = MemoryCache(config)
    assert cache.semantic is not None
    cache.semantic._embedder = embedder  # type: ignore[attr-defined]  # demo-only injection

    # --- First call: cold cache, so the "LLM" runs and we store the result. ---
    print("\n=== First call (cache miss — LLM runs) ===")
    print(f"  Query: {original!r}")
    result = await cache.lookup(
        ["qa", original], query_text=original, scope="global", scope_id=None
    )
    if result is None:
        result = await fake_llm_call(original)
        await cache.store(
            ["qa", original], query_text=original, value=result, scope="global", scope_id=None
        )
    print(f"  Result: {result!r}")

    # --- Second call: a *reworded* query. Exact match misses, semantic hits. ---
    print("\n=== Second call (reworded query — semantic hit, LLM NOT called) ===")
    print(f"  Query: {reworded!r}")
    result = await cache.lookup(
        ["qa", reworded], query_text=reworded, scope="global", scope_id=None
    )
    if result is None:
        result = await fake_llm_call(reworded)
        await cache.store(
            ["qa", reworded], query_text=reworded, value=result, scope="global", scope_id=None
        )
    print(f"  Result: {result!r}")

    # --- exact_only forces Phase 1 behaviour: no semantic match. ---
    print("\n=== exact_only=True (semantic cache skipped) ===")
    unseen = "What is 200 divided by 4?"
    print(f"  Query: {unseen!r}")
    result = await cache.lookup(
        ["qa", unseen], query_text=unseen, scope="global", scope_id=None, exact_only=True
    )
    print(f"  Result: {result!r}  (None — semantic never consulted)")

    print("\n=== Stats ===")
    stats = cache.stats
    print(f"  Exact hits:    {stats.exact_hits}")
    print(f"  Semantic hits: {stats.semantic_hits}")
    print(f"  Misses:        {stats.misses}")
    print(f"  Hit rate:      {stats.hit_rate:.1%}")

    await cache.close()


if __name__ == "__main__":
    asyncio.run(main())
