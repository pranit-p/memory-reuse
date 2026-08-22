"""Example: Using ExactCache to avoid redundant LLM calls.

Run this example:

    python examples/basic_exact_cache.py
"""

from __future__ import annotations

import asyncio
import logging

from memory_reuse import CacheConfig, MemoryCache

logging.basicConfig(level=logging.DEBUG)


async def fake_llm_call(prompt: str) -> str:
    """Simulate an expensive LLM API call."""
    print(f"  [LLM] Calling API for: {prompt!r}")
    await asyncio.sleep(0.1)  # Simulate network latency
    return f"LLM response for: {prompt}"


async def main() -> None:
    # Create a cache with the in-memory backend and a 1-hour TTL
    config = CacheConfig(backend="memory", default_ttl=3600)
    cache = MemoryCache(config)

    prompt = "Summarise the benefits of caching in AI agents"
    key_parts = ["gpt-4", prompt]

    print("\n=== First call (cache miss) ===")
    result = await cache.exact.get(key_parts, scope="global", scope_id=None)
    if result is None:
        result = await fake_llm_call(prompt)
        await cache.exact.set(key_parts, result, scope="global", scope_id=None, ttl=None)
    print(f"  Result: {result!r}")

    print("\n=== Second call (cache hit — LLM not called) ===")
    result = await cache.exact.get(key_parts, scope="global", scope_id=None)
    if result is None:
        result = await fake_llm_call(prompt)
        await cache.exact.set(key_parts, result, scope="global", scope_id=None, ttl=None)
    print(f"  Result: {result!r}")

    # Per-user scoped cache example
    print("\n=== User-scoped cache ===")
    cache.set_context(user_id="alice")
    user_key = ["user-preference", "theme"]
    await cache.exact.set(user_key, "dark", scope="user", scope_id="alice", ttl=None)
    alice_pref = await cache.exact.get(user_key, scope="user", scope_id="alice")
    bob_pref = await cache.exact.get(user_key, scope="user", scope_id="bob")
    print(f"  Alice's preference: {alice_pref}")  # "dark"
    print(f"  Bob's preference:   {bob_pref}")  # None — isolated

    print("\n=== Stats ===")
    stats = cache.stats
    print(f"  Hits:        {stats.hits}")
    print(f"  Misses:      {stats.misses}")
    print(f"  Hit rate:    {stats.hit_rate:.1%}")

    await cache.close()


if __name__ == "__main__":
    asyncio.run(main())
