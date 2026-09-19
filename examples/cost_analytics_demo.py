"""Example: Cost analytics — quantify tokens, cost, and latency saved.

Hit rate tells you *how often* the cache helped; the analytics layer tells you
*how much it saved* — tokens, money, and wall-clock time avoided. This demo
runs a simulated expensive LLM call several times with the same prompt: the
first call is a miss (the "LLM" runs), and every repeat is a hit that avoids the
call. On each hit we record what the hit avoided, then print the running
analytics snapshot.

Everything here is fully offline — no API key, no model download, no network.
The "LLM" is a local function that reports a token count and a latency, exactly
as a real provider's ``usage`` block would.

Run this example:

    python examples/cost_analytics_demo.py

Then try the CLI against the snapshot this writes:

    memory-reuse stats   --snapshot analytics_snapshot.json
    memory-reuse savings --snapshot analytics_snapshot.json
"""

from __future__ import annotations

import asyncio
import time

from memory_reuse import CacheConfig, CacheHitEvent, MemoryCache, PricingConfig
from memory_reuse.cli import dump_snapshot

# What one call to our fake model "costs". A real integration reads these from
# the provider response's ``usage`` block instead of hard-coding them.
PROMPT = "Summarise the Q3 revenue report."
TOKENS_IN = 1200
TOKENS_OUT = 400
LLM_LATENCY_S = 0.35


async def fake_llm_call(prompt: str) -> str:
    """Simulate an expensive LLM API call with real latency."""
    print(f"  [LLM] Calling API for: {prompt!r}")
    await asyncio.sleep(LLM_LATENCY_S)
    return "Q3 revenue rose 12% QoQ, driven by enterprise renewals."


async def answer(cache: MemoryCache, prompt: str) -> str:
    """Look the prompt up; on a miss run the LLM and store, on a hit record savings.

    The key idea: attribution is *caller-supplied*. The cache can't know a
    call's token or latency cost, so on a hit we tell it what the hit avoided
    via ``record_hit_event``.
    """
    key_parts = ["qa", prompt]

    cached = await cache.exact.get(key_parts, scope="global", scope_id=None)
    if cached is not None:
        print("  -> cache HIT (LLM not called)")
        # Record what this hit avoided: the tokens and latency of one LLM call.
        cache.record_hit_event(
            CacheHitEvent(
                tokens_in=TOKENS_IN,
                tokens_out=TOKENS_OUT,
                latency_saved=LLM_LATENCY_S,
                operation="summarise_report",
            )
        )
        return cached

    print("  -> cache MISS")
    started = time.perf_counter()
    result = await fake_llm_call(prompt)
    print(f"     (real call took {time.perf_counter() - started:.2f}s)")
    await cache.exact.set(key_parts, result, scope="global", scope_id=None, ttl=3600)
    return result


async def main() -> None:
    # Pricing is provider-agnostic: supply the per-token prices for your model.
    # These are illustrative (roughly GPT-4o-mini-style rates).
    cache = MemoryCache(
        CacheConfig(
            backend="memory",
            pricing=PricingConfig(
                input_token_price=0.00000015,  # $0.15 / 1M input tokens
                output_token_price=0.00000060,  # $0.60 / 1M output tokens
                currency="USD",
            ),
        )
    )

    print("\n=== 1 miss + 4 hits on the same prompt ===")
    for i in range(5):
        print(f"\nRequest {i + 1}:")
        await answer(cache, PROMPT)

    # Now simulate a busy production day: thousands more hits on the same prompt.
    # Each hit's saving is a fraction of a cent, but they accumulate at full
    # precision and only round when read — so the total shows real money.
    print("\n=== + 5,000 more hits (simulated production traffic) ===")
    for _ in range(5000):
        cache.record_hit_event(
            CacheHitEvent(
                tokens_in=TOKENS_IN,
                tokens_out=TOKENS_OUT,
                latency_saved=LLM_LATENCY_S,
                operation="summarise_report",
            )
        )

    # The analytics snapshot: hit rate (from the same counters as `stats`) plus
    # the savings the four hits accumulated.
    snap = cache.analytics
    print("\n=== Analytics snapshot ===")
    print(f"  Hit rate:      {snap.hit_rate:.1%}")
    print(f"  Tokens saved:  {snap.tokens_saved}")
    print(f"  Cost saved:    {snap.cost_saved} {snap.currency}")
    print(f"  Latency saved: {snap.latency_saved:.2f}s")

    # Sanity: (4 + 5000) hits x (1200 + 400) tokens = 8,006,400 tokens saved.
    # Per hit: 1200*1.5e-7 + 400*6e-7 = $0.00042; x5004 hits ~= $2.10 saved.
    # Contributions accumulate at full precision and round only when read, so a
    # workload of sub-cent savings still totals real money here.

    # Dump a snapshot file so you can inspect it with the `memory-reuse` CLI.
    dump_snapshot(cache, "analytics_snapshot.json")
    print("\nWrote analytics_snapshot.json — try:")
    print("  memory-reuse stats   --snapshot analytics_snapshot.json")
    print("  memory-reuse savings --snapshot analytics_snapshot.json")

    await cache.close()


if __name__ == "__main__":
    asyncio.run(main())
