"""Example tests: LiteLLM completion cache hits attribute token savings (Phase 5).

On a cache hit, ``cached_litellm_completion`` records the tokens the hit avoided
(from the stored response's ``usage`` block) into the analytics layer. These
tests exercise only the hit path — the response is pre-stored in the exact cache
— so no real LiteLLM call is made and the package's ``litellm`` extra is not
required.
"""

from __future__ import annotations

import pytest

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse.integrations.litellm import cached_litellm_completion

_MODEL = "gpt-4o-mini"
_MESSAGES = [{"role": "user", "content": "What is 2+2?"}]
_STORED = {
    "choices": [{"message": {"role": "assistant", "content": "4"}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


async def _prestore(cache: MemoryCache) -> None:
    """Pre-store a completion response so the next call is a cache hit."""
    key_parts = ["litellm.completion", _MODEL, _MESSAGES, {}]
    await cache.exact.set(key_parts, _STORED, scope="global", scope_id=None, ttl=3600)


@pytest.mark.asyncio
async def test_hit_records_token_savings() -> None:
    """A completion cache hit attributes prompt/completion tokens as saved."""
    cache = MemoryCache(CacheConfig(backend="memory"))
    await _prestore(cache)

    result = await cached_litellm_completion(cache, model=_MODEL, messages=_MESSAGES)
    assert result == _STORED  # served from cache, no LiteLLM call

    snap = cache.analytics
    assert snap.tokens_saved == 12 + 3


@pytest.mark.asyncio
async def test_hit_attribution_is_noop_when_stats_disabled() -> None:
    """With enable_stats=False the hit still serves but records no savings."""
    cache = MemoryCache(CacheConfig(backend="memory", enable_stats=False))
    await _prestore(cache)

    result = await cached_litellm_completion(cache, model=_MODEL, messages=_MESSAGES)
    assert result == _STORED
    assert cache.analytics.tokens_saved == 0


@pytest.mark.asyncio
async def test_hit_without_usage_block_does_not_raise() -> None:
    """A stored response lacking a usage block records nothing and never raises."""
    cache = MemoryCache(CacheConfig(backend="memory"))
    key_parts = ["litellm.completion", _MODEL, _MESSAGES, {}]
    await cache.exact.set(
        key_parts,
        {"choices": [{"message": {"content": "4"}}]},  # no "usage"
        scope="global",
        scope_id=None,
        ttl=3600,
    )

    result = await cached_litellm_completion(cache, model=_MODEL, messages=_MESSAGES)
    assert result["choices"][0]["message"]["content"] == "4"
    assert cache.analytics.tokens_saved == 0
