"""Example: LangGraph agent with cached nodes and cached tools.

This example shows how to use the ``cached_node`` and ``cached_tool``
decorators in a simulated LangGraph workflow.

Run this example:

    python examples/langgraph_agent_example.py
"""

from __future__ import annotations

import asyncio
import logging

from memory_reuse import CacheConfig, MemoryCache
from memory_reuse.integrations import cached_node, cached_tool

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

cache = MemoryCache(CacheConfig(backend="memory", default_ttl=600))
cache.set_context(user_id="user-42", session_id="session-abc")

# ---------------------------------------------------------------------------
# Simulated external tools
# ---------------------------------------------------------------------------


@cached_tool(cache, scope="global", ttl=300)
async def web_search(query: str) -> list[str]:
    """Fetch search results (cached for 5 minutes globally)."""
    print(f"  [TOOL] web_search(query={query!r}) — making real call")
    await asyncio.sleep(0.05)  # Simulate HTTP latency
    return [f"Result 1 for {query}", f"Result 2 for {query}"]


@cached_tool(cache, scope="user", ttl=120)
async def get_user_profile(user_id: str) -> dict:
    """Fetch user profile (cached per user for 2 minutes)."""
    print(f"  [TOOL] get_user_profile(user_id={user_id!r}) — making real call")
    await asyncio.sleep(0.02)
    return {"user_id": user_id, "name": "Alice", "plan": "pro"}


# ---------------------------------------------------------------------------
# LangGraph-style nodes
# ---------------------------------------------------------------------------


@cached_node(cache, scope="user", key_fields=["messages"])
async def summarise_node(state: dict) -> dict:
    """Summarise messages (cached per user, keyed on 'messages' field)."""
    print(f"  [NODE] summarise_node — computing summary for user={state.get('user_id')!r}")
    await asyncio.sleep(0.1)  # Simulate LLM call
    messages = state.get("messages", [])
    return {"summary": f"Summary of {len(messages)} message(s)"}


# ---------------------------------------------------------------------------
# Simulated agent run
# ---------------------------------------------------------------------------


async def run_agent(user_id: str, query: str) -> None:
    print(f"\n--- Agent run: user={user_id!r}, query={query!r} ---")

    # Tool calls
    results = await web_search(query=query)
    print(f"  Search results: {results}")

    profile = await get_user_profile(user_id=user_id)
    print(f"  User profile: {profile}")

    # Node execution
    state = {
        "user_id": user_id,
        "messages": [query, "follow-up question"],
    }
    output = await summarise_node(state)
    print(f"  Node output: {output}")


async def main() -> None:
    # First run — all cache misses
    await run_agent("user-42", "latest AI agent frameworks")

    print("\n=== Repeating the same run (should be all cache hits) ===")

    # Second run — all cache hits, no real calls made
    await run_agent("user-42", "latest AI agent frameworks")

    print("\n=== Stats ===")
    stats = cache.stats
    print(f"  Total requests: {stats.total_requests}")
    print(f"  Hits:           {stats.hits}")
    print(f"  Misses:         {stats.misses}")
    print(f"  Hit rate:       {stats.hit_rate:.1%}")

    await cache.close()


if __name__ == "__main__":
    asyncio.run(main())
