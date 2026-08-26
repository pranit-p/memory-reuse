"""Example: caching Strands and CrewAI tool calls with ``cached_tool``.

memory-reuse ships a ``cached_tool`` decorator for the Strands Agents and CrewAI
frameworks that mirrors the LangGraph one exactly — same signature, same keying,
scope resolution, TTL expiry, and exact-vs-semantic routing. Each delegates to
the shared caching machinery, so behaviour is identical across frameworks.

    from memory_reuse.integrations.strands import cached_tool   # Strands
    from memory_reuse.integrations.crewai import cached_tool    # CrewAI

    cached_tool(cache, *, scope="global", ttl=300, semantic=False, exact_only=False)

The real Strands/CrewAI packages are optional:

    pip install "memory-reuse[strands]"      # Strands Agents
    pip install "memory-reuse[crewai]"       # CrewAI

Each dependency is imported lazily inside the returned wrapper, so a missing
package surfaces as a clear ``BackendNotAvailableError`` naming the extra rather
than a raw ``ImportError``.

This demo stays fully offline: it neutralises the lazy dependency guards so the
decorators run without either framework installed, and the "tool" is a plain
async function with a call counter so you can see a hit skip the body. In a real
app you install the extra and drop the guard patching.

Run this example:

    python examples/framework_tool_caching.py
"""

from __future__ import annotations

import asyncio
from unittest import mock

from memory_reuse import MemoryCache
from memory_reuse.exceptions import ConfigurationError
from memory_reuse.integrations import crewai as crewai_integration
from memory_reuse.integrations import strands as strands_integration


class Counter:
    """Tracks how many times the underlying tool body actually executed."""

    def __init__(self) -> None:
        self.calls = 0


async def demo_wrapper(name: str, module: object, guard_name: str) -> None:
    """Run a store-then-replay round trip through one framework's wrapper.

    ``module`` is the integration module (strands or crewai) and ``guard_name``
    is the lazy dependency guard to neutralise so the demo runs offline. In your
    own code you install the extra and skip the ``mock.patch.object`` line.
    """
    print(f"\n=== {name}: cached_tool store-and-replay ===")
    counter = Counter()

    # Offline only: pretend the framework is installed by making the guard a
    # no-op. Remove this line once you `pip install "memory-reuse[<extra>]"`.
    with mock.patch.object(module, guard_name, lambda: None):
        cache = MemoryCache()

        @module.cached_tool(cache, scope="global", ttl=300)  # type: ignore[attr-defined]
        async def fetch_weather(city: str) -> dict[str, object]:
            counter.calls += 1
            print(f"  [tool] Fetching weather for {city!r} (real call)")
            await asyncio.sleep(0.05)  # simulate an API round trip
            return {"city": city, "temp_c": 21, "sky": "clear"}

        # First call: a miss — the tool body runs and the result is stored.
        first = await fetch_weather(city="Pune")
        print(f"  Result: {first}")

        # Second call, same args: a hit — the stored value is replayed and the
        # body does NOT run again.
        second = await fetch_weather(city="Pune")
        print(f"  Result: {second}  (replayed)")

        print(f"  Tool body executed {counter.calls} time(s) for 2 calls.")
        assert counter.calls == 1
        assert first == second

        await cache.close()


async def demo_crewai_conflict() -> None:
    """CrewAI rejects the contradictory ``exact_only`` + ``semantic`` combo."""
    print("\n=== CrewAI: exact_only + semantic is rejected up front ===")
    cache = MemoryCache()
    try:
        crewai_integration.cached_tool(cache, exact_only=True, semantic=True)
    except ConfigurationError as exc:
        print(f"  Raised ConfigurationError as expected: {exc}")
    else:  # pragma: no cover - demo guard
        print("  (unexpected: no error raised)")
    await cache.close()


async def main() -> None:
    await demo_wrapper("Strands", strands_integration, "_require_strands")
    await demo_wrapper("CrewAI", crewai_integration, "_require_crewai")
    await demo_crewai_conflict()

    print(
        "\nBoth wrappers reuse the same caching primitives, so a cached tool "
        "call costs ~0ms and 0 API calls on a hit."
    )


if __name__ == "__main__":
    asyncio.run(main())
