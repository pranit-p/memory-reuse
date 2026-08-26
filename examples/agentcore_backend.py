"""Example: the AWS AgentCore shared backend (cross-microVM cache sharing).

The ``agentcore`` backend stores cache entries in the managed AWS AgentCore
store instead of process memory, so a value cached in one AgentCore microVM is
served to a request handled by another — solving the cold-cache-per-VM problem
where each microVM otherwise starts empty.

In a real deployment you select it via config and install the extra:

    pip install "memory-reuse[agentcore]"

    from memory_reuse import MemoryCache, CacheConfig

    cache = MemoryCache(CacheConfig(
        backend="agentcore",
        agentcore_region="us-east-1",
        agentcore_memory_id="mem-123",
    ))

    # ...or from the environment:
    #   export MEMORY_REUSE_BACKEND=agentcore
    #   export MEMORY_REUSE_AGENTCORE_REGION=us-east-1
    #   export MEMORY_REUSE_AGENTCORE_MEMORY_ID=mem-123
    cache = MemoryCache.from_env()

This demo stays fully offline: it talks to a tiny in-process fake AgentCore
service (a dict) shared by two backend instances, so no AWS call is made. The
``AgentCoreBackend`` accepts an injected ``client=`` for exactly this — real
code omits it and lets the backend build a boto3 client from the settings.

Run this example:

    python examples/agentcore_backend.py
"""

from __future__ import annotations

import asyncio

from memory_reuse.backends.agentcore import AgentCoreBackend, AgentCoreSettings
from memory_reuse.exceptions import BackendConnectionError


class FakeAgentCoreService:
    """A dict-backed stand-in for the managed AgentCore store (shared state)."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, object]] = {}
        self.reachable = True


class FakeAgentCoreClient:
    """Offline client over a shared :class:`FakeAgentCoreService`.

    Two clients built over one service share state, modelling two microVMs that
    point at the same AgentCore memory resource. Matches the transport-shaped
    surface the real backend calls: ``put_item`` / ``get_item`` /
    ``delete_item`` / ``scan`` / ``ping``.
    """

    def __init__(self, service: FakeAgentCoreService) -> None:
        self._service = service

    def _check(self) -> None:
        if not self._service.reachable:
            raise ConnectionError("fake AgentCore service is unreachable")

    def put_item(self, item_id: str, item: dict[str, object]) -> None:
        self._check()
        self._service.store[item_id] = dict(item)

    def get_item(self, item_id: str) -> dict[str, object] | None:
        self._check()
        item = self._service.store.get(item_id)
        return dict(item) if item is not None else None

    def delete_item(self, item_id: str) -> None:
        self._check()
        self._service.store.pop(item_id, None)

    def scan(self, prefix: str = "") -> list[str]:
        self._check()
        return [k for k in self._service.store if k.startswith(prefix)]

    def ping(self) -> bool:
        return self._service.reachable


def _backend(service: FakeAgentCoreService) -> AgentCoreBackend:
    """Build a backend over a client bound to the shared fake service."""
    return AgentCoreBackend(
        AgentCoreSettings(region="us-east-1", memory_id="mem-demo"),
        client=FakeAgentCoreClient(service),
    )


async def main() -> None:
    # One shared managed store, two independent backends (two "microVMs").
    service = FakeAgentCoreService()
    vm_a = _backend(service)
    vm_b = _backend(service)

    print("\n=== Cross-microVM sharing ===")
    await vm_a.set("prompt:hello", b"cached-answer", ttl=300)
    print("  microVM A stored 'prompt:hello'")
    seen_by_b = await vm_b.get("prompt:hello")
    print(f"  microVM B reads it back: {seen_by_b!r}  (shared, no cold start)")
    assert seen_by_b == b"cached-answer"

    print("\n=== Raw bytes round-trip (any 0..1 MiB payload) ===")
    blob = bytes(range(256)) * 8  # 2 KiB of arbitrary bytes
    await vm_a.set("blob", blob)
    assert await vm_b.get("blob") == blob
    print(f"  {len(blob)} bytes round-tripped unchanged.")

    print("\n=== TTL expiry (lazy, on read) ===")
    # A short TTL; entries past expiry read as absent. We won't sleep here — the
    # point is that get/exists treat elapsed entries as missing.
    await vm_a.set("temp", b"soon-gone", ttl=1)
    print(f"  exists right after set: {await vm_b.exists('temp')}")

    print("\n=== Absence + delete semantics ===")
    print(f"  get on an absent key: {await vm_b.get('never-set')!r}")
    await vm_a.delete("prompt:hello")
    print(f"  after delete, B sees: {await vm_b.get('prompt:hello')!r}")

    print("\n=== Connectivity: ping never raises, ops do ===")
    service.reachable = False
    print(f"  ping() while unreachable: {await vm_a.ping()}")
    try:
        await vm_a.get("prompt:hello")
    except BackendConnectionError as exc:
        print(f"  get() while unreachable raised: {type(exc).__name__}")
    service.reachable = True

    print("\nThe same store backs both the exact and tool caches when you select")
    print('backend="agentcore" on CacheConfig.')


if __name__ == "__main__":
    asyncio.run(main())
