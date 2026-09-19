"""Single-flight coalescing properties (Phase 6, Properties 1-4).

Feature: intelligent-execution-optimization.

* Property 1 — concurrent misses compute exactly once (Req 1.2, 1.4).
* Property 2 — a cached key never recomputes (Req 1.3).
* Property 3 — a failing compute propagates and stores nothing (Req 1.5, 1.6).
* Property 4 — distinct keys are independent (Req 1.7).

All tests run over a real :class:`MemoryCache` with the in-memory backend and
``single_flight=True``. Concurrency is forced with ``asyncio.gather`` over a
compute stub that awaits a shared barrier so the callers genuinely overlap
in-flight, exercising the coalescing path rather than a serialised sequence.
"""

from __future__ import annotations

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse import CacheConfig, MemoryCache


def _sf_cache() -> MemoryCache:
    return MemoryCache(CacheConfig(backend="memory", single_flight=True))


class _CountingCompute:
    """A compute stub that counts invocations and overlaps concurrent callers.

    Each call increments ``calls`` and then waits on a shared event so that all
    concurrent callers are in-flight simultaneously before any completes — the
    condition single-flight must coalesce.
    """

    def __init__(self, value: object, release: asyncio.Event) -> None:
        self.calls = 0
        self._value = value
        self._release = release

    async def __call__(self) -> object:
        self.calls += 1
        await self._release.wait()
        return self._value


class TestProperty1ComputeOnce:
    """Feature: intelligent-execution-optimization, Property 1.

    Concurrent misses compute exactly once.

    Validates: Requirements 1.2, 1.4
    """

    @settings(max_examples=100, deadline=None)
    @given(n=st.integers(min_value=2, max_value=12), value=st.text(max_size=20))
    async def test_concurrent_misses_compute_once(self, n: int, value: str) -> None:
        cache = _sf_cache()
        release = asyncio.Event()
        compute = _CountingCompute(value, release)

        async def call() -> object:
            return await cache.get_or_compute(["k"], compute, scope="global", scope_id=None)

        tasks = [asyncio.ensure_future(call()) for _ in range(n)]
        # Let all callers reach the in-flight barrier, then release them.
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(*tasks)

        assert compute.calls == 1  # coalesced (Req 1.2)
        assert all(r == value for r in results)  # all share the one result
        # A subsequent call is a hit (Req 1.4) — compute count unchanged.
        again = await cache.get_or_compute(["k"], compute, scope="global", scope_id=None)
        assert again == value
        assert compute.calls == 1


class TestProperty2NoRecompute:
    """Feature: intelligent-execution-optimization, Property 2.

    A cached key never recomputes.

    Validates: Requirements 1.3
    """

    @settings(max_examples=100, deadline=None)
    @given(value=st.text(max_size=20))
    async def test_cached_key_never_recomputes(self, value: str) -> None:
        cache = _sf_cache()
        release = asyncio.Event()
        release.set()
        compute = _CountingCompute(value, release)

        first = await cache.get_or_compute(["k"], compute, scope="global", scope_id=None)
        assert first == value
        assert compute.calls == 1

        # A second, non-concurrent call hits the stored value; compute untouched.
        second = await cache.get_or_compute(["k"], compute, scope="global", scope_id=None)
        assert second == value
        assert compute.calls == 1


class TestProperty3FailurePropagates:
    """Feature: intelligent-execution-optimization, Property 3.

    A failing compute propagates and stores nothing.

    Validates: Requirements 1.5, 1.6
    """

    @settings(max_examples=100, deadline=None)
    @given(n=st.integers(min_value=2, max_value=8))
    async def test_failure_propagates_and_stores_nothing(self, n: int) -> None:
        cache = _sf_cache()
        calls = {"n": 0}
        release = asyncio.Event()

        async def failing() -> object:
            calls["n"] += 1
            await release.wait()
            raise RuntimeError("boom")

        async def call() -> object:
            return await cache.get_or_compute(["k"], failing, scope="global", scope_id=None)

        tasks = [asyncio.ensure_future(call()) for _ in range(n)]
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Every waiter saw the exception (Req 1.5).
        assert all(isinstance(r, RuntimeError) for r in results)
        # Coalesced: the failing body ran once for the concurrent burst.
        assert calls["n"] == 1
        # Nothing stored + key cleared: a later call retries (Req 1.5, 1.6).
        succeeded = {"ok": False}

        async def recover() -> object:
            succeeded["ok"] = True
            return "recovered"

        result = await cache.get_or_compute(["k"], recover, scope="global", scope_id=None)
        assert result == "recovered"
        assert succeeded["ok"] is True


class TestProperty4DistinctKeys:
    """Feature: intelligent-execution-optimization, Property 4.

    Distinct keys are independent.

    Validates: Requirements 1.7
    """

    @settings(max_examples=100, deadline=None)
    @given(keys=st.lists(st.text(min_size=1, max_size=8), min_size=2, max_size=6, unique=True))
    async def test_distinct_keys_independent(self, keys: list[str]) -> None:
        cache = _sf_cache()
        release = asyncio.Event()
        counts: dict[str, int] = dict.fromkeys(keys, 0)

        def make_compute(k: str):
            async def compute() -> object:
                counts[k] += 1
                await release.wait()
                return f"value::{k}"

            return compute

        async def call(k: str) -> object:
            return await cache.get_or_compute([k], make_compute(k), scope="global", scope_id=None)

        tasks = [asyncio.ensure_future(call(k)) for k in keys]
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(*tasks)

        # Each distinct key computed once and got its own result (Req 1.7).
        assert all(counts[k] == 1 for k in keys)
        assert results == [f"value::{k}" for k in keys]
