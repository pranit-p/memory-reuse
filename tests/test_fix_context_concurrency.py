"""Regression tests for concurrent request-context isolation (Fix #2).

``MemoryCache`` holds request context (user/session) in a ``ContextVar`` instead
of a shared mutable dict, so concurrent asyncio tasks sharing one cache instance
never overwrite one another's context. These tests interleave requests from
different users and assert each sees only its own context, that nested scopes
restore correctly (including on exceptions), and that cached values stay
isolated per user.
"""

from __future__ import annotations

import asyncio

import pytest

from memory_reuse import CacheConfig, MemoryCache


@pytest.fixture
def cache() -> MemoryCache:
    return MemoryCache(CacheConfig(backend="memory", default_ttl=3600))


class TestContextVarIsolation:
    async def test_interleaved_tasks_keep_separate_context(self, cache: MemoryCache) -> None:
        """Concurrent tasks each set and observe only their own user context."""
        observed: dict[str, str | None] = {}

        async def worker(user: str) -> None:
            # Each task sets its own context; a ContextVar makes this task-local.
            cache.set_context(user_id=user)
            # Yield control repeatedly so the scheduler interleaves the workers,
            # which is exactly the race that a shared dict would lose.
            for _ in range(5):
                await asyncio.sleep(0)
                assert cache._context.get("user_id") == user
            observed[user] = cache._context.get("user_id")

        await asyncio.gather(*(worker(f"user-{i}") for i in range(10)))
        assert observed == {f"user-{i}": f"user-{i}" for i in range(10)}

    async def test_set_context_in_task_does_not_leak_to_parent(self, cache: MemoryCache) -> None:
        cache.set_context(user_id="parent")

        async def child() -> None:
            cache.set_context(user_id="child")
            assert cache._context.get("user_id") == "child"

        await asyncio.create_task(child())
        # The child's assignment happened in its own copied context.
        assert cache._context.get("user_id") == "parent"

    async def test_context_manager_restores_on_exit(self, cache: MemoryCache) -> None:
        cache.set_context(user_id="base")
        with cache.context(user_id="temp"):
            assert cache._context.get("user_id") == "temp"
        assert cache._context.get("user_id") == "base"

    async def test_context_manager_restores_on_exception(self, cache: MemoryCache) -> None:
        cache.set_context(user_id="base")
        with pytest.raises(RuntimeError), cache.context(user_id="temp"):
            assert cache._context.get("user_id") == "temp"
            raise RuntimeError("boom")
        # Even though the body raised, the previous context is restored.
        assert cache._context.get("user_id") == "base"

    async def test_nested_context_managers(self, cache: MemoryCache) -> None:
        with cache.context(user_id="outer"):
            assert cache._context.get("user_id") == "outer"
            with cache.context(user_id="inner"):
                assert cache._context.get("user_id") == "inner"
            assert cache._context.get("user_id") == "outer"


class TestConcurrentCacheIsolationViaContext:
    async def test_user_scoped_tool_uses_task_local_context(self) -> None:
        """Interleaved users resolve their own scope from context, no cross-leak."""
        from memory_reuse.integrations.langgraph import cached_tool

        cache = MemoryCache(CacheConfig(backend="memory", default_ttl=3600))

        @cached_tool(cache, scope="user", ttl=300)
        async def whoami() -> str:
            # The tool resolves user_id from the cache context (set per task).
            return f"profile-of-{cache._context.get('user_id')}"

        results: dict[str, str] = {}

        async def worker(user: str) -> None:
            with cache.context(user_id=user):
                for _ in range(3):
                    await asyncio.sleep(0)
                results[user] = await whoami()

        await asyncio.gather(*(worker(f"u{i}") for i in range(8)))
        # Each user got its own result; nobody saw another user's profile.
        assert results == {f"u{i}": f"profile-of-u{i}" for i in range(8)}

    async def test_clear_context_is_task_local(self) -> None:
        cache = MemoryCache(CacheConfig(backend="memory"))
        cache.set_context(user_id="main")

        async def child() -> None:
            cache.clear_context()
            assert cache._context.get("user_id") is None

        await asyncio.create_task(child())
        assert cache._context.get("user_id") == "main"
