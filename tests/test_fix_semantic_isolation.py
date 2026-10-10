"""Regression tests for semantic-cache isolation (Fix #1).

These assert that a semantic (similarity) hit can only be served to the *same*
operation and the *same* scope that stored it — two unrelated tools/nodes, or
two different users, must never share a cached result merely because their query
texts are identical or similar.

The ``StubEmbedder`` maps identical text to an identical vector, so *without*
isolation these lookups would all hit on text alone; the assertions below prove
the operation/scope segmentation prevents that leakage while a same-operation
reworded lookup still hits.
"""

from __future__ import annotations

from tests.conftest import StubEmbedder, make_semantic_cache


class TestCrossOperationIsolation:
    async def test_different_operations_do_not_share_semantic_result(self) -> None:
        """Two operations with identical query text must not cross-hit (Fix #1)."""
        cache = make_semantic_cache(StubEmbedder(), similarity_threshold=0.0)

        # Operation A stores a value under its own operation namespace.
        await cache.store(
            ["tool_a", {"q": "x"}],
            "same question text",
            "answer-A",
            scope="global",
            scope_id=None,
            operation="tool_a",
        )

        # Operation B looks up the identical text but a different operation.
        result = await cache.lookup(
            ["tool_b", {"q": "x"}],
            "same question text",
            scope="global",
            scope_id=None,
            operation="tool_b",
        )

        assert result is None, "operation B must not see operation A's cached result"

    async def test_same_operation_reworded_still_hits(self) -> None:
        """Isolation must not block a legitimate same-operation semantic hit."""
        cache = make_semantic_cache(StubEmbedder(), similarity_threshold=0.0)

        await cache.store(
            ["tool_a", {"q": "x"}],
            "how many moons does mars have",
            42,
            scope="global",
            scope_id=None,
            operation="tool_a",
        )
        # Different exact key, same operation + same text → semantic hit.
        result = await cache.lookup(
            ["tool_a", {"q": "y"}],
            "how many moons does mars have",
            scope="global",
            scope_id=None,
            operation="tool_a",
        )
        assert result == 42

    async def test_no_operation_shares_namespace_backcompat(self) -> None:
        """Without an operation, raw lookup/store still match on text within a scope.

        This preserves the documented low-level ``lookup``/``store`` behaviour so
        callers managing their own keys are not broken by the isolation fix.
        """
        cache = make_semantic_cache(StubEmbedder(), similarity_threshold=0.0)
        await cache.store(["k1"], "shared text", "v", scope="global", scope_id=None)
        result = await cache.lookup(["k2"], "shared text", scope="global", scope_id=None)
        assert result == "v"


class TestCrossScopeIsolation:
    async def test_different_users_do_not_share_semantic_result(self) -> None:
        """A user-scoped semantic entry must not leak to another user (Fix #1)."""
        cache = make_semantic_cache(StubEmbedder(), similarity_threshold=0.0)

        await cache.store(
            ["tool", {}],
            "what are my orders",
            "alice-orders",
            scope="user",
            scope_id="alice",
            operation="tool",
        )
        # Bob asks the identical question; must not see alice's cached answer.
        bob_result = await cache.lookup(
            ["tool", {}],
            "what are my orders",
            scope="user",
            scope_id="bob",
            operation="tool",
        )
        assert bob_result is None

        # Alice still hits her own entry.
        alice_result = await cache.lookup(
            ["tool", {}],
            "what are my orders",
            scope="user",
            scope_id="alice",
            operation="tool",
        )
        assert alice_result == "alice-orders"

    async def test_global_and_user_scopes_are_isolated(self) -> None:
        cache = make_semantic_cache(StubEmbedder(), similarity_threshold=0.0)
        await cache.store(
            ["tool", {}], "q", "global-val", scope="global", scope_id=None, operation="tool"
        )
        result = await cache.lookup(
            ["tool", {}], "q", scope="user", scope_id="alice", operation="tool"
        )
        assert result is None


class TestDecoratorIsolation:
    async def test_two_cached_tools_do_not_cross_hit(self) -> None:
        """Two different semantic ``cached_tool``s with equal args never cross-hit."""
        from memory_reuse.integrations.langgraph import cached_tool

        cache = make_semantic_cache(StubEmbedder(), similarity_threshold=0.0)

        calls = {"a": 0, "b": 0}

        @cached_tool(cache, scope="global", ttl=300, semantic=True)
        async def tool_a(q: str) -> str:
            calls["a"] += 1
            return "from-a"

        @cached_tool(cache, scope="global", ttl=300, semantic=True)
        async def tool_b(q: str) -> str:
            calls["b"] += 1
            return "from-b"

        assert await tool_a("hello") == "from-a"
        # Same argument/text, different tool: must run its own body, not reuse A's.
        assert await tool_b("hello") == "from-b"
        assert calls == {"a": 1, "b": 1}
