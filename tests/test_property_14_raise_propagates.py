"""Property-based test: a raised tool body propagates and stores nothing.

Exercises the Strands and CrewAI ``cached_tool`` wrappers over the shared
``memory_reuse.integrations.langgraph`` machinery (Phase 4, Task 3.5). Both
wrappers delegate to the same base decorator, so a body that raises must behave
identically under either wrapper: the exception propagates unchanged and nothing
is stored for that cache key, so a subsequent call with the same arguments
re-runs the body and raises again.

Uses the Phase 4 scaffolding in ``tests/conftest.py`` (``RaisingTool``,
``make_tool_cache``, and the ``tool_arguments`` strategy). No real network, LLM,
or AWS calls are made; the Strands/CrewAI dependency guards are monkeypatched to
no-ops so the wrappers can run without those optional packages installed.

* **Property 14 — A raised tool body propagates and stores nothing:** for any
  tool arguments, when a decorated tool body raises, the exception propagates to
  the caller and no entry is stored; a second call with the same arguments
  re-runs the body (call count increments again) and re-raises.

Validates: Requirements 6.8, 7.10
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
from hypothesis import given, settings

from memory_reuse.integrations import crewai as crewai_integration
from memory_reuse.integrations import strands as strands_integration
from tests.conftest import RaisingTool, make_tool_cache, tool_arguments

# The two wrappers under test, each paired with the name of its lazy dependency
# guard so we can neutralise it to a no-op (the optional package is absent).
_WRAPPERS = [
    pytest.param(strands_integration, "_require_strands", id="strands"),
    pytest.param(crewai_integration, "_require_crewai", id="crewai"),
]


@contextlib.contextmanager
def _guard_disabled(module: object, guard_name: str) -> Iterator[None]:
    """Temporarily replace a wrapper's lazy dependency guard with a no-op.

    The optional Strands/CrewAI packages are absent in the test environment, so
    the guard would raise. This context manager stubs it out and restores the
    original afterwards. It is used instead of the function-scoped ``monkeypatch``
    fixture because Hypothesis does not reset function-scoped fixtures between
    generated inputs.
    """
    original = getattr(module, guard_name)
    setattr(module, guard_name, lambda: None)
    try:
        yield
    finally:
        setattr(module, guard_name, original)


class TestProperty14RaisePropagatesAndStoresNothing:
    """Feature: analytics-and-integrations, Property 14.

    A raised tool body propagates and stores nothing.

    Validates: Requirements 6.8, 7.10
    """

    @pytest.mark.parametrize("module, guard_name", _WRAPPERS)
    @settings(max_examples=100)
    @given(kwargs=tool_arguments)
    @pytest.mark.asyncio
    async def test_raise_propagates_and_stores_nothing(
        self,
        module: object,
        guard_name: str,
        kwargs: dict[str, object],
    ) -> None:
        """A raising body propagates and leaves the cache untouched.

        For arbitrary tool arguments, decorating an always-raising tool body and
        invoking it must re-raise the body's error rather than swallow it, and
        must store nothing. So a second invocation with the same arguments is
        still a miss: the body runs again (its call counter increments) and the
        error propagates again.
        """
        # Fresh cache per example so the first invocation is a guaranteed miss.
        cache = make_tool_cache()
        tool = RaisingTool()
        # The optional integration package is absent in the test env, so
        # neutralise its lazy import guard for this wrapper.
        with _guard_disabled(module, guard_name):
            decorated = module.cached_tool(cache, scope="global", ttl=300)(tool.run)

            # First call (a miss) must propagate the body's error unchanged.
            with pytest.raises(RuntimeError, match="tool boom"):
                await decorated(**kwargs)
            assert tool.calls == 1

            # Nothing was stored: the same-argument call still misses, so the
            # body runs again and the error propagates again.
            with pytest.raises(RuntimeError, match="tool boom"):
                await decorated(**kwargs)
            assert tool.calls == 2

    @pytest.mark.parametrize("module, guard_name", _WRAPPERS)
    @settings(max_examples=100)
    @given(kwargs=tool_arguments)
    @pytest.mark.asyncio
    async def test_raise_propagates_and_stores_nothing_async_body(
        self,
        module: object,
        guard_name: str,
        kwargs: dict[str, object],
    ) -> None:
        """Same guarantee holds for an async tool body.

        The wrappers always return an async wrapper and support both sync and
        async bodies; this exercises the async ``arun`` path so the
        propagate-and-store-nothing property is verified for both callable
        shapes.
        """
        cache = make_tool_cache()
        tool = RaisingTool()
        with _guard_disabled(module, guard_name):
            decorated = module.cached_tool(cache, scope="global", ttl=300)(tool.arun)

            with pytest.raises(RuntimeError, match="tool boom"):
                await decorated(**kwargs)
            assert tool.calls == 1

            with pytest.raises(RuntimeError, match="tool boom"):
                await decorated(**kwargs)
            assert tool.calls == 2
