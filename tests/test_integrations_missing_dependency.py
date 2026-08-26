"""Example tests: missing-dependency named errors for Strands/CrewAI.

Task 3.7 (Reqs 6.11, 7.11): when the optional ``strands`` / ``crewai``
dependency is absent, invoking the respective ``cached_tool``-decorated tool
raises :exc:`~memory_reuse.exceptions.BackendNotAvailableError` naming the extra
to install (``memory-reuse[strands]`` / ``memory-reuse[crewai]``) rather than
surfacing a raw ``ImportError`` traceback.

The dependency is simulated absent by setting ``sys.modules["strands"]`` (resp.
``"crewai"``) to ``None`` via ``monkeypatch.setitem`` — the same pattern the
embedder optionality tests use. A ``None`` entry in ``sys.modules`` makes the
lazy ``import strands`` inside the ``_require_*`` guard raise ``ImportError``,
which the guard converts into a named ``BackendNotAvailableError``. The guard
runs inside the returned wrapper on first invocation, so the error surfaces when
the decorated tool is awaited.
"""

from __future__ import annotations

import sys

import pytest

from memory_reuse.exceptions import BackendNotAvailableError
from memory_reuse.integrations.crewai import cached_tool as crewai_cached_tool
from memory_reuse.integrations.strands import cached_tool as strands_cached_tool
from tests.conftest import StubTool, make_tool_cache


class TestStrandsMissingDependency:
    """Req 6.11: absent Strands surfaces a clear, named error."""

    @pytest.mark.asyncio
    async def test_async_tool_raises_named_error(
        self, monkeypatch: pytest.MonkeyPatch, stub_tool: StubTool
    ) -> None:
        # Force the lazy ``import strands`` to fail even if it is installed.
        monkeypatch.setitem(sys.modules, "strands", None)
        cache = make_tool_cache()

        @strands_cached_tool(cache, scope="global", ttl=300)
        async def fetch(query: str) -> object:
            return await stub_tool.arun(query=query)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            await fetch(query="q")

        message = str(exc_info.value)
        assert "memory-reuse[strands]" in message
        # The tool body must not have executed.
        assert stub_tool.calls == 0

    @pytest.mark.asyncio
    async def test_sync_tool_raises_named_error(
        self, monkeypatch: pytest.MonkeyPatch, stub_tool: StubTool
    ) -> None:
        monkeypatch.setitem(sys.modules, "strands", None)
        cache = make_tool_cache()

        @strands_cached_tool(cache, scope="global", ttl=300)
        def compute(x: int) -> object:
            return stub_tool.run(x=x)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            await compute(x=1)

        assert "memory-reuse[strands]" in str(exc_info.value)
        assert stub_tool.calls == 0

    @pytest.mark.asyncio
    async def test_error_is_not_raw_import_error(
        self, monkeypatch: pytest.MonkeyPatch, stub_tool: StubTool
    ) -> None:
        monkeypatch.setitem(sys.modules, "strands", None)
        cache = make_tool_cache()

        @strands_cached_tool(cache)
        async def fetch(query: str) -> object:
            return await stub_tool.arun(query=query)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            await fetch(query="q")

        # A raw ImportError must not be what the caller sees.
        assert not isinstance(exc_info.value, ImportError)


class TestCrewAIMissingDependency:
    """Req 7.11: absent CrewAI surfaces a clear, named error."""

    @pytest.mark.asyncio
    async def test_async_tool_raises_named_error(
        self, monkeypatch: pytest.MonkeyPatch, stub_tool: StubTool
    ) -> None:
        monkeypatch.setitem(sys.modules, "crewai", None)
        cache = make_tool_cache()

        @crewai_cached_tool(cache, scope="global", ttl=300)
        async def fetch(query: str) -> object:
            return await stub_tool.arun(query=query)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            await fetch(query="q")

        message = str(exc_info.value)
        assert "memory-reuse[crewai]" in message
        assert stub_tool.calls == 0

    @pytest.mark.asyncio
    async def test_sync_tool_raises_named_error(
        self, monkeypatch: pytest.MonkeyPatch, stub_tool: StubTool
    ) -> None:
        monkeypatch.setitem(sys.modules, "crewai", None)
        cache = make_tool_cache()

        @crewai_cached_tool(cache, scope="global", ttl=300)
        def compute(x: int) -> object:
            return stub_tool.run(x=x)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            await compute(x=1)

        assert "memory-reuse[crewai]" in str(exc_info.value)
        assert stub_tool.calls == 0

    @pytest.mark.asyncio
    async def test_error_is_not_raw_import_error(
        self, monkeypatch: pytest.MonkeyPatch, stub_tool: StubTool
    ) -> None:
        monkeypatch.setitem(sys.modules, "crewai", None)
        cache = make_tool_cache()

        @crewai_cached_tool(cache)
        async def fetch(query: str) -> object:
            return await stub_tool.arun(query=query)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            await fetch(query="q")

        assert not isinstance(exc_info.value, ImportError)
