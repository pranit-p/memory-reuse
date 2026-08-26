"""Lazy-import and core-import smoke tests (Task 8.1).

Covers the cross-cutting optionality requirements for Phase 4:

* **Req 10.1 / 10.2 — bare core import loads no optional dep.** A fresh
  ``import memory_reuse`` (plus every always-present submodule, including the
  Phase 4 integration/backend modules) must succeed and must *not* pull in
  ``strands`` / ``crewai`` / ``boto3``. Because ``sys.modules`` is a
  process-global that other tests in the session may have already populated,
  the "loads none of them" check runs in a **fresh subprocess** via
  ``subprocess.run([sys.executable, "-c", ...])`` so the assertion is not
  polluted by the rest of the test session.
* **Req 10.3 — lazy import at first construction/use.** Each capability imports
  its third-party dependency only when first constructed or used. We assert this
  in-process by simulating the dependency as absent
  (``monkeypatch.setitem(sys.modules, dep, None)``): importing the module still
  succeeds, and the third-party dependency is only demanded (surfacing a named
  :exc:`~memory_reuse.exceptions.BackendNotAvailableError`) on first
  construction/use — the Strands/CrewAI ``cached_tool`` on first invocation, and
  the ``AgentCoreBackend(client=None)`` on construction.
* **Req 10.5 — core caches behave identically regardless of optional deps.** A
  round-trip through the exact and tool caches works with the optional deps
  simulated absent.

Everything here is deterministic and fully offline: no real network, LLM, or
AWS call.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from memory_reuse.backends.agentcore import AgentCoreBackend, AgentCoreSettings
from memory_reuse.exceptions import BackendNotAvailableError
from memory_reuse.integrations.crewai import cached_tool as crewai_cached_tool
from memory_reuse.integrations.strands import cached_tool as strands_cached_tool
from tests.conftest import StubTool, make_tool_cache

# The optional Phase 4 third-party dependencies that must stay lazily imported.
_OPTIONAL_DEPS = ("strands", "crewai", "boto3")

# Always-present submodules a bare core install must import without any optional
# Phase 4 dependency present.
_CORE_SUBMODULES = (
    "memory_reuse.config",
    "memory_reuse.core",
    "memory_reuse.backends",
    "memory_reuse.integrations.strands",
    "memory_reuse.integrations.crewai",
    "memory_reuse.backends.agentcore",
)


class TestBareCoreImportLoadsNoOptionalDep:
    """Req 10.1, 10.2: bare core import succeeds and pulls in no optional dep."""

    def test_bare_import_loads_none_of_the_optional_deps(self) -> None:
        # sys.modules is process-global and other tests in this session may have
        # already imported an optional dep. Run the check in a fresh subprocess
        # so the "loads none of them" assertion reflects a clean interpreter.
        lines = ["import sys", "import memory_reuse"]
        lines += [f"import {name}" for name in _CORE_SUBMODULES]
        lines += [
            f'assert "{dep}" not in sys.modules, ' f'"{dep} was imported by a bare core import"'
            for dep in _OPTIONAL_DEPS
        ]
        script = "\n".join(lines) + "\n"

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            "bare core import pulled in an optional dependency:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_bare_import_of_package_succeeds_in_subprocess(self) -> None:
        # A minimal sanity check: importing just the top-level package succeeds
        # in a clean interpreter and still loads none of the optional deps.
        lines = ["import sys", "import memory_reuse  # noqa: F401"]
        lines += [f'assert "{dep}" not in sys.modules' for dep in _OPTIONAL_DEPS]
        script = "\n".join(lines) + "\n"

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


class TestModulesImportWithoutOptionalDep:
    """Req 10.1, 10.3: capability modules import even with their dep absent."""

    @pytest.mark.parametrize("dep", _OPTIONAL_DEPS)
    def test_reimport_core_submodules_with_dep_absent(
        self, monkeypatch: pytest.MonkeyPatch, dep: str
    ) -> None:
        # A ``None`` entry in sys.modules makes ``import <dep>`` raise
        # ImportError, simulating the dependency being uninstalled.
        monkeypatch.setitem(sys.modules, dep, None)

        # Force a fresh import of each capability module under the simulated
        # absence so we exercise module top-level (which must not import the dep).
        for name in (
            "memory_reuse.integrations.strands",
            "memory_reuse.integrations.crewai",
            "memory_reuse.backends.agentcore",
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)
            __import__(name)


class TestStrandsLazyImport:
    """Req 10.3, 10.4: Strands imported only on first tool invocation."""

    @pytest.mark.asyncio
    async def test_dep_only_required_on_first_use(
        self, monkeypatch: pytest.MonkeyPatch, stub_tool: StubTool
    ) -> None:
        monkeypatch.setitem(sys.modules, "strands", None)
        cache = make_tool_cache()

        # Decoration must not trigger the dependency guard.
        @strands_cached_tool(cache, scope="global", ttl=300)
        async def fetch(query: str) -> object:
            return await stub_tool.arun(query=query)

        # The dependency is only required on first invocation/use.
        with pytest.raises(BackendNotAvailableError) as exc_info:
            await fetch(query="q")

        assert "memory-reuse[strands]" in str(exc_info.value)
        assert stub_tool.calls == 0


class TestCrewAILazyImport:
    """Req 10.3, 10.4: CrewAI imported only on first tool invocation."""

    @pytest.mark.asyncio
    async def test_dep_only_required_on_first_use(
        self, monkeypatch: pytest.MonkeyPatch, stub_tool: StubTool
    ) -> None:
        monkeypatch.setitem(sys.modules, "crewai", None)
        cache = make_tool_cache()

        # Decoration must not trigger the dependency guard.
        @crewai_cached_tool(cache, scope="global", ttl=300)
        async def fetch(query: str) -> object:
            return await stub_tool.arun(query=query)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            await fetch(query="q")

        assert "memory-reuse[crewai]" in str(exc_info.value)
        assert stub_tool.calls == 0


class TestAgentCoreLazyImport:
    """Req 10.3, 10.4: boto3 imported only on first backend construction."""

    def test_dep_only_required_on_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "boto3", None)

        # client=None forces the constructor to run ``_require_agentcore()``.
        with pytest.raises(BackendNotAvailableError) as exc_info:
            AgentCoreBackend(
                AgentCoreSettings(region="us-east-1", memory_id="mem-test"),
                client=None,
            )

        assert "memory-reuse[agentcore]" in str(exc_info.value)


class TestCoreCachesUnaffectedByAbsentOptionalDeps:
    """Req 10.5: exact/tool caches behave identically with optional deps absent."""

    @pytest.mark.asyncio
    async def test_exact_and_tool_round_trip_with_deps_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for dep in _OPTIONAL_DEPS:
            monkeypatch.setitem(sys.modules, dep, None)

        cache = make_tool_cache()

        key_parts = ["prompt", "hello"]

        # Miss before store.
        assert (
            await cache.lookup(key_parts, "hello", scope="global", scope_id=None, exact_only=True)
            is None
        )

        # Store then hit (exact-only path — no optional deps involved).
        await cache.store(
            key_parts, "hello", "answer", scope="global", scope_id=None, exact_only=True
        )
        assert (
            await cache.lookup(key_parts, "hello", scope="global", scope_id=None, exact_only=True)
            == "answer"
        )
