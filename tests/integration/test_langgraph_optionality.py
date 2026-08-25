"""Task 10.3: LangGraph-optionality smoke test.

These example tests (not property-based) verify that LangGraph stays a fully
optional, lazily-imported dependency:

* the core ``memory_reuse`` package imports cleanly without LangGraph installed;
* importing the LangGraph integration module does not import LangGraph at module
  load time;
* calling ``wrap_graph`` is what triggers the lazy import, raising a clear
  :class:`~memory_reuse.exceptions.BackendNotAvailableError` (naming the extra)
  when LangGraph is absent, rather than a low-level ``ImportError``.

LangGraph may well be installed in the test environment, so absence is
*simulated* two ways: a subprocess whose import machinery blocks ``langgraph``
(for the import-time assertions, which must run in a clean interpreter), and a
monkeypatched ``builtins.__import__`` (for the ``wrap_graph`` runtime guard).

Requirements: 7.1, 7.2, 14.4
"""

from __future__ import annotations

import builtins
import subprocess
import sys
import textwrap

import pytest

from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.exceptions import BackendNotAvailableError
from tests.conftest import StubGraph

# A meta-path finder that makes ``langgraph`` (and submodules) unimportable, so
# a subprocess simulates an environment where LangGraph is not installed even
# when it happens to be installed in the test venv.
_BLOCKER = textwrap.dedent("""
    import sys

    class _BlockLangGraph:
        def find_spec(self, name, path=None, target=None):
            if name == "langgraph" or name.startswith("langgraph."):
                raise ImportError("langgraph is not installed (simulated)")
            return None

    sys.meta_path.insert(0, _BlockLangGraph())
    # Drop anything already imported so the blocker actually bites.
    for _mod in [m for m in sys.modules if m == "langgraph" or m.startswith("langgraph.")]:
        del sys.modules[_mod]
    """)


def _run_without_langgraph(body: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a fresh interpreter with LangGraph imports blocked."""
    script = _BLOCKER + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


class TestCoreImportsWithoutLangGraph:
    """Requirement 7.1: core package imports without LangGraph installed."""

    def test_core_package_imports_cleanly(self) -> None:
        result = _run_without_langgraph("""
            import importlib.util
            # LangGraph must genuinely be unimportable in this subprocess.
            try:
                import langgraph  # noqa: F401
            except ImportError:
                pass
            else:
                raise AssertionError("langgraph should be blocked in this subprocess")

            import memory_reuse
            from memory_reuse import MemoryCache, CacheConfig  # noqa: F401

            print("CORE_OK")
            """)
        assert result.returncode == 0, result.stderr
        assert "CORE_OK" in result.stdout

    def test_integration_module_imports_without_langgraph(self) -> None:
        result = _run_without_langgraph("""
            import memory_reuse.integrations.langgraph as lg  # noqa: F401

            print("INTEGRATION_OK")
            """)
        assert result.returncode == 0, result.stderr
        assert "INTEGRATION_OK" in result.stdout


class TestLangGraphNotImportedAtLoadTime:
    """Requirements 7.2, 14.4: LangGraph is not imported until wrap_graph runs."""

    def test_langgraph_absent_from_sys_modules_after_core_import(self) -> None:
        result = _run_without_langgraph("""
            import sys
            import memory_reuse  # noqa: F401
            import memory_reuse.core  # noqa: F401
            import memory_reuse.integrations.langgraph  # noqa: F401

            assert "langgraph" not in sys.modules, (
                "langgraph was imported at module load time"
            )
            print("NOT_LOADED_OK")
            """)
        assert result.returncode == 0, result.stderr
        assert "NOT_LOADED_OK" in result.stdout

    def test_constructing_memory_cache_does_not_import_langgraph(self) -> None:
        result = _run_without_langgraph("""
            import sys
            from memory_reuse import CacheConfig, MemoryCache

            MemoryCache(CacheConfig(backend="memory"))
            assert "langgraph" not in sys.modules, (
                "constructing MemoryCache must not import langgraph"
            )
            print("CACHE_OK")
            """)
        assert result.returncode == 0, result.stderr
        assert "CACHE_OK" in result.stdout


class TestWrapGraphTriggersLazyImport:
    """Requirements 7.2, 7.3: wrap_graph triggers the lazy LangGraph import."""

    def test_wrap_graph_attempts_the_langgraph_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = MemoryCache(CacheConfig(backend="memory"))
        real_import = builtins.__import__
        seen: list[str] = []

        def recording_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "langgraph":
                seen.append(name)
                raise ImportError("no langgraph")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", recording_import)

        with pytest.raises(BackendNotAvailableError):
            cache.wrap_graph(StubGraph())

        # The lazy import is what wrap_graph attempted (proving it is deferred to
        # the call, not done at module load time).
        assert seen == ["langgraph"]

    def test_missing_langgraph_raises_named_error_not_importerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = MemoryCache(CacheConfig(backend="memory"))
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "langgraph":
                raise ImportError("no langgraph")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            cache.wrap_graph(StubGraph())

        # A clear, named error mentioning the extra rather than a raw ImportError.
        message = str(exc_info.value)
        assert "langgraph" in message
        assert "memory-reuse[langgraph]" in message
