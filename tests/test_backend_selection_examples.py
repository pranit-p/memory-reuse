"""Example tests: backend selection via config + ``from_env`` end-to-end.

Task 6.5 (Reqs 9.2, 9.3, 9.4, 9.5, 9.6): concrete example checks over how
:class:`~memory_reuse.core.MemoryCache` selects a storage backend from a
:class:`~memory_reuse.config.CacheConfig`, complementing the Property 19/20
tests. All examples run fully offline — no real network, LLM, or AWS call.

The examples cover:

* **AgentCore selection with the dependency present (Req 9.2, 9.6):** with a
  fake ``boto3`` injected via ``_require_agentcore``, ``backend="agentcore"``
  wires an :class:`~memory_reuse.backends.agentcore.AgentCoreBackend` for the
  cache.
* **AgentCore selection with the dependency absent (Req 9.6):** with ``boto3``
  simulated absent, selecting ``backend="agentcore"`` raises
  :exc:`~memory_reuse.exceptions.BackendNotAvailableError` naming the
  ``memory-reuse[agentcore]`` extra rather than a raw ``ImportError``.
* **End-to-end ``from_env`` (Req 9.3, 9.4):** ``MemoryCache.from_env`` builds a
  cache from ``MEMORY_REUSE_*`` variables, and raises
  :exc:`~memory_reuse.exceptions.ConfigurationError` naming a missing AgentCore
  setting.
* **Default unchanged (Req 9.5):** the bare ``CacheConfig()`` default stays
  ``"memory"`` and builds an :class:`~memory_reuse.backends.memory.InMemoryBackend`.

The ``from_env`` *unit* tests for parsing individual variables live in
``tests/unit/test_config.py`` (class ``TestFromEnvAgentCore``); this file focuses
on the ``MemoryCache`` backend-selection path plus a couple of end-to-end
``from_env`` -> ``MemoryCache`` checks.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from memory_reuse.backends.agentcore import AgentCoreBackend
from memory_reuse.backends.memory import InMemoryBackend
from memory_reuse.config import CacheConfig
from memory_reuse.core import MemoryCache
from memory_reuse.exceptions import BackendNotAvailableError, ConfigurationError


class _FakeAgentCoreClient:
    """A trivial stand-in returned by the fake ``boto3.client`` call.

    Its only job is to be *something* the ``AgentCoreBackend`` can hold as its
    client; no method is exercised by these construction-time selection tests.
    """


class _FakeBoto3:
    """A fake ``boto3`` module whose ``.client(...)`` returns a dummy client.

    Injected in place of the real ``boto3`` via ``_require_agentcore`` so the
    ``case "agentcore"`` construction path runs without ``boto3`` installed and
    without any AWS call. Records the last ``client`` call for optional
    assertions.
    """

    def __init__(self) -> None:
        self.client_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def client(self, *args: Any, **kwargs: Any) -> _FakeAgentCoreClient:
        self.client_calls.append((args, kwargs))
        return _FakeAgentCoreClient()


class TestAgentCoreSelectionDependencyPresent:
    """Req 9.2, 9.6: with the dependency present, selection builds the backend."""

    def test_backend_agentcore_builds_agentcore_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Real boto3 may not be installed, so make the backend's lazy import
        # guard return a fake boto3 whose ``.client(...)`` yields a dummy client.
        fake_boto3 = _FakeBoto3()
        monkeypatch.setattr(
            "memory_reuse.backends.agentcore._require_agentcore",
            lambda: fake_boto3,
        )

        cache = MemoryCache(
            CacheConfig(
                backend="agentcore",
                agentcore_region="us-east-1",
                agentcore_memory_id="mem-1",
            )
        )

        # The selected backend is the AgentCore one, shared by exact + tool caches.
        assert isinstance(cache._backend, AgentCoreBackend)
        assert cache.exact._backend is cache._backend
        assert cache.tool._backend is cache._backend

        # The lazy guard was used to build the client from the configured region.
        assert fake_boto3.client_calls == [(("bedrock-agentcore",), {"region_name": "us-east-1"})]


class TestAgentCoreSelectionDependencyAbsent:
    """Req 9.6: with boto3 absent, selection raises a clear, named error."""

    def test_backend_agentcore_raises_named_error_when_boto3_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A ``None`` entry in ``sys.modules`` makes ``import boto3`` raise
        # ImportError even if boto3 happens to be installed.
        monkeypatch.setitem(sys.modules, "boto3", None)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            MemoryCache(
                CacheConfig(
                    backend="agentcore",
                    agentcore_region="us-east-1",
                    agentcore_memory_id="mem-1",
                )
            )

        message = str(exc_info.value)
        assert "memory-reuse[agentcore]" in message
        # The caller must not see a low-level ImportError traceback.
        assert not isinstance(exc_info.value, ImportError)


class TestFromEnvEndToEnd:
    """Req 9.3, 9.4: ``MemoryCache.from_env`` builds from env / raises on gaps."""

    def test_from_env_builds_agentcore_cache_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMORY_REUSE_BACKEND", "agentcore")
        monkeypatch.setenv("MEMORY_REUSE_AGENTCORE_REGION", "eu-west-1")
        monkeypatch.setenv("MEMORY_REUSE_AGENTCORE_MEMORY_ID", "mem-env")

        fake_boto3 = _FakeBoto3()
        monkeypatch.setattr(
            "memory_reuse.backends.agentcore._require_agentcore",
            lambda: fake_boto3,
        )

        cache = MemoryCache.from_env()

        # from_env parsed the region/memory id and the backend was built from them.
        assert isinstance(cache._backend, AgentCoreBackend)
        assert fake_boto3.client_calls == [(("bedrock-agentcore",), {"region_name": "eu-west-1"})]

    def test_from_env_raises_when_region_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_REUSE_BACKEND", "agentcore")
        monkeypatch.delenv("MEMORY_REUSE_AGENTCORE_REGION", raising=False)
        monkeypatch.setenv("MEMORY_REUSE_AGENTCORE_MEMORY_ID", "mem-env")

        with pytest.raises(ConfigurationError) as exc_info:
            MemoryCache.from_env()

        # The error names the missing setting so the developer can fix it.
        assert "MEMORY_REUSE_AGENTCORE_REGION" in str(exc_info.value)

    def test_from_env_raises_when_memory_id_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_REUSE_BACKEND", "agentcore")
        monkeypatch.setenv("MEMORY_REUSE_AGENTCORE_REGION", "eu-west-1")
        monkeypatch.delenv("MEMORY_REUSE_AGENTCORE_MEMORY_ID", raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            MemoryCache.from_env()

        assert "MEMORY_REUSE_AGENTCORE_MEMORY_ID" in str(exc_info.value)


class TestDefaultBackendUnchanged:
    """Req 9.5: the default stays ``"memory"`` and builds the in-memory backend."""

    def test_default_config_backend_is_memory(self) -> None:
        assert CacheConfig().backend == "memory"

    def test_default_cache_uses_in_memory_backend(self) -> None:
        cache = MemoryCache(CacheConfig())

        assert isinstance(cache._backend, InMemoryBackend)
        # Selecting the default must not have imported the AgentCore dependency.
        assert "boto3" not in sys.modules
