"""Example tests: AgentCore cross-microVM sharing, connectivity, named error.

Task 5.5 (Reqs 8.2, 8.6, 8.7, 8.11): concrete example checks over the
:class:`~memory_reuse.backends.agentcore.AgentCoreBackend`, complementing the
Property 16-18 tests. All three examples run fully offline against the
dict-backed :class:`~tests.conftest.FakeAgentCoreService` /
:class:`~tests.conftest.FakeAgentCoreClient` scaffolding — no real network, LLM,
or AWS call.

The examples cover:

* **Cross-microVM sharing (Req 8.2):** two ``AgentCoreBackend`` instances
  constructed against the *same* fake service observe each other's writes, the
  way two AgentCore microVMs sharing one managed store would.
* **Connectivity (Reqs 8.6, 8.7):** when the shared service is marked
  unreachable, ``get`` / ``set`` (and friends) raise
  :exc:`~memory_reuse.exceptions.BackendConnectionError`, while ``ping`` returns
  ``False`` without raising.
* **Missing dependency (Req 8.11):** with ``boto3`` simulated absent, building a
  backend with ``client=None`` runs the lazy ``_require_agentcore()`` guard and
  raises :exc:`~memory_reuse.exceptions.BackendNotAvailableError` naming the
  ``memory-reuse[agentcore]`` extra rather than a raw ``ImportError``.
"""

from __future__ import annotations

import sys

import pytest

from memory_reuse.backends.agentcore import AgentCoreBackend, AgentCoreSettings
from memory_reuse.exceptions import BackendConnectionError, BackendNotAvailableError
from tests.conftest import FakeAgentCoreService


def _make_backend(service: FakeAgentCoreService, **overrides: object) -> AgentCoreBackend:
    """Build an ``AgentCoreBackend`` over a client bound to ``service``.

    Uses the fake client so the backend never touches ``boto3`` or the network.
    """
    from tests.conftest import FakeAgentCoreClient

    settings = AgentCoreSettings(region="us-east-1", memory_id="mem-test", **overrides)  # type: ignore[arg-type]
    return AgentCoreBackend(settings, client=FakeAgentCoreClient(service))


class TestCrossMicroVMSharing:
    """Req 8.2: a value set via one backend is read via another."""

    @pytest.mark.asyncio
    async def test_value_set_via_one_backend_is_read_via_the_other(
        self, fake_agentcore_service: FakeAgentCoreService
    ) -> None:
        # Two independent backends sharing the same managed store, modelling two
        # AgentCore microVMs pointed at one memory resource.
        backend_a = _make_backend(fake_agentcore_service)
        backend_b = _make_backend(fake_agentcore_service)

        await backend_a.set("shared-key", b"from-a")

        # The write made through backend_a is visible through backend_b.
        assert await backend_b.get("shared-key") == b"from-a"
        assert await backend_b.exists("shared-key") is True

    @pytest.mark.asyncio
    async def test_delete_via_one_backend_is_seen_by_the_other(
        self, fake_agentcore_service: FakeAgentCoreService
    ) -> None:
        backend_a = _make_backend(fake_agentcore_service)
        backend_b = _make_backend(fake_agentcore_service)

        await backend_a.set("k", b"v")
        assert await backend_b.get("k") == b"v"

        await backend_b.delete("k")
        # The delete made through backend_b is visible through backend_a.
        assert await backend_a.get("k") is None
        assert await backend_a.exists("k") is False

    @pytest.mark.asyncio
    async def test_backends_must_share_prefix_to_share_entries(
        self, fake_agentcore_service: FakeAgentCoreService
    ) -> None:
        # A backend with a different key_prefix namespaces its items away, so a
        # sibling using the default prefix does not observe them.
        default_backend = _make_backend(fake_agentcore_service)
        prefixed_backend = _make_backend(fake_agentcore_service, key_prefix="other:")

        await prefixed_backend.set("k", b"isolated")

        assert await default_backend.get("k") is None
        assert await prefixed_backend.get("k") == b"isolated"


class TestConnectivity:
    """Reqs 8.6, 8.7: unreachable service raises on ops; ping returns False."""

    @pytest.mark.asyncio
    async def test_get_raises_backend_connection_error_when_unreachable(
        self, fake_agentcore_service: FakeAgentCoreService
    ) -> None:
        backend = _make_backend(fake_agentcore_service)
        await backend.set("k", b"v")

        fake_agentcore_service.reachable = False

        with pytest.raises(BackendConnectionError):
            await backend.get("k")

    @pytest.mark.asyncio
    async def test_set_raises_backend_connection_error_when_unreachable(
        self, fake_agentcore_service: FakeAgentCoreService
    ) -> None:
        backend = _make_backend(fake_agentcore_service)

        fake_agentcore_service.reachable = False

        with pytest.raises(BackendConnectionError):
            await backend.set("k", b"v")

    @pytest.mark.asyncio
    async def test_delete_exists_flush_raise_when_unreachable(
        self, fake_agentcore_service: FakeAgentCoreService
    ) -> None:
        backend = _make_backend(fake_agentcore_service)

        fake_agentcore_service.reachable = False

        with pytest.raises(BackendConnectionError):
            await backend.delete("k")
        with pytest.raises(BackendConnectionError):
            await backend.exists("k")
        with pytest.raises(BackendConnectionError):
            await backend.flush()

    @pytest.mark.asyncio
    async def test_ping_returns_false_without_raising_when_unreachable(
        self, fake_agentcore_service: FakeAgentCoreService
    ) -> None:
        backend = _make_backend(fake_agentcore_service)

        # Reachable service pings True.
        assert await backend.ping() is True

        fake_agentcore_service.reachable = False

        # Unreachable service pings False and never raises (Req 8.7).
        assert await backend.ping() is False


class TestMissingDependency:
    """Req 8.11: absent boto3 surfaces a clear, named error at construction."""

    def test_construction_without_boto3_raises_named_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the lazy ``import boto3`` in ``_require_agentcore`` to fail even
        # if boto3 happens to be installed. A ``None`` entry in ``sys.modules``
        # makes ``import boto3`` raise ImportError.
        monkeypatch.setitem(sys.modules, "boto3", None)

        # client=None forces construction to call ``_require_agentcore()``.
        with pytest.raises(BackendNotAvailableError) as exc_info:
            AgentCoreBackend(
                AgentCoreSettings(region="us-east-1", memory_id="mem-test"),
                client=None,
            )

        message = str(exc_info.value)
        assert "memory-reuse[agentcore]" in message

    def test_construction_error_is_not_raw_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "boto3", None)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            AgentCoreBackend(
                AgentCoreSettings(region="us-east-1", memory_id="mem-test"),
                client=None,
            )

        # The caller must not see a low-level ImportError traceback.
        assert not isinstance(exc_info.value, ImportError)
