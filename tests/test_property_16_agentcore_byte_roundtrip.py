"""Property 16: AgentCore byte round-trip.

Feature: analytics-and-integrations, Property 16.

*For any* byte blob in the 0..1_048_576-byte range, storing it via
``AgentCoreBackend.set`` and reading it back via ``AgentCoreBackend.get``
returns exactly the same bytes. This exercises the raw-bytes round-trip
contract (Req 8.5): values are base64-encoded on write and decoded on read, so
any supported payload — including the empty blob and up to 1 MiB — survives the
transport unchanged.

Validates: Requirements 8.1, 8.5.

The backend is constructed against an in-process, dict-backed
:class:`~tests.conftest.FakeAgentCoreClient` (injected via the ``client=``
parameter), so the property runs fully offline: no real network, LLM, or AWS
call. ``value_blobs()`` biases toward small blobs while still reaching the 1 MiB
boundary sparsely, so ``max_examples=100`` stays fast without losing coverage of
the size range.
"""

from __future__ import annotations

from hypothesis import given, settings

from memory_reuse.backends.agentcore import AgentCoreBackend, AgentCoreSettings
from tests.conftest import FakeAgentCoreClient, FakeAgentCoreService, value_blobs


class TestProperty16AgentCoreByteRoundTrip:
    """Feature: analytics-and-integrations, Property 16.

    AgentCore byte round-trip.

    Validates: Requirements 8.1, 8.5
    """

    @settings(max_examples=100)
    @given(blob=value_blobs())
    async def test_set_then_get_returns_identical_bytes(self, blob: bytes) -> None:
        """set(key, blob) then get(key) returns exactly the same bytes.

        For an arbitrary byte blob across the whole 0..1_048_576-byte supported
        range, the value read back is byte-for-byte identical to the value
        stored — the raw-bytes round-trip guarantee (Req 8.5) over the
        ``AbstractBackend`` interface the backend implements (Req 8.1).
        """
        # Fresh service + client per example so each round-trip is isolated.
        service = FakeAgentCoreService()
        client = FakeAgentCoreClient(service)
        backend = AgentCoreBackend(
            AgentCoreSettings(region="us-east-1", memory_id="mem-test"),
            client=client,
        )

        key = "round-trip-key"
        await backend.set(key, blob)
        result = await backend.get(key)

        assert result == blob
