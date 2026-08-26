"""Property 18: AgentCore key validation rejects empty/oversized keys.

Feature: analytics-and-integrations, Property 18.

*For any* value bytes and *for any* invalid key -- the empty string or a key
whose UTF-8 encoding exceeds 512 bytes -- ``await backend.set(key, value)``
raises ``ValueError`` and stores nothing, and it does so *before any network
call* (the shared fake service's store stays empty / no ``put_item`` is ever
recorded). Conversely, *for any* valid key (1..512 UTF-8 bytes) ``set`` does not
raise on key-validation grounds.

Validates: Requirements 8.10.

The backend validates keys in :meth:`AgentCoreBackend._validate_key`, which
rejects an empty key and any key whose ``key.encode("utf-8")`` length exceeds
``_MAX_KEY_BYTES`` (512). Because validation runs at the top of ``set`` before
the ``put_item`` transport call, an injected :class:`FakeAgentCoreClient` (whose
store starts empty) must remain untouched on rejection -- proving no network
call occurred. Oversized keys are generated both from long ASCII text and from
multibyte characters whose byte length pushes over 512 while the character count
stays smaller, exercising the byte-length (not character-length) boundary.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.backends.agentcore import (
    _MAX_KEY_BYTES,
    AgentCoreBackend,
    AgentCoreSettings,
)
from tests.conftest import FakeAgentCoreClient, FakeAgentCoreService

# Values kept modest here: this property is about key validation, not value
# round-tripping (that is Property 16), so small payloads keep examples fast.
value_bytes = st.binary(min_size=0, max_size=256)


def _make_backend() -> tuple[AgentCoreBackend, FakeAgentCoreService]:
    """Build a backend over a fresh, empty, reachable fake service.

    Returns the backend and the shared service so a test can assert the
    service's store stayed empty after a rejected ``set`` (no network call).
    """
    service = FakeAgentCoreService()
    client = FakeAgentCoreClient(service)
    backend = AgentCoreBackend(
        AgentCoreSettings(region="us-east-1", memory_id="mem"), client=client
    )
    return backend, service


# Oversized keys: long single-byte ASCII whose byte length exceeds 512.
oversized_ascii_keys = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=_MAX_KEY_BYTES + 1,
    max_size=_MAX_KEY_BYTES + 64,
)

# Oversized keys via multibyte characters: each "\U0001F600" is 4 UTF-8 bytes,
# so >=129 of them exceed 512 bytes while the character count stays far smaller,
# exercising the byte-length (not character-length) boundary.
oversized_multibyte_keys = st.integers(min_value=129, max_value=200).map(lambda n: "\U0001f600" * n)

oversized_keys = st.one_of(oversized_ascii_keys, oversized_multibyte_keys)

# Valid keys: 1..512 UTF-8 bytes. ASCII keeps byte length == char length so the
# 1..512 bound is exact; the boundary key of exactly 512 bytes is included.
valid_keys = st.one_of(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=_MAX_KEY_BYTES),
    st.just("k" * _MAX_KEY_BYTES),
)


class TestProperty18AgentCoreKeyValidation:
    """Feature: analytics-and-integrations, Property 18.

    AgentCore key validation rejects empty/oversized keys.

    Validates: Requirements 8.10
    """

    @settings(max_examples=100)
    @given(value=value_bytes)
    async def test_empty_key_rejected_before_network(self, value: bytes) -> None:
        """An empty key raises ``ValueError`` and stores nothing.

        The rejection must precede any transport call, so the shared fake
        service store stays empty.
        """
        backend, service = _make_backend()

        with pytest.raises(ValueError):
            await backend.set("", value)

        assert service.store == {}

    @settings(max_examples=100)
    @given(key=oversized_keys, value=value_bytes)
    async def test_oversized_key_rejected_before_network(self, key: str, value: bytes) -> None:
        """A key over 512 UTF-8 bytes raises ``ValueError`` and stores nothing.

        Covers both long ASCII keys and multibyte keys whose byte length (not
        character count) crosses the 512-byte boundary. The rejection precedes
        any transport call, so the fake service store stays empty.
        """
        # Precondition the generators guarantee, asserted for clarity.
        assert len(key.encode("utf-8")) > _MAX_KEY_BYTES

        backend, service = _make_backend()

        with pytest.raises(ValueError):
            await backend.set(key, value)

        assert service.store == {}

    @settings(max_examples=100)
    @given(key=valid_keys, value=value_bytes)
    async def test_valid_key_not_rejected_on_validation(self, key: str, value: bytes) -> None:
        """A 1..512-byte key does not raise on validation grounds and is stored.

        For a valid key the reachable fake service accepts the write, so exactly
        one entry lands in the store -- confirming ``set`` proceeded past
        validation to the transport call.
        """
        # Precondition the generators guarantee, asserted for clarity.
        encoded_len = len(key.encode("utf-8"))
        assert 1 <= encoded_len <= _MAX_KEY_BYTES

        backend, service = _make_backend()

        await backend.set(key, value)

        assert len(service.store) == 1
