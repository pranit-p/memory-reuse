"""Property 17: AgentCore TTL and absence semantics.

Feature: analytics-and-integrations, Property 17.

*For any* key and value, the AgentCore backend honours TTL and absence
semantics: a positive TTL makes the entry visible before it elapses and absent
(``get`` -> ``None``, ``exists`` -> ``False``) once it has elapsed (lazy
expiry); a missing or zero TTL persists the entry so a later ``get`` still
returns the value; ``get`` on a never-stored key returns ``None`` and ``exists``
returns ``False``; and ``delete`` on a missing key is a no-op that never raises,
with ``get``/``exists`` reporting the key as absent afterwards.

Validates: Requirements 8.3, 8.4, 8.8, 8.9.

The backend records a positive TTL as an *absolute* expiry using
``time.time()`` (``memory_reuse.backends.agentcore.time.time``) and filters
elapsed entries lazily on ``get`` / ``exists``. TTL elapse is simulated exactly
as the backend measures it, by advancing a controllable fake clock patched over
that time source past the stored entry's expiry, so no test ever sleeps. The
fake AgentCore service/client from ``tests/conftest.py`` provide the
deterministic, offline storage the backend talks to.

The clock substitution is applied via ``unittest.mock.patch`` as a context
manager inside the test body rather than the function-scoped ``monkeypatch``
fixture, because Hypothesis does not reset function-scoped fixtures between
generated inputs (which trips ``HealthCheck.function_scoped_fixture``).
"""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.backends.agentcore import AgentCoreBackend, AgentCoreSettings
from tests.conftest import FakeAgentCoreClient, FakeAgentCoreService, value_blobs

# Values are generated from the full 0..1_048_576-byte range the backend must
# round-trip, but capped small here so the TTL/absence property stays fast; the
# byte-round-trip boundary is exercised by Property 16.
_values = value_blobs(max_size=4096)

# Keys are non-empty and well under the 512-byte cap so key validation (a
# separate concern, Property 18) never interferes with the TTL/absence checks.
_keys = st.text(min_size=1, max_size=64)


class _FakeClock:
    """A movable stand-in for ``time.time`` used to age backend entries.

    The backend records an entry's expiry as ``time() + ttl`` and later treats
    it as expired once ``time() >= expires_at``. Driving that read through this
    fake clock lets a test advance wall-clock time deterministically across a
    TTL boundary without sleeping.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _make_backend(service: FakeAgentCoreService) -> AgentCoreBackend:
    """Build an AgentCoreBackend wired to an injected fake client."""
    return AgentCoreBackend(
        AgentCoreSettings(region="us-east-1", memory_id="mem-test"),
        client=FakeAgentCoreClient(service),
    )


class TestProperty17AgentCoreTtlAbsence:
    """Feature: analytics-and-integrations, Property 17.

    AgentCore TTL and absence semantics.

    Validates: Requirements 8.3, 8.4, 8.8, 8.9
    """

    @settings(max_examples=100)
    @given(key=_keys, value=_values, ttl=st.integers(min_value=1, max_value=3600))
    async def test_positive_ttl_visible_then_absent_after_elapse(
        self, key: str, value: bytes, ttl: int
    ) -> None:
        """A TTL>0 entry is visible before elapse and absent after (Req 8.3).

        Before the TTL elapses ``get`` returns the value and ``exists`` is
        ``True``; once the clock advances past the absolute expiry the same
        ``get`` returns ``None`` and ``exists`` returns ``False`` (lazy expiry).
        """
        clock = _FakeClock()
        with patch("memory_reuse.backends.agentcore.time.time", clock):
            backend = _make_backend(FakeAgentCoreService())
            await backend.set(key, value, ttl=ttl)

            # Before elapse: the entry is present and returns the exact value.
            assert await backend.get(key) == value
            assert await backend.exists(key) is True

            # Advance just past the absolute expiry.
            clock.advance(ttl + 1)

            # After elapse: the entry is treated as absent for both reads.
            assert await backend.get(key) is None
            assert await backend.exists(key) is False

    @settings(max_examples=100)
    @given(
        key=_keys,
        value=_values,
        ttl=st.sampled_from([None, 0]),
        later=st.integers(min_value=1, max_value=10_000_000),
    )
    async def test_no_or_zero_ttl_persists(
        self, key: str, value: bytes, ttl: int | None, later: int
    ) -> None:
        """A missing or zero TTL persists the entry indefinitely (Req 8.3).

        With no TTL (``None``) or a zero TTL, advancing the clock arbitrarily far
        leaves the entry retained: ``get`` still returns the stored value and
        ``exists`` is still ``True``.
        """
        clock = _FakeClock()
        with patch("memory_reuse.backends.agentcore.time.time", clock):
            backend = _make_backend(FakeAgentCoreService())
            await backend.set(key, value, ttl=ttl)

            # Advance the clock far into the future; a no/zero-TTL entry stays.
            clock.advance(later)

            assert await backend.get(key) == value
            assert await backend.exists(key) is True

    @settings(max_examples=100)
    @given(key=_keys)
    async def test_absent_key_get_none_exists_false(self, key: str) -> None:
        """``get`` on an absent key is ``None`` and ``exists`` is ``False`` (Req 8.4, 8.9).

        A key that was never stored reports as absent through both read paths.
        """
        clock = _FakeClock()
        with patch("memory_reuse.backends.agentcore.time.time", clock):
            backend = _make_backend(FakeAgentCoreService())

            assert await backend.get(key) is None
            assert await backend.exists(key) is False

    @settings(max_examples=100)
    @given(key=_keys)
    async def test_delete_missing_key_is_noop(self, key: str) -> None:
        """``delete`` on a missing key is a no-op that never raises (Req 8.8).

        Deleting a never-stored key completes without raising, and the key still
        reports as absent through ``get`` and ``exists`` afterwards.
        """
        clock = _FakeClock()
        with patch("memory_reuse.backends.agentcore.time.time", clock):
            backend = _make_backend(FakeAgentCoreService())

            # No entry exists yet; delete must complete silently.
            await backend.delete(key)

            assert await backend.get(key) is None
            assert await backend.exists(key) is False

    @settings(max_examples=100)
    @given(key=_keys, value=_values)
    async def test_delete_present_key_makes_absent(self, key: str, value: bytes) -> None:
        """After ``delete`` a stored key is absent (Req 8.8).

        A stored entry is visible until deleted; once deleted ``get`` returns
        ``None`` and ``exists`` returns ``False``.
        """
        clock = _FakeClock()
        with patch("memory_reuse.backends.agentcore.time.time", clock):
            backend = _make_backend(FakeAgentCoreService())
            await backend.set(key, value)

            # Present before delete.
            assert await backend.get(key) == value
            assert await backend.exists(key) is True

            await backend.delete(key)

            # Absent after delete.
            assert await backend.get(key) is None
            assert await backend.exists(key) is False
