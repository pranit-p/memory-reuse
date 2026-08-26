"""Property 13: TTL expiry re-executes the tool body.

Feature: analytics-and-integrations, Property 13.

*For any* stored tool result and a TTL, once the TTL has elapsed the next call
with the same arguments is a miss and re-executes the tool body; before expiry a
second call with the same arguments is a hit that replays the stored value
without re-running the body.

Validates: Requirements 6.6, 7.9.

Both the Strands and the CrewAI ``cached_tool`` wrappers delegate to the shared
LangGraph decorator machinery, which stores tool results in the in-memory
backend with an absolute ``time.monotonic()``-based expiry. TTL expiry is
simulated exactly as the backend measures it: by advancing the ``monotonic``
clock the backend reads (``memory_reuse.backends.memory.time.monotonic``) past
the stored entry's expiry, so ``get`` treats the entry as absent and the next
call re-executes the tool body (``StubTool.calls`` increments) instead of
replaying a stale cached value.

The wrappers import their optional third-party dependency lazily inside the
returned async wrapper; ``_require_strands`` / ``_require_crewai`` are replaced
with no-ops so the property runs offline without ``strands`` / ``crewai``
installed. Both neutralisations are applied via context managers inside the test
body rather than the function-scoped ``monkeypatch`` fixture, because Hypothesis
does not reset function-scoped fixtures between generated inputs (which trips
``HealthCheck.function_scoped_fixture``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.integrations import crewai as crewai_integration
from memory_reuse.integrations import strands as strands_integration
from tests.conftest import StubTool, make_tool_cache, tool_arguments

# The two Phase 4 tool-cache wrappers under test. Each entry is
# ``(module, no-op-guard-attribute)`` so the shared property body can decorate
# with either wrapper and disable its lazy dependency guard.
_WRAPPERS = [
    pytest.param(strands_integration, "_require_strands", id="strands"),
    pytest.param(crewai_integration, "_require_crewai", id="crewai"),
]


class _FakeClock:
    """A movable stand-in for ``time.monotonic`` used to age cache entries.

    The in-memory backend records an entry's expiry as ``monotonic() + ttl`` and
    later treats it as expired once ``monotonic() >= expires_at``. Driving both
    reads through this fake clock lets a test advance time deterministically and
    precisely past a TTL boundary without sleeping.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@contextlib.contextmanager
def _guard_disabled(module: object, guard_name: str) -> Iterator[None]:
    """Temporarily replace a wrapper's lazy dependency guard with a no-op.

    The optional Strands / CrewAI packages are absent in the test environment,
    so the real guard would raise ``BackendNotAvailableError`` on first call.
    Using ``patch.object`` as a context manager (rather than the function-scoped
    ``monkeypatch`` fixture) keeps the substitution inside the test body, which
    Hypothesis re-enters for every generated example.
    """
    with patch.object(module, guard_name, lambda: None):
        yield


class TestProperty13TtlExpiry:
    """Feature: analytics-and-integrations, Property 13.

    TTL expiry re-executes the tool body.

    Validates: Requirements 6.6, 7.9
    """

    @pytest.mark.parametrize("module, guard_name", _WRAPPERS)
    @settings(max_examples=100)
    @given(args=tool_arguments, ttl=st.integers(min_value=1, max_value=3600))
    async def test_ttl_expiry_re_executes_tool_body(
        self,
        module: object,
        guard_name: str,
        args: dict[str, object],
        ttl: int,
    ) -> None:
        """A hit before expiry, then a miss (re-execution) after expiry.

        For arbitrary tool arguments and TTL, the first call is a miss that runs
        the body and stores the result; a second call before the TTL elapses is
        a hit that replays the stored value without re-running the body; and the
        first call after the TTL elapses is a miss again, so the body
        re-executes (its call counter increments) rather than replaying a stale
        cached value.
        """
        # Drive the backend's expiry clock through a controllable fake so TTL
        # boundaries are crossed deterministically, exactly as the backend
        # measures them. Both the guard no-op and the clock are applied via
        # context managers inside the test body (Hypothesis-safe).
        clock = _FakeClock()
        with (
            _guard_disabled(module, guard_name),
            patch("memory_reuse.backends.memory.time.monotonic", clock),
        ):
            cache = make_tool_cache()
            tool = StubTool()

            decorate = module.cached_tool(cache, ttl=ttl)  # type: ignore[attr-defined]

            @decorate
            async def run_tool(**kwargs: object) -> object:
                return tool.run(**kwargs)

            # First call: a miss that executes the body and stores the result.
            first = await run_tool(**args)
            assert tool.calls == 1

            # Second call before expiry: a hit that replays the stored value
            # without re-executing the body.
            second = await run_tool(**args)
            assert tool.calls == 1
            assert second == first

            # Advance the clock just past the TTL so the entry is now expired.
            clock.advance(ttl + 1)

            # Next call: the entry is treated as absent, so the body
            # re-executes rather than replaying a stale cached value.
            third = await run_tool(**args)
            assert tool.calls == 2
            assert third == first
