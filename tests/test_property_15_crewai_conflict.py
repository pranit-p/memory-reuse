"""Property-based test for the CrewAI ``exact_only`` + ``semantic`` conflict.

Exercises the decoration-time validation guard in
``memory_reuse.integrations.crewai.cached_tool`` (Task 3.6 of the
analytics-and-integrations spec). Uses the Phase 4 ``make_tool_cache`` helper
from ``tests/conftest.py``; no real LLM, network, or CrewAI install is involved
(the guard fires before any tool call, so CrewAI never needs importing).

* **Property 15 — CrewAI rejects the exact_only+semantic conflict:** for any
  scope / ttl parameter combination, calling ``cached_tool`` with both
  ``exact_only=True`` and ``semantic=True`` raises ``ConfigurationError`` at
  decoration time (before any tool is defined or called); whenever the two flags
  are not both ``True`` the decorator is created without raising
  ``ConfigurationError``.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.exceptions import ConfigurationError
from memory_reuse.integrations.crewai import cached_tool
from tests.conftest import make_tool_cache

# ---------------------------------------------------------------------------
# Hypothesis strategies: the scope / ttl parameter space the guard must ignore.
# ---------------------------------------------------------------------------

# All valid scopes the decorator accepts.
_scopes = st.sampled_from(["global", "user", "session"])

# A broad spread of ttl values (including 0 and large values); the conflict
# guard must fire regardless of ttl.
_ttls = st.integers(min_value=0, max_value=86_400)


# ---------------------------------------------------------------------------
# Property 15: CrewAI rejects the exact_only+semantic conflict
# ---------------------------------------------------------------------------


class TestProperty15CrewaiConflict:
    """Feature: analytics-and-integrations, Property 15.

    CrewAI rejects the exact_only+semantic conflict.

    Validates: Requirements 7.8
    """

    @settings(max_examples=100)
    @given(scope=_scopes, ttl=_ttls)
    def test_both_flags_true_raises_at_decoration_time(self, scope: str, ttl: int) -> None:
        """Both ``exact_only`` and ``semantic`` true is rejected up front.

        For any scope / ttl combination, constructing the decorator with both
        flags set raises ``ConfigurationError`` at decoration time — before any
        tool function is defined or called (Req 7.8).
        """
        cache = make_tool_cache()

        with pytest.raises(ConfigurationError):
            cached_tool(
                cache,
                scope=scope,  # type: ignore[arg-type]
                ttl=ttl,
                exact_only=True,
                semantic=True,
            )

    @settings(max_examples=100)
    @given(
        scope=_scopes,
        ttl=_ttls,
        exact_only=st.booleans(),
        semantic=st.booleans(),
    )
    def test_not_both_true_does_not_raise_conflict(
        self, scope: str, ttl: int, exact_only: bool, semantic: bool
    ) -> None:
        """Whenever the flags are not both true, no conflict error is raised.

        Only the ``exact_only=True`` + ``semantic=True`` combination is
        contradictory; every other combination must build the decorator without
        raising ``ConfigurationError`` (Req 7.8).
        """
        cache = make_tool_cache()

        # Constrain to the "not both true" region of the flag space.
        if exact_only and semantic:
            exact_only = False

        decorator = cached_tool(
            cache,
            scope=scope,  # type: ignore[arg-type]
            ttl=ttl,
            exact_only=exact_only,
            semantic=semantic,
        )
        assert callable(decorator)
