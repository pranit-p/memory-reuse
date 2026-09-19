"""Property 5: No pricing means zero cost but preserved token tracking.

Feature: observability-and-cost-analytics, Property 5.

*For any* sequence of token-attributed events recorded with no pricing
configuration, ``cost_saved`` is exactly ``0`` while ``tokens_saved`` equals the
summed non-negative token counts.

Validates: Requirements 2.3, 2.4.
"""

from __future__ import annotations

from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.analytics import AnalyticsTracker, CacheHitEvent
from memory_reuse.stats import StatsTracker

_tokens = st.integers(min_value=0, max_value=10_000)
_events = st.lists(
    st.builds(CacheHitEvent, tokens_in=_tokens, tokens_out=_tokens),
    max_size=40,
)


class TestProperty5NoPricing:
    """Feature: observability-and-cost-analytics, Property 5.

    No pricing means zero cost but preserved token tracking.

    Validates: Requirements 2.3, 2.4
    """

    @settings(max_examples=100)
    @given(events=_events)
    def test_no_pricing_zero_cost_tokens_tracked(self, events: list[CacheHitEvent]) -> None:
        """Without pricing, cost_saved stays 0 while tokens_saved accumulates."""
        tracker = AnalyticsTracker(StatsTracker(), pricing=None)
        expected_tokens = 0
        for event in events:
            tracker.record_hit_event(event)
            expected_tokens += (event.tokens_in or 0) + (event.tokens_out or 0)

        snap = tracker.snapshot()
        assert snap.cost_saved == Decimal("0")
        assert snap.tokens_saved == expected_tokens
        assert snap.currency == "USD"
