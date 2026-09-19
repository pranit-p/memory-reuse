"""Property 1: Saved-totals are the sum of non-negative attributions.

Feature: observability-and-cost-analytics, Property 1.

*For any* sequence of recorded hit events, ``tokens_saved`` equals the sum of
each event's non-negative input and output token counts, and ``latency_saved``
equals the sum of each event's non-negative latency, with negative (and None)
fields ignored.

Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.analytics import AnalyticsTracker, CacheHitEvent
from memory_reuse.stats import StatsTracker

# Token counts span negatives (ignored), None (absent), and non-negative ints.
_tokens = st.one_of(st.none(), st.integers(min_value=-1000, max_value=10_000))
# Latency spans negatives (ignored), None (absent), and non-negative floats.
_latency = st.one_of(
    st.none(),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)

_events = st.lists(
    st.builds(
        CacheHitEvent,
        tokens_in=_tokens,
        tokens_out=_tokens,
        latency_saved=_latency,
    ),
    max_size=30,
)


def _expected_tokens(events: list[CacheHitEvent]) -> int:
    total = 0
    for e in events:
        for tok in (e.tokens_in, e.tokens_out):
            if isinstance(tok, int) and not isinstance(tok, bool) and tok > 0:
                total += tok
    return total


def _expected_latency(events: list[CacheHitEvent]) -> float:
    total = 0.0
    for e in events:
        lat = e.latency_saved
        if isinstance(lat, (int, float)) and not isinstance(lat, bool) and lat > 0:
            total += float(lat)
    return total


class TestProperty1SavedTotals:
    """Feature: observability-and-cost-analytics, Property 1.

    Saved-totals are the sum of non-negative attributions.

    Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6
    """

    @settings(max_examples=100)
    @given(events=_events)
    def test_tokens_and_latency_sum_non_negative(self, events: list[CacheHitEvent]) -> None:
        """tokens_saved and latency_saved sum only the non-negative attributions."""
        tracker = AnalyticsTracker(StatsTracker())
        for event in events:
            tracker.record_hit_event(event)

        snap = tracker.snapshot()
        assert snap.tokens_saved == _expected_tokens(events)
        assert math.isclose(
            snap.latency_saved, _expected_latency(events), rel_tol=1e-9, abs_tol=1e-9
        )
