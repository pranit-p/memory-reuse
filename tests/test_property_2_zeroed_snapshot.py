"""Property 2: Snapshot starts and stays zeroed without events / when disabled.

Feature: observability-and-cost-analytics, Property 2.

*For any* tracker with no recorded events, and *for any* tracker with
``enabled=False`` regardless of recorded events, the snapshot reports
``hit_rate == 0.0`` and ``tokens_saved == cost_saved == latency_saved == 0``.

Validates: Requirements 1.7, 1.9.
"""

from __future__ import annotations

from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.analytics import AnalyticsTracker, CacheHitEvent, PricingConfig
from memory_reuse.stats import StatsTracker

_events = st.lists(
    st.builds(
        CacheHitEvent,
        tokens_in=st.integers(min_value=0, max_value=10_000),
        tokens_out=st.integers(min_value=0, max_value=10_000),
        latency_saved=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    ),
    max_size=30,
)


class TestProperty2ZeroedSnapshot:
    """Feature: observability-and-cost-analytics, Property 2.

    Snapshot starts and stays zeroed without events / when disabled.

    Validates: Requirements 1.7, 1.9
    """

    def test_no_events_is_zeroed(self) -> None:
        """A tracker with no recorded events reports zeroed totals."""
        snap = AnalyticsTracker(StatsTracker()).snapshot()
        assert snap.hit_rate == 0.0
        assert snap.tokens_saved == 0
        assert snap.cost_saved == Decimal("0")
        assert snap.latency_saved == 0.0

    @settings(max_examples=100)
    @given(events=_events)
    def test_disabled_ignores_events_and_hit_rate(self, events: list[CacheHitEvent]) -> None:
        """A disabled tracker records nothing and reports zeroed totals.

        Even when the shared StatsTracker has recorded hits (so its own hit rate
        is non-zero) and events are fed in, a disabled AnalyticsTracker reports a
        zeroed snapshot (Req 1.9).
        """
        stats = StatsTracker()
        stats.record_exact_hit()  # make the underlying hit rate non-zero
        tracker = AnalyticsTracker(
            stats,
            pricing=PricingConfig(input_token_price=1e-3, output_token_price=1e-3),
            enabled=False,
        )
        for event in events:
            tracker.record_hit_event(event)

        snap = tracker.snapshot()
        assert snap.hit_rate == 0.0
        assert snap.tokens_saved == 0
        assert snap.cost_saved == Decimal("0")
        assert snap.latency_saved == 0.0
