"""Property 7: Recording failures never propagate.

Feature: observability-and-cost-analytics, Property 7.

*For any* recorded event, if internal accumulation raises, ``record_hit_event``
swallows the error and a subsequent ``snapshot()`` is still returned — mirroring
the best-effort ``StatsTracker`` contract.

Validates: Requirements 1.8.
"""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.analytics import AnalyticsTracker, CacheHitEvent
from memory_reuse.stats import StatsTracker

_events = st.lists(
    st.builds(
        CacheHitEvent,
        tokens_in=st.integers(min_value=0, max_value=1000),
        tokens_out=st.integers(min_value=0, max_value=1000),
        latency_saved=st.floats(
            min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    ),
    min_size=1,
    max_size=20,
)


class TestProperty7RecordingFailures:
    """Feature: observability-and-cost-analytics, Property 7.

    Recording failures never propagate.

    Validates: Requirements 1.8
    """

    @settings(max_examples=100)
    @given(events=_events)
    def test_internal_failure_is_swallowed(self, events: list[CacheHitEvent]) -> None:
        """A raising internal accumulation never escapes record_hit_event."""
        tracker = AnalyticsTracker(StatsTracker())

        # Force the token-accumulation helper to raise on every event; the
        # error must be swallowed so recording never becomes fatal (Req 1.8).
        with patch.object(
            AnalyticsTracker,
            "_accumulate_tokens",
            side_effect=RuntimeError("boom"),
        ):
            for event in events:
                # Must not raise.
                tracker.record_hit_event(event)

        # A snapshot is still returned afterwards.
        snap = tracker.snapshot()
        assert snap is not None
        assert snap.hit_rate == 0.0
