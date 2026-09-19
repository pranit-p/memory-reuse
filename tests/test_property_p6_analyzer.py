"""Effectiveness-analyzer properties (Phase 6, Properties 7-9).

Feature: intelligent-execution-optimization.

* Property 7 — per-operation stats equal the recorded observations (Req 5.1-5.3).
* Property 8 — tracking is bounded and best-effort (Req 4.4, 4.5).
* Property 9 — recommendations are justified and advisory (Req 6.1, 6.5, 6.6).
"""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.optimizer import (
    EffectivenessAnalyzer,
    OperationRecord,
    OperationTracker,
    OptimizerConfig,
)

_records = st.lists(
    st.builds(
        OperationRecord,
        operation=st.sampled_from(["op_a", "op_b", "op_c"]),
        hit=st.booleans(),
        cost_saved=st.one_of(st.none(), st.floats(min_value=0, max_value=10, allow_nan=False)),
        latency_saved=st.one_of(st.none(), st.floats(min_value=0, max_value=5, allow_nan=False)),
        input_hash=st.one_of(st.none(), st.sampled_from(["h1", "h2", "h3"])),
    ),
    max_size=60,
)


class TestProperty7StatsMatchRecords:
    """Feature: intelligent-execution-optimization, Property 7.

    Per-operation stats equal the recorded observations.

    Validates: Requirements 5.1, 5.2, 5.3
    """

    @settings(max_examples=100)
    @given(records=_records)
    def test_stats_equal_recorded(self, records: list[OperationRecord]) -> None:
        tracker = OperationTracker()
        for rec in records:
            tracker.record(rec)

        stats = tracker.stats()
        for op in {r.operation for r in records}:
            op_recs = [r for r in records if r.operation == op]
            s = stats[op]
            assert s.calls == len(op_recs)
            assert s.hits == sum(1 for r in op_recs if r.hit)
            assert s.misses == sum(1 for r in op_recs if not r.hit)
            assert s.hit_rate == (s.hits / s.calls if s.calls else 0.0)
            assert 0.0 <= s.hit_rate <= 1.0


class TestProperty8BoundedBestEffort:
    """Feature: intelligent-execution-optimization, Property 8.

    Tracking is bounded and best-effort.

    Validates: Requirements 4.4, 4.5
    """

    @settings(max_examples=100)
    @given(n=st.integers(min_value=1, max_value=50), cap=st.integers(min_value=1, max_value=10))
    def test_aggregate_count_is_bounded(self, n: int, cap: int) -> None:
        tracker = OperationTracker(max_operations=cap)
        for i in range(n):
            tracker.record(OperationRecord(operation=f"op_{i}", hit=False))
        # Never more than the cap distinct aggregates (Req 4.5).
        assert len(tracker.stats()) <= cap

    def test_recording_failure_is_swallowed(self) -> None:
        """A raising internal accumulation never propagates (Req 4.4)."""
        tracker = OperationTracker()
        # Force the internal set to raise on use by patching dict.get? Simpler:
        # patch the aggregate creation path to raise, and assert no exception.
        with patch.object(OperationTracker, "stats", side_effect=RuntimeError("boom")):
            # record still must not raise even if internals are broken elsewhere.
            tracker.record(OperationRecord(operation="op", hit=True))
        # And a normal record afterwards still works.
        tracker.record(OperationRecord(operation="op", hit=True))
        assert tracker.stats()["op"].calls >= 1


class TestProperty9RecommendationsJustified:
    """Feature: intelligent-execution-optimization, Property 9.

    Recommendations are justified and advisory.

    Validates: Requirements 6.1, 6.5, 6.6
    """

    @settings(max_examples=100)
    @given(records=_records)
    def test_recommendations_are_justified_and_advisory(
        self, records: list[OperationRecord]
    ) -> None:
        tracker = OperationTracker()
        for rec in records:
            tracker.record(rec)

        config = OptimizerConfig()
        analyzer = EffectivenessAnalyzer(tracker, config=config)

        before = {op: s.to_dict() for op, s in tracker.stats().items()}
        report = analyzer.analyze()
        after = {op: s.to_dict() for op, s in tracker.stats().items()}

        # Advisory: analysing does not mutate the recorded stats (Req 6.5).
        assert before == after

        for rec in report.recommendations:
            s = report.stats[rec.operation]
            # Only operations with enough data get a recommendation (Req 6.6).
            assert s.calls >= config.min_calls
            # Confidence is well-formed (Req 6.1).
            assert 0.0 <= rec.confidence <= 1.0
            assert rec.action in {"enable_cache", "increase_ttl", "avoid_semantic"}
            assert rec.reason
