"""Operation recording and effectiveness analysis (Phase 6, Req 4, 5, 6).

:class:`OperationTracker` accumulates :class:`~memory_reuse.optimizer.records.OperationRecord`
observations into bounded per-operation aggregates, best-effort and never fatal.
:class:`EffectivenessAnalyzer` turns those aggregates into an
:class:`EffectivenessReport` — per-operation statistics plus advisory
recommendations. Neither touches the existing ``StatsTracker`` /
``AnalyticsTracker`` counters (Req 5.5).
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from decimal import Decimal

from memory_reuse.optimizer.recommendations import (
    OptimizerConfig,
    Recommendation,
    recommend,
)
from memory_reuse.optimizer.records import OperationRecord, OperationStats

# Cap on distinct input fingerprints tracked per operation, bounding memory
# while still giving a useful repeat-ratio estimate.
_MAX_INPUTS_PER_OP = 512


class _Aggregate:
    """Mutable per-operation accumulator (internal to the tracker)."""

    __slots__ = (
        "calls",
        "hits",
        "misses",
        "total_cost",
        "total_latency",
        "inputs",
        "cached",
        "volatile",
    )

    def __init__(self) -> None:
        self.calls = 0
        self.hits = 0
        self.misses = 0
        self.total_cost = Decimal("0")
        self.total_latency = 0.0
        self.inputs: set[str] = set()
        self.cached = False
        self.volatile = False


class OperationTracker:
    """Bounded, best-effort recorder of per-operation observations (Req 4).

    Args:
        enabled: When ``False``, :meth:`record` is a no-op and :meth:`stats`
            returns an empty mapping (Req 4.3).
        max_operations: Maximum number of distinct operation aggregates retained.
            Once reached, observations for *new* operations are dropped rather
            than evicting existing ones, so memory never grows without bound
            (Req 4.5).
    """

    def __init__(self, *, enabled: bool = True, max_operations: int = 1000) -> None:
        self._enabled = enabled
        self._max_operations = max_operations
        self._aggregates: dict[str, _Aggregate] = {}
        self._lock = threading.Lock()

    def record(self, record: OperationRecord) -> None:
        """Record one observation (best-effort, bounded, no-op when disabled)."""
        if not self._enabled:
            return
        with contextlib.suppress(Exception), self._lock:
            agg = self._aggregates.get(record.operation)
            if agg is None:
                if len(self._aggregates) >= self._max_operations:
                    # Bounded: drop observations for new operations at capacity.
                    return
                agg = _Aggregate()
                self._aggregates[record.operation] = agg

            agg.calls += 1
            if record.hit:
                agg.hits += 1
            else:
                agg.misses += 1
            if isinstance(record.cost_saved, (int, float)) and record.cost_saved > 0:
                agg.total_cost += Decimal(str(record.cost_saved))
            if isinstance(record.latency_saved, (int, float)) and record.latency_saved > 0:
                agg.total_latency += float(record.latency_saved)
            if record.input_hash is not None and len(agg.inputs) < _MAX_INPUTS_PER_OP:
                agg.inputs.add(record.input_hash)
            # Sticky flags: once an operation is seen as cached/volatile it stays.
            agg.cached = agg.cached or record.cached
            agg.volatile = agg.volatile or record.volatile

    def stats(self) -> dict[str, OperationStats]:
        """Return an immutable per-operation statistics snapshot (Req 5.1, 5.4)."""
        with self._lock:
            return {
                op: OperationStats(
                    operation=op,
                    calls=agg.calls,
                    hits=agg.hits,
                    misses=agg.misses,
                    total_cost=agg.total_cost,
                    total_latency=agg.total_latency,
                    distinct_inputs=len(agg.inputs),
                    cached=agg.cached,
                    volatile=agg.volatile,
                )
                for op, agg in self._aggregates.items()
            }

    def reset(self) -> None:
        """Clear all recorded aggregates."""
        with self._lock:
            self._aggregates.clear()


@dataclass(frozen=True)
class EffectivenessReport:
    """Per-operation statistics plus advisory recommendations (Req 5, 6)."""

    stats: dict[str, OperationStats]
    recommendations: list[Recommendation]

    def to_dict(self) -> dict:
        """Return a plain-dict representation suitable for serialisation."""
        return {
            "stats": {op: s.to_dict() for op, s in self.stats.items()},
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


class EffectivenessAnalyzer:
    """Produces an :class:`EffectivenessReport` from recorded observations.

    Args:
        tracker: The :class:`OperationTracker` supplying per-operation stats.
        config: Optional threshold configuration for the recommendation rules.
    """

    def __init__(self, tracker: OperationTracker, *, config: OptimizerConfig | None = None) -> None:
        self._tracker = tracker
        self._config = config if config is not None else OptimizerConfig()

    def analyze(self) -> EffectivenessReport:
        """Return the current statistics and recommendations (Req 5, 6).

        On an empty tracker this returns an empty report rather than raising
        (Req 4.6, 5.3). Producing the report never mutates any cache counters
        (Req 5.5).
        """
        stats = self._tracker.stats()
        recommendations: list[Recommendation] = []
        for op_stats in stats.values():
            rec = recommend(op_stats, self._config)
            if rec is not None:
                recommendations.append(rec)
        return EffectivenessReport(stats=stats, recommendations=recommendations)
