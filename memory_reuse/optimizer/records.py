"""Immutable data models for the effectiveness analyzer (Phase 6, Req 4, 5).

:class:`OperationRecord` is one recorded execution observation — an operation
label, a hit-or-miss outcome, and optional attributed cost/latency and an input
fingerprint. :class:`OperationStats` is the per-operation aggregate the analyzer
computes from a stream of records.

Both are frozen/immutable and expose ``to_dict()`` for serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class OperationRecord:
    """A single recorded execution observation (Req 4.1).

    Attributes:
        operation: The label identifying the operation (tool/node/graph name).
        hit: ``True`` when the operation was served from cache, ``False`` on a
            miss (the underlying work ran).
        cost_saved: Optional monetary cost attributed to this observation.
            Negative values are ignored on aggregation.
        latency_saved: Optional wall-clock seconds attributed. Negative values
            are ignored on aggregation.
        input_hash: Optional stable fingerprint of the operation inputs, used to
            estimate the repeat ratio (how often identical inputs recur).
        cached: Whether the operation is *already* configured to be cached — a
            caller-supplied signal used by the recommendation rules.
        volatile: Whether the operation returns rapidly-changing / non-
            deterministic data — a caller-supplied signal used to recommend
            against semantic caching.
    """

    operation: str
    hit: bool
    cost_saved: float | None = None
    latency_saved: float | None = None
    input_hash: str | None = None
    cached: bool = False
    volatile: bool = False


@dataclass(frozen=True)
class OperationStats:
    """Per-operation aggregate computed from recorded observations (Req 5).

    Attributes:
        operation: The operation label.
        calls: Total recorded observations for the operation.
        hits: Observations that were cache hits.
        misses: Observations that were cache misses.
        total_cost: Sum of non-negative attributed cost.
        total_latency: Sum of non-negative attributed latency (seconds).
        distinct_inputs: Number of distinct input fingerprints seen (bounded).
        cached: Whether the operation is flagged as already cached.
        volatile: Whether the operation is flagged as volatile.
    """

    operation: str
    calls: int
    hits: int
    misses: int
    total_cost: Decimal
    total_latency: float
    distinct_inputs: int
    cached: bool
    volatile: bool

    @property
    def hit_rate(self) -> float:
        """Fraction of calls served from cache, ``0.0`` when no calls (Req 5.3)."""
        if self.calls == 0:
            return 0.0
        return self.hits / self.calls

    @property
    def avg_cost(self) -> Decimal:
        """Average attributed cost per call, ``0`` when no calls."""
        if self.calls == 0:
            return Decimal("0")
        return self.total_cost / self.calls

    @property
    def avg_latency(self) -> float:
        """Average attributed latency per call (seconds), ``0.0`` when no calls."""
        if self.calls == 0:
            return 0.0
        return self.total_latency / self.calls

    @property
    def repeat_ratio(self) -> float:
        """Estimated fraction of calls with repeated inputs, ``0.0`` when no calls.

        Computed as ``1 - distinct_inputs / calls``: a high ratio means the same
        inputs recur often (highly cacheable). ``0.0`` when nothing was recorded
        or no input fingerprints were supplied.
        """
        if self.calls == 0 or self.distinct_inputs == 0:
            return 0.0
        return max(0.0, 1.0 - self.distinct_inputs / self.calls)

    def to_dict(self) -> dict:
        """Return a plain-dict representation (``total_cost`` as a string)."""
        return {
            "operation": self.operation,
            "calls": self.calls,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "total_cost": str(self.total_cost),
            "avg_cost": str(self.avg_cost),
            "total_latency": self.total_latency,
            "avg_latency": self.avg_latency,
            "distinct_inputs": self.distinct_inputs,
            "repeat_ratio": self.repeat_ratio,
            "cached": self.cached,
            "volatile": self.volatile,
        }
