"""Immutable analytics data models: the recorded event and the snapshot.

:class:`CacheHitEvent` is one recorded cache hit carrying *optional*
attribution — how many tokens, how much cost, and/or how much latency the hit
avoided. Any field may be omitted; a missing attribution simply leaves the
corresponding saved-total unchanged (Req 1.5), and a negative attribution is
ignored on recording (Req 1.6).

:class:`AnalyticsSnapshot` is an immutable point-in-time view returned by
:meth:`~memory_reuse.analytics.tracker.AnalyticsTracker.snapshot`. It pairs
``cost_saved`` with the currency it is denominated in so a consumer never has to
guess the units.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class CacheHitEvent:
    """A single recorded cache hit with optional savings attribution (Req 1).

    Attributes:
        tokens_in: Saved input (prompt) tokens attributed to this hit, or
            ``None`` when unknown. Negative values are ignored on recording.
        tokens_out: Saved output (completion) tokens attributed to this hit, or
            ``None`` when unknown. Negative values are ignored on recording.
        cost_saved: A directly-attributed monetary saving for this hit, used only
            when no pricing configuration is active. ``None`` when not supplied;
            negative values are ignored on recording.
        latency_saved: Saved wall-clock seconds attributed to this hit, or
            ``None`` when unknown. Negative values are ignored on recording.
        operation: Optional label identifying the operation (node / tool / graph
            name) this hit belongs to. Reserved for a future per-operation
            breakdown; it does not affect the aggregate totals in this phase.
    """

    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_saved: float | None = None
    latency_saved: float | None = None
    operation: str | None = None


@dataclass(frozen=True)
class AnalyticsSnapshot:
    """Immutable snapshot of savings analytics at a point in time (Req 1.7).

    Attributes:
        hit_rate: Fraction of requests that were cache hits, in ``[0.0, 1.0]``.
            Read straight from the shared statistics so it never diverges from
            :attr:`~memory_reuse.stats.CacheStats.hit_rate`.
        tokens_saved: Total non-negative tokens (input + output) avoided by
            cache hits recorded so far. ``0`` before any event.
        cost_saved: Total monetary saving avoided by cache hits, as a
            :class:`~decimal.Decimal`. ``0`` when no pricing configuration is
            active. Denominated in :attr:`currency`.
        currency: The 3-letter ISO 4217 code ``cost_saved`` is expressed in.
        latency_saved: Total non-negative wall-clock seconds avoided by cache
            hits recorded so far. ``0.0`` before any event.
    """

    hit_rate: float = 0.0
    tokens_saved: int = 0
    cost_saved: Decimal = Decimal("0")
    currency: str = "USD"
    latency_saved: float = 0.0

    def to_dict(self) -> dict:
        """Return a plain-dict representation of the snapshot.

        ``cost_saved`` is rendered as a string to preserve its exact decimal
        value across a JSON boundary (JSON has no decimal type).

        Returns:
            Dictionary with keys ``hit_rate``, ``tokens_saved``, ``cost_saved``,
            ``currency``, and ``latency_saved``.
        """
        return {
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "cost_saved": str(self.cost_saved),
            "currency": self.currency,
            "latency_saved": self.latency_saved,
        }
