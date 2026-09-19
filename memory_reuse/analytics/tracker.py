"""Savings accumulation on top of the shared statistics (Req 1, 2).

:class:`AnalyticsTracker` accumulates the per-hit savings attribution carried by
:class:`~memory_reuse.analytics.snapshot.CacheHitEvent` values — tokens, cost,
and latency avoided — while reading the *hit rate* straight from the shared
:class:`~memory_reuse.stats.StatsTracker`. It never touches the underlying
``hits`` / ``misses`` / ``total_requests`` counters, so wiring analytics in
cannot change how those are recorded (Req 1.1, 8.4).

Recording is best-effort and never fatal, matching ``StatsTracker``: any
internal failure while recording an event is swallowed so the cache operation
that triggered it still returns its result (Req 1.8).

When the tracker is disabled (``enable_stats=False`` on the config) it records
nothing and every saved-total reads as ``0`` (Req 1.9).
"""

from __future__ import annotations

import contextlib
import threading
from decimal import Decimal

from memory_reuse.analytics.pricing import PricingConfig
from memory_reuse.analytics.snapshot import AnalyticsSnapshot, CacheHitEvent
from memory_reuse.stats import StatsTracker


class AnalyticsTracker:
    """Accumulates savings attribution and exposes an :class:`AnalyticsSnapshot`.

    The tracker layers on top of an existing :class:`StatsTracker` (the source
    of truth for hit rate) rather than duplicating its counters. It adds three
    running totals — tokens saved, cost saved, latency saved — each fed only by
    the non-negative portion of a recorded :class:`CacheHitEvent`'s attribution.

    Args:
        stats: The shared statistics tracker whose hit rate is surfaced in the
            snapshot. This is the same instance the caches record hits/misses
            on, so the analytics hit rate always matches
            :attr:`~memory_reuse.stats.CacheStats.hit_rate`.
        pricing: Optional per-token pricing used to derive ``cost_saved`` from a
            token-attributed event (Req 2.1). When ``None``, ``cost_saved``
            accumulates only from events that carry an explicit
            :attr:`CacheHitEvent.cost_saved`, and token tracking is unaffected
            (Req 2.4).
        enabled: When ``False`` the tracker records no events and reports zeroed
            saved-totals, mirroring ``CacheConfig.enable_stats=False`` (Req 1.9).

    Example::

        tracker = AnalyticsTracker(stats, pricing=PricingConfig(
            input_token_price=5e-7, output_token_price=1.5e-6))
        tracker.record_hit_event(CacheHitEvent(tokens_in=1000, tokens_out=500,
                                               latency_saved=0.35))
        snap = tracker.snapshot()
        print(snap.tokens_saved, snap.cost_saved, snap.latency_saved)
    """

    def __init__(
        self,
        stats: StatsTracker,
        *,
        pricing: PricingConfig | None = None,
        enabled: bool = True,
    ) -> None:
        self._stats = stats
        self._pricing = pricing
        self._enabled = enabled
        self._tokens_saved = 0
        self._cost_saved = Decimal("0")
        self._latency_saved = 0.0
        # A plain lock (not asyncio) so the three totals stay consistent under
        # threads and the tracker is usable from sync and async call sites alike.
        self._lock = threading.Lock()

    @property
    def pricing(self) -> PricingConfig | None:
        """The active pricing configuration, or ``None`` when cost is untracked."""
        return self._pricing

    def record_hit_event(self, event: CacheHitEvent) -> None:
        """Record a cache hit's savings attribution (Req 1.2–1.6, 1.8, 1.9).

        Only the non-negative portion of each attribution is accumulated;
        missing (``None``) or negative fields leave the corresponding total
        unchanged. When a pricing configuration is active and the event carries
        token attribution, cost is derived from tokens; otherwise the event's
        explicit ``cost_saved`` (when supplied and non-negative) is added.

        Recording is best-effort: any internal error is swallowed so the calling
        cache operation always returns its result (Req 1.8). When the tracker is
        disabled nothing is recorded (Req 1.9).

        Args:
            event: The recorded :class:`CacheHitEvent`.
        """
        if not self._enabled:
            return
        with contextlib.suppress(Exception), self._lock:
            self._accumulate_tokens(event)
            self._accumulate_cost(event)
            self._accumulate_latency(event)

    def snapshot(self) -> AnalyticsSnapshot:
        """Return an immutable snapshot of the current analytics (Req 1.7).

        Before any event — and whenever the tracker is disabled — the
        saved-totals are ``0`` and the hit rate is ``0.0``. The hit rate is read
        from the shared :class:`StatsTracker`, so it always equals
        :attr:`~memory_reuse.stats.CacheStats.hit_rate`.

        Returns:
            An :class:`AnalyticsSnapshot`.
        """
        currency = self._pricing.currency if self._pricing is not None else "USD"
        if not self._enabled:
            return AnalyticsSnapshot(currency=currency)

        hit_rate = self._stats.get_stats().hit_rate
        with self._lock:
            cost_saved = self._rounded_cost()
            return AnalyticsSnapshot(
                hit_rate=hit_rate,
                tokens_saved=self._tokens_saved,
                cost_saved=cost_saved,
                currency=currency,
                latency_saved=self._latency_saved,
            )

    def _rounded_cost(self) -> Decimal:
        """Round the full-precision accumulated cost to the currency minor unit.

        Rounding is applied here, at read time, over the accumulated total —
        never per hit — so many small savings that each fall below the currency
        minor unit still sum into the reported total (Req 2.2). With a pricing
        configuration the currency's minor unit is used; without one the total
        is always ``0`` and is returned unrounded.
        """
        if self._pricing is not None:
            return self._pricing.quantize(self._cost_saved)
        return self._cost_saved

    def reset(self) -> None:
        """Reset the accumulated saved-totals to zero (does not touch stats)."""
        with self._lock:
            self._tokens_saved = 0
            self._cost_saved = Decimal("0")
            self._latency_saved = 0.0

    # ------------------------------------------------------------------
    # Accumulation helpers (called under the lock)
    # ------------------------------------------------------------------

    def _accumulate_tokens(self, event: CacheHitEvent) -> None:
        """Add the non-negative input + output token counts (Req 1.2, 1.6)."""
        if self._is_positive_int(event.tokens_in):
            self._tokens_saved += event.tokens_in  # type: ignore[operator]
        if self._is_positive_int(event.tokens_out):
            self._tokens_saved += event.tokens_out  # type: ignore[operator]

    def _accumulate_cost(self, event: CacheHitEvent) -> None:
        """Add the cost contribution (Req 1.3, 2.1, 2.4, 1.6).

        With pricing active and token attribution present, cost is derived from
        the per-token formula. Otherwise the event's explicit ``cost_saved`` is
        used when supplied and non-negative.
        """
        if self._pricing is not None and (
            event.tokens_in is not None or event.tokens_out is not None
        ):
            # Accumulate the exact, unrounded contribution; rounding to the
            # currency minor unit happens once, at snapshot time (Req 2.2).
            contribution = self._pricing.cost_for(
                tokens_in=event.tokens_in or 0,
                tokens_out=event.tokens_out or 0,
            )
            if contribution > 0:
                self._cost_saved += contribution
            return

        cost = event.cost_saved
        if self._is_positive_number(cost):
            self._cost_saved += Decimal(str(cost))

    def _accumulate_latency(self, event: CacheHitEvent) -> None:
        """Add the non-negative latency saving (Req 1.4, 1.6)."""
        if self._is_positive_number(event.latency_saved):
            self._latency_saved += float(event.latency_saved)  # type: ignore[arg-type]

    @staticmethod
    def _is_positive_int(value: object) -> bool:
        """True when ``value`` is a positive int (excluding ``bool``)."""
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @staticmethod
    def _is_positive_number(value: object) -> bool:
        """True when ``value`` is a positive int/float (excluding ``bool``)."""
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
