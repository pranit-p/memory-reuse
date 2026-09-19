"""Analytics — savings accounting on top of the core statistics (Phase 5).

This subpackage is **additive and opt-in**. It layers *savings attribution*
(tokens, cost, and latency avoided by cache hits) on top of the existing
best-effort :class:`~memory_reuse.stats.StatsTracker` without changing how the
core ``hits`` / ``misses`` / ``total_requests`` counters are recorded.

The pieces are:

* :class:`PricingConfig` — a validated, provider-agnostic per-token pricing
  model used to turn saved tokens into a saved monetary amount.
* :class:`CacheHitEvent` — one recorded cache hit carrying optional
  token / cost / latency attribution.
* :class:`AnalyticsSnapshot` — an immutable point-in-time view of hit rate,
  ``tokens_saved``, ``cost_saved`` (paired with its currency), and
  ``latency_saved``.
* :class:`AnalyticsTracker` — accumulates :class:`CacheHitEvent` attributions,
  reading hit rate straight from the shared ``StatsTracker`` so the two never
  drift.

Nothing here imports a third-party dependency. The exporters (Prometheus /
OpenTelemetry) that consume an :class:`AnalyticsSnapshot` live under
``memory_reuse.analytics.exporters`` and are **not** imported by this package,
so ``import memory_reuse.analytics`` stays free of any optional dependency.
"""

from __future__ import annotations

from memory_reuse.analytics.pricing import PricingConfig
from memory_reuse.analytics.snapshot import AnalyticsSnapshot, CacheHitEvent
from memory_reuse.analytics.tracker import AnalyticsTracker

__all__ = [
    "PricingConfig",
    "CacheHitEvent",
    "AnalyticsSnapshot",
    "AnalyticsTracker",
]
