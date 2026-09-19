"""Prometheus exporter for cache analytics (Phase 5, Req 5).

:class:`PrometheusExporter` registers a Prometheus custom *collector* that reads
the analytics snapshot (and the underlying stats counters) **at scrape time**,
so exported values always reflect the current state without any background loop.

``prometheus_client`` is an **optional** dependency. It is imported lazily inside
:func:`_require_prometheus`, which is called from the exporter constructor. The
module itself imports without ``prometheus_client`` installed; construction
raises :class:`~memory_reuse.exceptions.BackendNotAvailableError` naming the
extra when it is missing (Req 5.4, 7).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from memory_reuse.exceptions import BackendNotAvailableError

if TYPE_CHECKING:
    from memory_reuse.analytics.snapshot import AnalyticsSnapshot
    from memory_reuse.stats import CacheStats

logger = logging.getLogger(__name__)


def _require_prometheus() -> Any:
    """Return the ``prometheus_client`` module, raising a clear error if absent.

    Returns:
        The imported ``prometheus_client`` module.

    Raises:
        BackendNotAvailableError: If ``prometheus_client`` is not installed,
            naming the extra to install (Req 5.4).
    """
    try:
        import prometheus_client  # noqa: PLC0415

        return prometheus_client
    except ImportError as exc:
        raise BackendNotAvailableError(
            "Prometheus export requires prometheus_client. Install it with "
            '`pip install "memory-reuse[prometheus]"`.'
        ) from exc


class PrometheusExporter:
    """Exposes cache analytics as Prometheus metrics via a custom collector.

    The exporter registers a collector whose ``collect()`` reads the supplied
    providers at scrape time (Req 5.3) and yields:

    * ``memory_reuse_hit_rate`` — gauge in ``[0.0, 1.0]``
    * ``memory_reuse_tokens_saved`` — gauge
    * ``memory_reuse_cost_saved`` — gauge (in the snapshot currency)
    * ``memory_reuse_latency_saved_seconds`` — gauge
    * ``memory_reuse_hits`` / ``memory_reuse_misses`` / ``memory_reuse_errors``
      — gauges mirroring the :class:`~memory_reuse.stats.CacheStats` counters.

    A failure reading any single metric omits only that metric from the scrape
    rather than failing the whole scrape (Req 5.6).

    Args:
        snapshot_provider: Callable returning the current
            :class:`~memory_reuse.analytics.AnalyticsSnapshot`.
        stats_provider: Callable returning the current
            :class:`~memory_reuse.stats.CacheStats`.
        registry: Optional Prometheus registry to register with. Defaults to the
            client's default registry.

    Raises:
        BackendNotAvailableError: If ``prometheus_client`` is not installed.

    Example::

        exporter = PrometheusExporter(lambda: cache.analytics, lambda: cache.stats)
        # `prometheus_client.generate_latest()` now includes the cache metrics.
    """

    def __init__(
        self,
        snapshot_provider: Callable[[], AnalyticsSnapshot],
        stats_provider: Callable[[], CacheStats],
        *,
        registry: Any | None = None,
    ) -> None:
        client = _require_prometheus()
        from prometheus_client.core import GaugeMetricFamily  # noqa: PLC0415

        self._snapshot_provider = snapshot_provider
        self._stats_provider = stats_provider
        self._client = client
        self._collector = _AnalyticsCollector(snapshot_provider, stats_provider, GaugeMetricFamily)
        self._registry = registry if registry is not None else client.REGISTRY
        self._registry.register(self._collector)

    def unregister(self) -> None:
        """Remove the collector from its registry (idempotent, best-effort)."""
        try:
            self._registry.unregister(self._collector)
        except Exception:  # noqa: BLE001 — unregister is best-effort
            logger.debug("PrometheusExporter: unregister skipped")


class _AnalyticsCollector:
    """A Prometheus custom collector reading the providers at scrape time."""

    def __init__(
        self,
        snapshot_provider: Callable[[], AnalyticsSnapshot],
        stats_provider: Callable[[], CacheStats],
        gauge_metric_family: Any,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._stats_provider = stats_provider
        self._gauge_metric_family = gauge_metric_family

    def collect(self) -> Iterable[Any]:
        """Yield one gauge per metric, reading current values (Req 5.1–5.3, 5.6).

        Each metric read is guarded independently so a single failing read omits
        only that metric rather than failing the scrape (Req 5.6).
        """
        gauge_metric = self._gauge_metric_family

        # (metric name, help text, value-producing callable). Each callable is
        # invoked under its own guard below.
        specs: list[tuple[str, str, Callable[[], float]]] = [
            (
                "memory_reuse_hit_rate",
                "Cache hit rate in [0,1].",
                lambda: float(self._snapshot_provider().hit_rate),
            ),
            (
                "memory_reuse_tokens_saved",
                "Total tokens saved by cache hits.",
                lambda: float(self._snapshot_provider().tokens_saved),
            ),
            (
                "memory_reuse_cost_saved",
                "Total cost saved by cache hits.",
                lambda: float(self._snapshot_provider().cost_saved),
            ),
            (
                "memory_reuse_latency_saved_seconds",
                "Total latency saved (seconds).",
                lambda: float(self._snapshot_provider().latency_saved),
            ),
            ("memory_reuse_hits", "Total cache hits.", lambda: float(self._stats_provider().hits)),
            (
                "memory_reuse_misses",
                "Total cache misses.",
                lambda: float(self._stats_provider().misses),
            ),
            (
                "memory_reuse_errors",
                "Total backend errors.",
                lambda: float(self._stats_provider().errors),
            ),
        ]

        for name, doc, producer in specs:
            try:
                value = producer()
            except Exception:  # noqa: BLE001 — omit only the failing metric (Req 5.6)
                logger.debug("PrometheusExporter: metric '%s' omitted (read failed)", name)
                continue
            yield gauge_metric(name, doc, value=value)
