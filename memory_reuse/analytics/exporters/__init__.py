"""Opt-in metrics exporters for the analytics snapshot (Phase 5).

These modules publish an :class:`~memory_reuse.analytics.AnalyticsSnapshot` (and,
for Prometheus, the underlying :class:`~memory_reuse.stats.CacheStats` counters)
to an external monitoring system. Each exporter imports its third-party
dependency **lazily**, inside a ``_require_*`` guard called at construction, so
importing this package pulls in neither ``prometheus_client`` nor
``opentelemetry`` (Req 7).

Import the exporter you need directly, e.g.::

    from memory_reuse.analytics.exporters.prometheus import PrometheusExporter
    from memory_reuse.analytics.exporters.opentelemetry import OpenTelemetryExporter

They are intentionally **not** re-exported here, so that
``import memory_reuse.analytics.exporters`` stays free of any optional
dependency.
"""

from __future__ import annotations
