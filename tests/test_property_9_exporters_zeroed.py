"""Property 9: Exporters report zeroed metrics on empty state.

Feature: observability-and-cost-analytics, Property 9.

*For any* exporter reading a snapshot with no recorded requests, every metric is
reported with value ``0`` (hit rate ``0.0``), never raising or omitting the
metric for the empty-state reason.

Validates: Requirements 6.5.
"""

from __future__ import annotations

import pytest

from memory_reuse.analytics.snapshot import AnalyticsSnapshot
from memory_reuse.stats import CacheStats

prometheus_client = pytest.importorskip("prometheus_client")
pytest.importorskip("opentelemetry")

from memory_reuse.analytics.exporters.opentelemetry import (  # noqa: E402
    OpenTelemetryExporter,
)
from memory_reuse.analytics.exporters.prometheus import PrometheusExporter  # noqa: E402


class TestProperty9ExportersZeroed:
    """Feature: observability-and-cost-analytics, Property 9.

    Exporters report zeroed metrics on empty state.

    Validates: Requirements 6.5
    """

    def test_prometheus_empty_state_is_zeroed(self) -> None:
        """A fresh (empty) snapshot yields zero for every Prometheus metric."""
        registry = prometheus_client.CollectorRegistry()
        PrometheusExporter(
            AnalyticsSnapshot,  # zero-arg call returns the all-zero default snapshot
            CacheStats,
            registry=registry,
        )
        for name in (
            "memory_reuse_hit_rate",
            "memory_reuse_tokens_saved",
            "memory_reuse_cost_saved",
            "memory_reuse_latency_saved_seconds",
            "memory_reuse_hits",
            "memory_reuse_misses",
            "memory_reuse_errors",
        ):
            assert registry.get_sample_value(name) == 0.0

    def test_otel_empty_state_is_zeroed(self) -> None:
        """A fresh (empty) snapshot yields zero for every OTel instrument."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        OpenTelemetryExporter(AnalyticsSnapshot, meter_provider=provider)

        collected: dict[str, float] = {}
        for rm in reader.get_metrics_data().resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    for point in metric.data.data_points:
                        collected[metric.name] = point.value

        assert collected["memory_reuse.hit_rate"] == 0.0
        assert collected["memory_reuse.tokens_saved"] == 0.0
        assert collected["memory_reuse.cost_saved"] == 0.0
        assert collected["memory_reuse.latency_saved_seconds"] == 0.0
