"""Property 10: A single failing metric does not fail the whole collection.

Feature: observability-and-cost-analytics, Property 10.

*For any* set of metrics where one metric read raises internally, the Prometheus
scrape and the OTel collection omit only the failing metric and still report the
others.

Validates: Requirements 5.6, 6.6.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from memory_reuse.stats import CacheStats

prometheus_client = pytest.importorskip("prometheus_client")
pytest.importorskip("opentelemetry")

from memory_reuse.analytics.exporters.opentelemetry import (  # noqa: E402
    OpenTelemetryExporter,
)
from memory_reuse.analytics.exporters.prometheus import PrometheusExporter  # noqa: E402


class _PartialFailSnapshot:
    """A snapshot-shaped stand-in whose ``hit_rate`` read raises.

    Not a real :class:`AnalyticsSnapshot` (which is a frozen dataclass with a
    plain ``hit_rate`` field); this duck-typed object exposes the same
    attributes but raises when ``hit_rate`` is read, to force exactly one metric
    to fail while the others succeed.
    """

    tokens_saved = 0
    cost_saved = Decimal("0")
    latency_saved = 0.0
    currency = "USD"

    @property
    def hit_rate(self) -> float:
        raise RuntimeError("boom")


class TestProperty10FailingMetricIsolated:
    """Feature: observability-and-cost-analytics, Property 10.

    A single failing metric does not fail the whole collection.

    Validates: Requirements 5.6, 6.6
    """

    def test_prometheus_omits_only_failing_metric(self) -> None:
        """A raising hit_rate read omits that metric but keeps the others."""
        bad = _PartialFailSnapshot()
        registry = prometheus_client.CollectorRegistry()
        PrometheusExporter(lambda: bad, lambda: CacheStats(hits=3, misses=1), registry=registry)

        # The failing metric is absent, the scrape as a whole still succeeds.
        assert registry.get_sample_value("memory_reuse_hit_rate") is None
        assert registry.get_sample_value("memory_reuse_tokens_saved") == 0.0
        assert registry.get_sample_value("memory_reuse_hits") == 3.0
        assert registry.get_sample_value("memory_reuse_misses") == 1.0

    def test_otel_omits_only_failing_instrument(self) -> None:
        """A raising hit_rate callback omits that instrument but keeps the others."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        bad = _PartialFailSnapshot()
        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        OpenTelemetryExporter(lambda: bad, meter_provider=provider)

        collected: dict[str, float] = {}
        for rm in reader.get_metrics_data().resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    for point in metric.data.data_points:
                        collected[metric.name] = point.value

        # hit_rate omitted (its callback returned no observation); others present.
        assert "memory_reuse.hit_rate" not in collected
        assert collected.get("memory_reuse.tokens_saved") == 0.0
