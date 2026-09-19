"""Property 8: Exporters report the current snapshot values.

Feature: observability-and-cost-analytics, Property 8.

*For any* analytics snapshot, a scrape from the Prometheus custom collector and
a collection from the OpenTelemetry observable gauges report metric values equal
to that snapshot's ``hit_rate``, ``tokens_saved``, ``cost_saved``, and
``latency_saved`` (and, for Prometheus, ``hits`` / ``misses`` / ``errors`` from
CacheStats).

Validates: Requirements 5.1, 5.2, 5.3, 6.1, 6.4.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.analytics.snapshot import AnalyticsSnapshot
from memory_reuse.stats import CacheStats

prometheus_client = pytest.importorskip("prometheus_client")
pytest.importorskip("opentelemetry")

from memory_reuse.analytics.exporters.opentelemetry import (  # noqa: E402
    OpenTelemetryExporter,
)
from memory_reuse.analytics.exporters.prometheus import PrometheusExporter  # noqa: E402


def _snapshot(hit_rate: float, tokens: int, cost: str, latency: float) -> AnalyticsSnapshot:
    return AnalyticsSnapshot(
        hit_rate=hit_rate,
        tokens_saved=tokens,
        cost_saved=Decimal(cost),
        currency="USD",
        latency_saved=latency,
    )


class TestProperty8ExportersReportSnapshot:
    """Feature: observability-and-cost-analytics, Property 8.

    Exporters report the current snapshot values.

    Validates: Requirements 5.1, 5.2, 5.3, 6.1, 6.4
    """

    @settings(max_examples=100)
    @given(
        hit_rate=st.floats(min_value=0.0, max_value=1.0),
        tokens=st.integers(min_value=0, max_value=10_000_000),
        cost_cents=st.integers(min_value=0, max_value=10_000_000),
        latency=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        hits=st.integers(min_value=0, max_value=1_000_000),
        misses=st.integers(min_value=0, max_value=1_000_000),
        errors=st.integers(min_value=0, max_value=1_000_000),
    )
    def test_prometheus_scrape_matches_snapshot(
        self,
        hit_rate: float,
        tokens: int,
        cost_cents: int,
        latency: float,
        hits: int,
        misses: int,
        errors: int,
    ) -> None:
        """A Prometheus scrape reports exactly the current snapshot / stats values."""
        cost = f"{cost_cents / 100:.2f}"
        snap = _snapshot(hit_rate, tokens, cost, latency)
        stats = CacheStats(hits=hits, misses=misses, errors=errors)

        registry = prometheus_client.CollectorRegistry()
        PrometheusExporter(lambda: snap, lambda: stats, registry=registry)

        def value(name: str) -> float:
            got = registry.get_sample_value(name)
            assert got is not None, f"metric {name} missing"
            return got

        assert value("memory_reuse_hit_rate") == pytest.approx(hit_rate)
        assert value("memory_reuse_tokens_saved") == pytest.approx(float(tokens))
        assert value("memory_reuse_cost_saved") == pytest.approx(float(Decimal(cost)))
        assert value("memory_reuse_latency_saved_seconds") == pytest.approx(latency)
        assert value("memory_reuse_hits") == pytest.approx(float(hits))
        assert value("memory_reuse_misses") == pytest.approx(float(misses))
        assert value("memory_reuse_errors") == pytest.approx(float(errors))

    @settings(max_examples=50)
    @given(
        hit_rate=st.floats(min_value=0.0, max_value=1.0),
        tokens=st.integers(min_value=0, max_value=10_000_000),
        cost_cents=st.integers(min_value=0, max_value=10_000_000),
        latency=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_otel_collection_matches_snapshot(
        self,
        hit_rate: float,
        tokens: int,
        cost_cents: int,
        latency: float,
    ) -> None:
        """An OTel in-memory collection reports exactly the current snapshot values."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        cost = f"{cost_cents / 100:.2f}"
        snap = _snapshot(hit_rate, tokens, cost, latency)

        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        OpenTelemetryExporter(lambda: snap, meter_provider=provider)

        collected: dict[str, float] = {}
        data = reader.get_metrics_data()
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    for point in metric.data.data_points:
                        collected[metric.name] = point.value

        assert collected["memory_reuse.hit_rate"] == pytest.approx(hit_rate)
        assert collected["memory_reuse.tokens_saved"] == pytest.approx(float(tokens))
        assert collected["memory_reuse.cost_saved"] == pytest.approx(float(Decimal(cost)))
        assert collected["memory_reuse.latency_saved_seconds"] == pytest.approx(latency)
