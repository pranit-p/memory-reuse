"""Example tests for exporter wiring and named-error behaviour (Phase 5).

Covers:
* the Prometheus registry exposes the expected metric names in exposition
  format (Req 5.1),
* the OpenTelemetry exporter registers instruments with a supplied
  ``MeterProvider`` and with the global one (Req 6.2, 6.3),
* with each dependency simulated absent, constructing the exporter raises
  ``BackendNotAvailableError`` naming the extra rather than a raw ImportError
  (Req 5.4, 6.7).
"""

from __future__ import annotations

import builtins
from decimal import Decimal

import pytest

from memory_reuse.analytics.snapshot import AnalyticsSnapshot
from memory_reuse.exceptions import BackendNotAvailableError
from memory_reuse.stats import CacheStats

prometheus_client = pytest.importorskip("prometheus_client")
pytest.importorskip("opentelemetry")

from memory_reuse.analytics.exporters import opentelemetry as otel_mod  # noqa: E402
from memory_reuse.analytics.exporters import prometheus as prom_mod  # noqa: E402
from memory_reuse.analytics.exporters.opentelemetry import (  # noqa: E402
    OpenTelemetryExporter,
)
from memory_reuse.analytics.exporters.prometheus import PrometheusExporter  # noqa: E402


def _snap() -> AnalyticsSnapshot:
    return AnalyticsSnapshot(
        hit_rate=0.5, tokens_saved=10, cost_saved=Decimal("0.10"), latency_saved=0.1
    )


def test_prometheus_exposition_contains_metric_names() -> None:
    """generate_latest over the registry contains the metric names (Req 5.1)."""
    registry = prometheus_client.CollectorRegistry()
    PrometheusExporter(_snap, lambda: CacheStats(hits=1), registry=registry)
    text = prometheus_client.generate_latest(registry).decode("utf-8")
    for name in (
        "memory_reuse_hit_rate",
        "memory_reuse_tokens_saved",
        "memory_reuse_cost_saved",
        "memory_reuse_latency_saved_seconds",
        "memory_reuse_hits",
        "memory_reuse_misses",
        "memory_reuse_errors",
    ):
        assert name in text


def test_otel_registers_with_supplied_and_global_provider() -> None:
    """The exporter works with a supplied MeterProvider and the global one (Req 6.2, 6.3)."""
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    # Supplied provider (Req 6.2).
    reader = InMemoryMetricReader()
    supplied = MeterProvider(metric_readers=[reader])
    OpenTelemetryExporter(_snap, meter_provider=supplied)
    names = {
        metric.name
        for rm in reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
    }
    assert "memory_reuse.hit_rate" in names

    # Global provider (Req 6.3) — construction must succeed using the global one.
    global_reader = InMemoryMetricReader()
    otel_metrics.set_meter_provider(MeterProvider(metric_readers=[global_reader]))
    OpenTelemetryExporter(_snap)  # no explicit provider
    global_names = {
        metric.name
        for rm in global_reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
    }
    assert "memory_reuse.hit_rate" in global_names


def test_prometheus_missing_dependency_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """With prometheus_client absent, construction raises a named error (Req 5.4)."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            raise ImportError("simulated missing prometheus_client")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BackendNotAvailableError, match="memory-reuse\\[prometheus\\]"):
        prom_mod._require_prometheus()


def test_otel_missing_dependency_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """With opentelemetry absent, construction raises a named error (Req 6.7)."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("simulated missing opentelemetry")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BackendNotAvailableError, match="memory-reuse\\[opentelemetry\\]"):
        otel_mod._require_opentelemetry()
