"""Example: seeing memory-reuse cost savings on AgentCore Runtime via CloudWatch.

This shows how the Phase 5 analytics surface on **Amazon Bedrock AgentCore
Runtime**, where each session runs in an isolated microVM. Two facts drive the
design:

1. The ``agentcore`` cache backend is **shared** across microVMs (a value cached
   in one VM is readable in another).
2. The analytics tracker is **per-process** — each microVM accumulates its own
   ``tokens_saved`` / ``cost_saved`` / ``latency_saved`` / ``hit_rate``. To see
   a *fleet-wide* total you must export the per-VM metrics to a central system
   and aggregate there.

On AgentCore that central system is **Amazon CloudWatch**, and the transport is
**OpenTelemetry via the AWS Distro for OpenTelemetry (ADOT)**. Per the AWS docs,
AgentCore emits built-in metrics automatically, and *custom* metrics require
instrumenting your code with the ADOT SDK. memory-reuse's ``OpenTelemetryExporter``
registers its instruments with the OTel meter provider that ADOT configures, so
the cache's savings metrics ride the same pipeline into CloudWatch and appear on
the CloudWatch **GenAI Observability** page — aggregated across every microVM.
(Content summarised from the AWS AgentCore observability docs for compliance.)

----------------------------------------------------------------------------
Deployment checklist (real AgentCore, not run by this file)
----------------------------------------------------------------------------

1. Dependencies — add to ``requirements.txt``:

       memory-reuse[agentcore,opentelemetry]
       aws-opentelemetry-distro>=0.18.0
       boto3

2. Enable CloudWatch Transaction Search once in your account (CloudWatch
   console → Application Signals (APM) → Transaction search → Enable).

3. Run the agent under ADOT auto-instrumentation so a meter provider exists:

       opentelemetry-instrument python my_agent.py

   In a container:

       CMD ["opentelemetry-instrument", "python", "main.py"]

4. View the metrics on the CloudWatch GenAI Observability page.

----------------------------------------------------------------------------
The agent handler pattern (the part that matters)
----------------------------------------------------------------------------

    from bedrock_agentcore.runtime import BedrockAgentCoreApp
    from memory_reuse import CacheConfig, CacheHitEvent, MemoryCache, PricingConfig
    from memory_reuse.analytics.exporters.opentelemetry import OpenTelemetryExporter

    app = BedrockAgentCoreApp()

    # Shared cache across microVMs + pricing so cost can be computed.
    cache = MemoryCache(CacheConfig(
        backend="agentcore",
        agentcore_region="us-east-1",
        agentcore_memory_id="mem-123",
        pricing=PricingConfig(input_token_price=1.5e-7, output_token_price=6e-7),
    ))

    # Register the savings instruments with ADOT's (global) meter provider.
    # ADOT ships them to CloudWatch; no meter_provider arg needed.
    OpenTelemetryExporter(lambda: cache.analytics)

    @app.entrypoint
    def handler(payload, context):
        cache.set_context(user_id=context.identity.user_id,
                          session_id=context.session_id)

        prompt = payload["prompt"]
        cached = cache.exact.get_sync(["qa", prompt])  # illustrative
        if cached is not None:
            # On a HIT, record what the skipped work would have cost. Only you
            # know that (the LLM wrapper does it automatically for LiteLLM).
            cache.record_hit_event(CacheHitEvent(
                tokens_in=1200, tokens_out=400, latency_saved=0.35,
                operation="summarise",
            ))
            return cached
        ...  # miss: run the model, store, return

Because the exporter reads ``cache.analytics`` at collection time, each microVM
reports its current savings on every OTel collection cycle, and CloudWatch sums
them across the fleet.

----------------------------------------------------------------------------
Runnable, offline demo below
----------------------------------------------------------------------------

To keep this file runnable with no AWS account and no ADOT, the demo uses a
local OpenTelemetry SDK ``MeterProvider`` with an in-memory reader instead of
CloudWatch, and simulates three microVMs each recording their own savings. It
then collects each provider and sums ``cost_saved`` to show how CloudWatch would
aggregate the fleet. Install the exporter extra to run it:

    pip install "memory-reuse[opentelemetry]"
    python examples/agentcore_analytics_otel.py
"""

from __future__ import annotations

from memory_reuse import CacheConfig, CacheHitEvent, MemoryCache, PricingConfig


def _collect_cost(reader: object) -> float:
    """Sum the ``memory_reuse.cost_saved`` gauge from an in-memory OTel reader."""
    total = 0.0
    data = reader.get_metrics_data()  # type: ignore[attr-defined]
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "memory_reuse.cost_saved":
                    for point in metric.data.data_points:
                        total += point.value
    return total


def main() -> None:
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    except ImportError:
        print(
            "This demo needs the OpenTelemetry SDK. Install it with:\n"
            '    pip install "memory-reuse[opentelemetry]"'
        )
        return

    from memory_reuse.analytics.exporters.opentelemetry import OpenTelemetryExporter

    pricing = PricingConfig(input_token_price=1.5e-7, output_token_price=6e-7)

    # Simulate three isolated microVMs. Each has its OWN cache/tracker (analytics
    # is per-process) but they would share one AgentCore cache backend in prod.
    # Each VM gets its own OTel meter provider + reader, standing in for the
    # per-VM ADOT pipeline that ships to CloudWatch.
    per_vm_hits = [5000, 1200, 8000]
    readers = []
    for vm_index, hits in enumerate(per_vm_hits):
        cache = MemoryCache(CacheConfig(backend="memory", pricing=pricing))

        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        OpenTelemetryExporter(lambda c=cache: c.analytics, meter_provider=provider)

        # Each hit records the tokens/latency the cache saved on that microVM.
        for _ in range(hits):
            cache.record_hit_event(
                CacheHitEvent(tokens_in=1200, tokens_out=400, latency_saved=0.35)
            )

        vm_cost = cache.analytics.cost_saved
        print(f"  microVM {vm_index + 1}: {hits:>5} hits -> cost saved {vm_cost} USD")
        readers.append(reader)

    # CloudWatch aggregates the per-VM OTel exports. We mimic that by summing the
    # cost_saved gauge collected from every VM's reader.
    fleet_cost = sum(_collect_cost(r) for r in readers)
    print("\n=== Fleet total (what CloudWatch would show) ===")
    print(f"  cost saved across all microVMs: ${fleet_cost:.2f} USD")
    print(
        "\nOn real AgentCore, each microVM's OpenTelemetryExporter ships these\n"
        "same metrics through ADOT to CloudWatch, which sums them on the\n"
        "GenAI Observability dashboard — no manual aggregation needed."
    )


if __name__ == "__main__":
    main()
