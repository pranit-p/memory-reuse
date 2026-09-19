# Cost analytics

Beyond hit rate, memory-reuse can quantify the work a cache eliminated —
**tokens saved**, **cost saved**, and **latency saved** — so you can answer
"how much did the cache actually save me?". The analytics layer sits on top of
the same counters as [Cache statistics](statistics.md); it is fully opt-in and
changes nothing when unused.

## How attribution works

The cache cannot know a call's token or latency cost, so **you record it on a
hit** via `record_hit_event`. The LiteLLM completion wrapper does this
automatically from the response's `usage` block; for your own tools you record
the savings yourself.

```python
from memory_reuse import CacheConfig, CacheHitEvent, MemoryCache, PricingConfig

cache = MemoryCache(CacheConfig(
    pricing=PricingConfig(
        input_token_price=0.0000005,     # $0.50 / 1M input tokens
        output_token_price=0.0000015,    # $1.50 / 1M output tokens
        currency="USD",
    ),
))

cache.record_hit_event(CacheHitEvent(
    tokens_in=1200, tokens_out=400, latency_saved=0.4,
    operation="search_confluence",
))

snap = cache.analytics
print(snap.tokens_saved, snap.cost_saved, snap.currency, snap.latency_saved)
```

`cache.analytics` returns an immutable `AnalyticsSnapshot` with `hit_rate`,
`tokens_saved`, `cost_saved` (a `Decimal`), `currency`, and `latency_saved`.

## Pricing and rounding

`PricingConfig` is provider-agnostic — you supply the input- and output-token
prices that match your model pricing. Cost contributions are **accumulated at
full precision** and rounded to the currency's minor unit (2 digits for most
currencies) only when the snapshot is read. This matters at volume: many small
per-hit savings that each fall below a cent still sum into the reported total
rather than each rounding to zero.

With no `PricingConfig`, `cost_saved` stays `0` while tokens and latency still
track. When `enable_stats=False`, analytics is fully zeroed. Recording is
best-effort and never fatal — a malformed event never breaks a cache operation,
and negative attributions are ignored.

## CLI

Dump a snapshot from your running app and inspect it with the `memory-reuse`
command (standard library only — no extra required):

```python
from memory_reuse.cli import dump_snapshot
dump_snapshot(cache, "snapshot.json")
```

```bash
memory-reuse stats   --snapshot snapshot.json
memory-reuse savings --snapshot snapshot.json
```

`stats` shows hit rate and the hit/miss counters; `savings` shows tokens, cost
(with currency), and latency saved. An empty snapshot renders zeros rather than
an error.

## Prometheus / OpenTelemetry export

Publish the analytics to your existing monitoring stack. Both exporters are
opt-in, lazily imported, and read the current snapshot at scrape/collection
time.

=== "Prometheus"
    ```bash
    pip install "memory-reuse[prometheus]"
    ```
    ```python
    from memory_reuse.analytics.exporters.prometheus import PrometheusExporter

    PrometheusExporter(lambda: cache.analytics, lambda: cache.stats)
    # exposes: memory_reuse_hit_rate, memory_reuse_tokens_saved,
    # memory_reuse_cost_saved, memory_reuse_latency_saved_seconds,
    # memory_reuse_hits / _misses / _errors
    ```

=== "OpenTelemetry"
    ```bash
    pip install "memory-reuse[opentelemetry]"
    ```
    ```python
    from memory_reuse.analytics.exporters.opentelemetry import OpenTelemetryExporter

    # Uses the global meter provider, or pass meter_provider=... explicitly.
    OpenTelemetryExporter(lambda: cache.analytics)
    ```

A failure reading any single metric omits only that metric rather than failing
the whole scrape/collection.

## On AWS AgentCore Runtime

AgentCore runs each session in an **isolated microVM**. Two consequences:

- The `agentcore` cache backend is **shared** across microVMs.
- The analytics tracker is **per-process** — each microVM accumulates its own
  `tokens_saved` / `cost_saved` / `latency_saved` / `hit_rate`. To see a
  *fleet-wide* total you export the per-VM metrics and aggregate centrally.

On AgentCore that central system is **Amazon CloudWatch**, reached over
**OpenTelemetry via the AWS Distro for OpenTelemetry (ADOT)**. AgentCore emits
built-in metrics automatically; *custom* metrics (like the cache's savings)
require instrumenting your code with the ADOT SDK. The `OpenTelemetryExporter`
registers its instruments with the meter provider ADOT configures, so the
savings ride the same pipeline into CloudWatch and appear — aggregated across
every microVM — on the CloudWatch **GenAI Observability** page.

To wire it up:

1. Add to `requirements.txt`: `memory-reuse[agentcore,opentelemetry]`,
   `aws-opentelemetry-distro>=0.18.0`, `boto3`.
2. Enable **CloudWatch Transaction Search** once in your account.
3. Run under ADOT auto-instrumentation: `opentelemetry-instrument python my_agent.py`
   (or `CMD ["opentelemetry-instrument", "python", "main.py"]` in a container).
4. In your handler, construct the cache once with `backend="agentcore"` and a
   `PricingConfig`, call `OpenTelemetryExporter(lambda: cache.analytics)`, and
   on a cache hit call `cache.record_hit_event(...)` to attribute what the hit
   avoided.

See `examples/agentcore_analytics_otel.py` for the full handler pattern plus a
runnable offline demo that simulates three microVMs aggregating into a fleet
total.
