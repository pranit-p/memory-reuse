"""Property 4: Cost equals the accumulated per-token formula, rounded at snapshot.

Feature: observability-and-cost-analytics, Property 4.

*For any* pricing configuration and *any* sequence of token-attributed events,
``snapshot.cost_saved`` equals
``round_half_up(Σ (tokens_in*input_price + tokens_out*output_price), minor_units)``
— the exact contributions are summed first and the *total* is rounded, so a
sequence of small savings that each round to zero still contributes to the total.

Validates: Requirements 2.1, 2.2.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.analytics import AnalyticsTracker, CacheHitEvent, PricingConfig
from memory_reuse.stats import StatsTracker

_prices = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_currencies = st.sampled_from(["USD", "EUR", "JPY", "KWD"])
_tokens = st.integers(min_value=0, max_value=5_000)
_events = st.lists(
    st.builds(CacheHitEvent, tokens_in=_tokens, tokens_out=_tokens),
    max_size=40,
)


class TestProperty4CostFormula:
    """Feature: observability-and-cost-analytics, Property 4.

    Cost equals the accumulated per-token formula, rounded at snapshot time.

    Validates: Requirements 2.1, 2.2
    """

    @settings(max_examples=100)
    @given(
        in_price=_prices,
        out_price=_prices,
        currency=_currencies,
        events=_events,
    )
    def test_cost_is_rounded_accumulated_total(
        self,
        in_price: float,
        out_price: float,
        currency: str,
        events: list[CacheHitEvent],
    ) -> None:
        """cost_saved equals the rounded sum of exact per-event contributions."""
        pricing = PricingConfig(
            input_token_price=in_price,
            output_token_price=out_price,
            currency=currency,
        )
        tracker = AnalyticsTracker(StatsTracker(), pricing=pricing)

        expected_exact = Decimal("0")
        for event in events:
            tracker.record_hit_event(event)
            expected_exact += Decimal(event.tokens_in or 0) * Decimal(str(in_price)) + Decimal(
                event.tokens_out or 0
            ) * Decimal(str(out_price))

        quantum = Decimal(1).scaleb(-pricing.minor_units)
        expected = expected_exact.quantize(quantum, rounding=ROUND_HALF_UP)

        assert tracker.snapshot().cost_saved == expected
