"""Property 6: Invalid pricing is rejected and leaves state unchanged.

Feature: observability-and-cost-analytics, Property 6.

*For any* pricing configuration with a negative, non-numeric, or over-max price,
or a non-ISO-4217 currency, construction raises a validation error identifying
the field, and any previously active pricing and accumulated cost are unchanged.

Validates: Requirements 2.5, 2.6.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from memory_reuse.analytics import AnalyticsTracker, CacheHitEvent, PricingConfig
from memory_reuse.analytics.pricing import MAX_TOKEN_PRICE
from memory_reuse.exceptions import ConfigurationError
from memory_reuse.stats import StatsTracker

# Prices that must be rejected: negative, NaN, infinite, or over the max.
_bad_prices = st.one_of(
    st.floats(min_value=-1e6, max_value=-1e-6, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
    st.just(float("inf")),
    st.floats(min_value=MAX_TOKEN_PRICE + 1, max_value=1e12, allow_infinity=False),
)
# Currencies that must be rejected: wrong length or non-alphabetic.
_bad_currencies = st.one_of(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=0, max_size=2),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=4, max_size=6),
    st.just("US1"),
    st.just("12$"),
)


class TestProperty6InvalidPricing:
    """Feature: observability-and-cost-analytics, Property 6.

    Invalid pricing is rejected and leaves state unchanged.

    Validates: Requirements 2.5, 2.6
    """

    @settings(max_examples=100)
    @given(bad_price=_bad_prices, field=st.sampled_from(["input", "output"]))
    def test_bad_price_rejected_state_unchanged(self, bad_price: float, field: str) -> None:
        """A bad price raises ConfigurationError and leaves prior state intact."""
        # A previously active, valid pricing + accumulated cost.
        good = PricingConfig(input_token_price=1e-3, output_token_price=1e-3)
        tracker = AnalyticsTracker(StatsTracker(), pricing=good)
        tracker.record_hit_event(CacheHitEvent(tokens_in=1000, tokens_out=1000))
        before = tracker.snapshot().cost_saved
        assert before > Decimal("0")

        kwargs = {"input_token_price": 1e-3, "output_token_price": 1e-3}
        kwargs["input_token_price" if field == "input" else "output_token_price"] = bad_price
        with pytest.raises(ConfigurationError):
            PricingConfig(**kwargs)  # type: ignore[arg-type]

        # The already-built tracker is untouched by the failed construction.
        assert tracker.snapshot().cost_saved == before

    @settings(max_examples=100)
    @given(bad_currency=_bad_currencies)
    def test_bad_currency_rejected(self, bad_currency: str) -> None:
        """A non-ISO-4217 currency raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            PricingConfig(input_token_price=0.0, output_token_price=0.0, currency=bad_currency)
