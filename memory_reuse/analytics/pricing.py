"""Provider-agnostic per-token pricing for cost-saved accounting (Req 2).

:class:`PricingConfig` turns *saved tokens* into a *saved monetary amount*. It
is deliberately provider-agnostic: the caller supplies the input- and
output-token prices that match their own model pricing (from LiteLLM / OpenAI /
Bedrock usage, a negotiated rate, or anything else), so nothing here hard-codes
unrealistic assumptions.

Validation happens once, at construction (Req 2.6): a negative, non-numeric, or
over-maximum price, or a currency identifier that is not a 3-letter ISO 4217
alphabetic code, raises :class:`~memory_reuse.exceptions.ConfigurationError`
naming the offending field — and constructs nothing (the dataclass is frozen, so
no partial instance exists), leaving any previously active pricing configuration
untouched.

Cost contributions are rounded to the currency's minor unit using round-half-up
(Req 2.2). A small override table captures the common ISO 4217 currencies whose
minor unit is not 2 digits; everything else defaults to 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

# Upper bound on a per-token price (Req 2.3). Prices must lie in
# ``[0.0, MAX_TOKEN_PRICE]`` inclusive.
MAX_TOKEN_PRICE: Final[float] = 1_000_000.0

# ISO 4217 currencies whose minor unit differs from the common 2 digits. Only
# the ones likely to appear in AI pricing are listed; anything not present here
# defaults to 2 fractional digits (Req 2.2).
_MINOR_UNITS: Final[dict[str, int]] = {
    "JPY": 0,  # Japanese yen
    "KRW": 0,  # South Korean won
    "CLP": 0,  # Chilean peso
    "VND": 0,  # Vietnamese dong
    "ISK": 0,  # Icelandic krona
    "BHD": 3,  # Bahraini dinar
    "KWD": 3,  # Kuwaiti dinar
    "OMR": 3,  # Omani rial
    "TND": 3,  # Tunisian dinar
}


def _minor_units(currency: str) -> int:
    """Return the number of fractional digits for a currency (default 2)."""
    return _MINOR_UNITS.get(currency, 2)


@dataclass(frozen=True)
class PricingConfig:
    """Validated per-token pricing used to compute ``cost_saved`` (Req 2).

    Attributes:
        input_token_price: Price per **input** (prompt) token, in the configured
            currency's major unit. Must be a finite number in
            ``[0.0, 1_000_000.0]``.
        output_token_price: Price per **output** (completion) token, in the
            configured currency's major unit. Must be a finite number in
            ``[0.0, 1_000_000.0]``.
        currency: A 3-letter ISO 4217 alphabetic currency code (case-insensitive
            on input; normalised to upper case). Defaults to ``"USD"``.

    Raises:
        ConfigurationError: At construction, if a price is negative, non-numeric,
            NaN, infinite, or greater than ``1_000_000.0``, or if ``currency`` is
            not a 3-letter ISO 4217 alphabetic code. The offending field is
            named and no instance is constructed (Req 2.6).

    Example::

        pricing = PricingConfig(
            input_token_price=0.0000005,    # $0.50 / 1M input tokens
            output_token_price=0.0000015,   # $1.50 / 1M output tokens
            currency="USD",
        )
        pricing.cost_for(tokens_in=1000, tokens_out=500)  # Decimal('0.00')
    """

    input_token_price: float = 0.0
    output_token_price: float = 0.0
    currency: str = "USD"

    def __post_init__(self) -> None:
        """Validate prices and currency once, at construction (Req 2.6)."""
        # Currency is validated and normalised first so the stored value is the
        # upper-cased code. ``object.__setattr__`` is required on a frozen
        # dataclass to write the normalised value.
        object.__setattr__(self, "currency", self._validate_currency(self.currency))
        self._validate_price("input_token_price", self.input_token_price)
        self._validate_price("output_token_price", self.output_token_price)

    @staticmethod
    def _validate_price(field: str, value: object) -> None:
        """Reject a price that is non-numeric, negative, NaN/inf, or over max."""
        from memory_reuse.exceptions import ConfigurationError

        # ``bool`` is an ``int`` subclass; a boolean is not a valid price.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(
                f"{field} must be a number in [0.0, {MAX_TOKEN_PRICE}], got {value!r}"
            )
        numeric = float(value)
        # NaN compares unequal to itself; infinities are out of range below.
        if numeric != numeric:
            raise ConfigurationError(f"{field} must be a finite number, got NaN")
        if numeric < 0.0 or numeric > MAX_TOKEN_PRICE:
            raise ConfigurationError(
                f"{field} must be within [0.0, {MAX_TOKEN_PRICE}], got {numeric}"
            )

    @staticmethod
    def _validate_currency(value: object) -> str:
        """Return the upper-cased currency, rejecting non-ISO-4217 codes."""
        from memory_reuse.exceptions import ConfigurationError

        if not isinstance(value, str) or len(value) != 3 or not value.isalpha():
            raise ConfigurationError(
                "currency must be a 3-letter ISO 4217 alphabetic code, " f"got {value!r}"
            )
        return value.upper()

    @property
    def minor_units(self) -> int:
        """Number of fractional digits for this configuration's currency."""
        return _minor_units(self.currency)

    def cost_for(self, tokens_in: int, tokens_out: int) -> Decimal:
        """Compute the **exact, unrounded** cost contribution for saved tokens (Req 2.1).

        Negative token counts are treated as ``0`` so a single malformed
        attribution never subtracts from an accumulated total. Prices are
        converted to :class:`~decimal.Decimal` from their string form to avoid
        binary-float drift.

        The contribution is returned at full precision; rounding to the currency
        minor unit is applied only when the accumulated total is reported (see
        :meth:`quantize` and :meth:`~memory_reuse.analytics.tracker.AnalyticsTracker.snapshot`).
        This ensures many small per-hit savings that each fall below the minor
        unit still sum into the reported total rather than each rounding to zero.

        Args:
            tokens_in: Saved input (prompt) tokens.
            tokens_out: Saved output (completion) tokens.

        Returns:
            The exact, unrounded cost contribution as a :class:`~decimal.Decimal`.
        """
        safe_in = max(0, tokens_in)
        safe_out = max(0, tokens_out)
        return Decimal(safe_in) * Decimal(str(self.input_token_price)) + Decimal(
            safe_out
        ) * Decimal(str(self.output_token_price))

    def quantize(self, amount: Decimal) -> Decimal:
        """Round an accumulated amount to the currency minor unit, half-up (Req 2.2).

        Args:
            amount: The full-precision accumulated cost.

        Returns:
            ``amount`` rounded to :attr:`minor_units` fractional digits using
            round-half-up.
        """
        quantum = Decimal(1).scaleb(-self.minor_units)
        return amount.quantize(quantum, rounding=ROUND_HALF_UP)
