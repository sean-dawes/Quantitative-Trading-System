"""Rule-based filters that veto a trade before it ever reaches sizing/execution."""
from datetime import datetime, timezone

import config
from data_pipeline.market_data import get_next_earnings_date


def passes_volatility_filter(atr_pct: float) -> bool:
    """Reject names that are either too wild (ATR% too high) or too flat (no edge)."""
    return config.MIN_ATR_PCT <= atr_pct <= config.MAX_ATR_PCT


def passes_earnings_filter(ticker: str) -> bool:
    """Reject trading a name within EARNINGS_BLACKOUT_DAYS of its next earnings date."""
    next_earnings = get_next_earnings_date(ticker)
    if next_earnings is None:
        return True  # unknown -> don't block the trade
    now = datetime.now(timezone.utc)
    if next_earnings.tzinfo is None:
        next_earnings = next_earnings.replace(tzinfo=timezone.utc)
    days_until = (next_earnings - now).days
    return days_until > config.EARNINGS_BLACKOUT_DAYS or days_until < 0
