"""Basic sanity checks for the indicator math -- run with `pytest`."""
import numpy as np
import pandas as pd

from indicators.technical import (
    add_all_indicators,
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
)


def _fake_price_series(n=100, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.05, scale=1.0, size=n)
    prices = 100 + np.cumsum(steps)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates)


def _fake_ohlcv_df(n=100, seed=0):
    close = _fake_price_series(n, seed)
    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.default_rng(seed).integers(1_000_000, 5_000_000, size=n),
    })
    return df


def test_rsi_bounds():
    close = _fake_price_series()
    rsi = compute_rsi(close)
    assert rsi.dropna().between(0, 100).all()


def test_macd_shapes():
    close = _fake_price_series()
    macd_line, signal_line, hist = compute_macd(close)
    assert len(macd_line) == len(close) == len(signal_line) == len(hist)
    # histogram should equal macd - signal
    assert np.allclose((macd_line - signal_line).dropna(), hist.dropna())


def test_bollinger_band_ordering():
    close = _fake_price_series()
    upper, mid, lower = compute_bollinger_bands(close)
    valid = upper.notna() & mid.notna() & lower.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_add_all_indicators_has_expected_columns():
    df = _fake_ohlcv_df()
    enriched = add_all_indicators(df)
    expected = {"rsi", "macd", "macd_signal", "macd_hist", "bb_pct", "bb_width", "atr_pct"}
    assert expected.issubset(set(enriched.columns))
