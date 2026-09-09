"""
Technical indicator library used to build model features:
RSI, MACD, Bollinger Bands, ATR (for the volatility filter), a 50-day
trend measure, a stochastic oscillator, on-balance volume, and
longer-horizon returns.
All functions are pure pandas/numpy -- no external TA dependency.
"""
import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def compute_stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """%K: where today's close sits within the last `period` days' high/low range."""
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    return (100 * (df["close"] - low_min) / denom).fillna(50)


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-balance volume: running total of volume, added on up days, subtracted on down days."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with every indicator column added."""
    out = df.copy()
    out["rsi"] = compute_rsi(out["close"])

    macd_line, signal_line, hist = compute_macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(out["close"])
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    # %B: where price sits within the bands (0 = lower band, 1 = upper band)
    out["bb_pct"] = (out["close"] - bb_lower) / (bb_upper - bb_lower)
    out["bb_width"] = (bb_upper - bb_lower) / bb_mid

    out["atr"] = compute_atr(out)
    out["atr_pct"] = out["atr"] / out["close"]

    out["returns_1d"] = out["close"].pct_change(1)
    out["returns_5d"] = out["close"].pct_change(5)
    out["returns_10d"] = out["close"].pct_change(10)
    out["returns_20d"] = out["close"].pct_change(20)
    out["volume_change"] = out["volume"].pct_change(1)

    # Trend: how far price sits above/below its own 50-day average
    out["sma_50"] = compute_sma(out["close"], 50)
    out["price_vs_sma50"] = (out["close"] - out["sma_50"]) / out["sma_50"]

    # Momentum: stochastic %K
    out["stoch_k"] = compute_stochastic(out)

    # Volume confirmation: is volume trending with or against the price move
    obv = compute_obv(out)
    out["obv_change"] = obv.pct_change(5).replace([np.inf, -np.inf], np.nan)

    return out
