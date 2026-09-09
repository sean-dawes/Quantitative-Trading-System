"""
Technical indicator library used to build model features:
RSI, MACD, Bollinger Bands, and ATR (for the volatility filter).
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
    out["volume_change"] = out["volume"].pct_change(1)

    return out
