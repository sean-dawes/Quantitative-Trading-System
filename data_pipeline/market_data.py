"""Fetches OHLCV price history and earnings-date data via yfinance."""
import pandas as pd
import yfinance as yf


def get_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download historical OHLCV bars for a single ticker."""
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df.dropna()


def get_latest_price(ticker: str) -> float:
    """Most recent close price for a ticker."""
    df = get_ohlcv(ticker, period="5d", interval="1d")
    return float(df["close"].iloc[-1])


def get_next_earnings_date(ticker: str):
    """Return the next known earnings date for a ticker, or None if unavailable."""
    try:
        t = yf.Ticker(ticker)
        edf = t.get_earnings_dates(limit=8)
        if edf is None or edf.empty:
            return None
        future = edf[edf.index >= pd.Timestamp.now(tz=edf.index.tz)]
        if future.empty:
            return None
        return future.index.min().to_pydatetime()
    except Exception:
        return None
