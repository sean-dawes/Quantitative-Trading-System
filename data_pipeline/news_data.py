"""Fetches recent headlines per ticker, used for sentiment gating."""
import yfinance as yf


def get_recent_headlines(ticker: str, limit: int = 10):
    """Return a list of recent headline strings for a ticker (best-effort)."""
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception:
        news_items = []

    headlines = []
    for item in news_items[:limit]:
        content = item.get("content", item)
        title = content.get("title") if isinstance(content, dict) else None
        if title:
            headlines.append(title)
    return headlines
