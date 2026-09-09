"""Loads the trained model and turns today's indicators into a BUY/SELL/HOLD signal."""
import joblib

import config
from data_pipeline.market_data import get_ohlcv
from indicators.technical import add_all_indicators
from model.features import latest_feature_row

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(config.MODEL_PATH)
    return _model_cache


def generate_signal(ticker: str) -> dict:
    """
    Fetch recent data for `ticker`, compute indicators, and return a signal dict:
    {ticker, price, probability (of next-day up move), signal, atr_pct}
    """
    model = load_model()
    raw = get_ohlcv(ticker, period="6mo", interval=config.DATA_INTERVAL)
    enriched = add_all_indicators(raw)

    latest = enriched.iloc[-1]
    features = latest_feature_row(enriched)
    probability = float(model.predict_proba(features)[0][1])  # P(price up)

    if probability >= config.BUY_PROB_THRESHOLD:
        signal = "BUY"
    elif probability <= config.SELL_PROB_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "ticker": ticker,
        "price": float(latest["close"]),
        "probability": probability,
        "signal": signal,
        "atr_pct": float(latest["atr_pct"]),
    }
