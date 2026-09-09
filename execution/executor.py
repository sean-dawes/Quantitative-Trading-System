"""
Combines a model signal with the risk-management layer to decide whether/how
to trade, then sends the order to Alpaca. This is the piece run_bot.py calls
for every ticker in the watchlist.
"""
import csv
import os
from datetime import datetime

import config
from execution.broker import AlpacaBroker
from risk.filters import passes_earnings_filter, passes_volatility_filter
from risk.position_sizing import calculate_position_size, stop_loss_price, take_profit_price
from risk.sentiment import passes_sentiment_gate


def _log_trade(row: dict):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    file_exists = os.path.isfile(config.TRADE_LOG_PATH)
    with open(config.TRADE_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def execute_signal(broker: AlpacaBroker, signal: dict) -> dict:
    """
    signal: the dict returned by model.predict.generate_signal()
    Returns a result dict describing what action (if any) was taken, and logs it.
    """
    ticker = signal["ticker"]
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "signal": signal["signal"],
        "probability": round(signal["probability"], 4),
        "price": signal["price"],
        "action": "NONE",
        "qty": 0,
        "reason": "",
    }

    existing_position = broker.get_open_position(ticker)

    # SELL signal: exit any existing long position.
    if signal["signal"] == "SELL":
        if existing_position is not None:
            broker.close_position(ticker)
            result["action"] = "CLOSED"
            result["qty"] = existing_position.qty
            result["reason"] = "sell signal, closed existing long"
        else:
            result["reason"] = "sell signal, no position held"
        _log_trade(result)
        return result

    if signal["signal"] == "HOLD":
        result["reason"] = "model uncertain (HOLD)"
        _log_trade(result)
        return result

    # signal["signal"] == "BUY" from here on.
    if existing_position is not None:
        result["reason"] = "already holding a position, skipping"
        _log_trade(result)
        return result

    if not passes_volatility_filter(signal["atr_pct"]):
        result["reason"] = f"volatility filter rejected (ATR%={signal['atr_pct']:.2%})"
        _log_trade(result)
        return result

    if not passes_earnings_filter(ticker):
        result["reason"] = "within earnings blackout window"
        _log_trade(result)
        return result

    if not passes_sentiment_gate(ticker):
        result["reason"] = "headline sentiment too negative"
        _log_trade(result)
        return result

    equity = broker.get_equity()
    qty = calculate_position_size(equity, signal["price"], signal["probability"])
    if qty <= 0:
        result["reason"] = "position size rounded down to 0 shares"
        _log_trade(result)
        return result

    tp = take_profit_price(signal["price"], side="buy")
    sl = stop_loss_price(signal["price"], side="buy")

    try:
        broker.submit_bracket_order(ticker, qty, "buy", take_profit=tp, stop_loss=sl)
        result["action"] = "BOUGHT"
        result["qty"] = qty
        result["reason"] = f"entered long, TP={tp}, SL={sl}"
    except Exception as exc:
        result["reason"] = f"order failed: {exc}"

    _log_trade(result)
    return result
