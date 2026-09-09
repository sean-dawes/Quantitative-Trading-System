"""Thin wrapper around Alpaca's paper-trading API (alpaca-py SDK)."""
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest

import config


class AlpacaBroker:
    def __init__(self):
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            raise RuntimeError(
                "Missing Alpaca API keys. Copy .env.example to .env and fill in your "
                "paper trading keys from https://app.alpaca.markets/paper/dashboard/overview"
            )
        self.client = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=config.PAPER_TRADING,
        )

    def get_account(self):
        return self.client.get_account()

    def get_equity(self) -> float:
        return float(self.get_account().equity)

    def get_open_position(self, ticker: str):
        try:
            return self.client.get_open_position(ticker)
        except Exception:
            return None

    def close_position(self, ticker: str):
        try:
            self.client.close_position(ticker)
            return True
        except Exception as exc:
            print(f"  could not close {ticker}: {exc}")
            return False

    def submit_bracket_order(self, ticker: str, qty: int, side: str,
                              take_profit: float, stop_loss: float):
        """Market entry with an attached take-profit and stop-loss leg."""
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            take_profit=TakeProfitRequest(limit_price=take_profit),
            stop_loss=StopLossRequest(stop_price=stop_loss),
        )
        return self.client.submit_order(request)
