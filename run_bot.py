"""
Main daily orchestration script.
For every ticker in the watchlist: generate a model signal, run it through the
risk-management layer, and (if it passes) submit a paper trade via Alpaca.

Run manually with `python run_bot.py`, or deploy scheduler.py for a 24/7
cloud-hosted daily run.
"""
import os

import config
from execution.broker import AlpacaBroker
from execution.executor import execute_signal
from model.predict import generate_signal
from watchlist import WATCHLIST


def run_once():
    if not os.path.exists(config.MODEL_PATH):
        raise SystemExit(
            "No trained model found. Run `python train_model.py` first."
        )

    broker = AlpacaBroker()
    account = broker.get_account()
    print(f"Connected to Alpaca paper account. Equity: ${float(account.equity):,.2f}\n")

    for ticker in WATCHLIST:
        try:
            signal = generate_signal(ticker)
            result = execute_signal(broker, signal)
            print(
                f"{ticker:6s} P(up)={signal['probability']:.2f}  "
                f"signal={signal['signal']:4s}  action={result['action']:7s}  "
                f"{result['reason']}"
            )
        except Exception as exc:
            print(f"{ticker:6s} ERROR: {exc}")


if __name__ == "__main__":
    run_once()
