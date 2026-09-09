"""
Standalone experiment script. NOT imported by run_bot.py, scheduler.py,
predict.py, or dashboard.py -- nothing here can ever affect the live bot,
no matter what it prints or how it's edited. Safe to rerun freely.

Tests two ideas for improving on the 50.4% honest baseline measured last
time (15 features, fair per-ticker time split, tuned XGBoost):

  Test A: add 2 "market-relative" features -- how far a stock's move is
          from the S&P 500's (SPY) move that same day.
  Test B: same features, but also stop forcing a label on tiny, noisy
          moves -- only moves bigger than MOVE_THRESHOLD count as a real
          "up" or "down" during training.

Run: python experiments/test_accuracy_ideas.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

import config
from data_pipeline.market_data import get_ohlcv
from indicators.technical import add_all_indicators
from model.train import PARAM_GRID, _fit_xgb, _three_way_split
from watchlist import WATCHLIST

BASE_FEATURES = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width", "atr_pct",
    "returns_1d", "returns_5d", "returns_10d", "returns_20d",
    "volume_change", "price_vs_sma50", "stoch_k", "obv_change",
]
RELATIVE_FEATURES = ["relative_return_1d", "relative_return_5d"]
MOVE_THRESHOLD = 0.005  # 0.5% -- smaller moves are dropped as noise, not labeled up/down


def add_relative_features(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """How much this stock's return differs from SPY's return on the same day."""
    out = stock_df.copy()
    market_close = market_df["close"].reindex(out.index).ffill()
    out["relative_return_1d"] = out["returns_1d"] - market_close.pct_change(1)
    out["relative_return_5d"] = out["returns_5d"] - market_close.pct_change(5)
    return out


def make_dataset(feature_columns, use_threshold: bool):
    print("  fetching SPY (market benchmark)...")
    market_raw = get_ohlcv("SPY", period=config.LOOKBACK_PERIOD, interval=config.DATA_INTERVAL)

    datasets = {}
    for ticker in WATCHLIST:
        try:
            raw = get_ohlcv(ticker, period=config.LOOKBACK_PERIOD, interval=config.DATA_INTERVAL)
            if len(raw) < 100:
                continue
            enriched = add_all_indicators(raw)
            enriched = add_relative_features(enriched, market_raw)

            forward_return = enriched["close"].shift(-config.PREDICTION_HORIZON_DAYS) / enriched["close"] - 1
            if use_threshold:
                label = pd.Series(
                    np.where(
                        forward_return > MOVE_THRESHOLD, 1.0,
                        np.where(forward_return < -MOVE_THRESHOLD, 0.0, np.nan),
                    ),
                    index=enriched.index,
                )
            else:
                label = (forward_return > 0).astype(float)

            enriched = enriched.copy()
            enriched["label"] = label
            enriched = enriched.dropna(subset=feature_columns + ["label"])

            X = enriched[feature_columns]
            y = enriched["label"].astype(int)
            datasets[ticker] = (X, y)
            print(f"    {ticker}: {len(X)} usable rows")
        except Exception as exc:
            print(f"    skipping {ticker}: {exc}")

    return datasets


def run_experiment(name: str, feature_columns, use_threshold: bool) -> float:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"Features ({len(feature_columns)}): {feature_columns}")
    datasets = make_dataset(feature_columns, use_threshold)

    total_rows = sum(len(X) for X, _ in datasets.values())
    pos_rate = sum(y.sum() for _, y in datasets.values()) / total_rows
    print(f"\nTotal usable rows: {total_rows}  (positive class rate: {pos_rate:.2%})")

    fit_X, fit_y, val_X, val_y, test_X, test_y = _three_way_split(datasets)

    best_params, best_val_acc = None, -1.0
    for params in PARAM_GRID:
        model = _fit_xgb(params, fit_X, fit_y)
        val_acc = accuracy_score(val_y, model.predict(val_X))
        if val_acc > best_val_acc:
            best_val_acc, best_params = val_acc, params

    train_X = pd.concat([fit_X, val_X], ignore_index=True)
    train_y = pd.concat([fit_y, val_y], ignore_index=True)
    model = _fit_xgb(best_params, train_X, train_y)

    preds = model.predict(test_X)
    test_acc = accuracy_score(test_y, preds)
    print(f"\nBest params: {best_params} (val acc {best_val_acc:.4f})")
    print(f"HOLD-OUT TEST ACCURACY: {test_acc:.4f}")
    print(classification_report(test_y, preds))
    return test_acc


if __name__ == "__main__":
    acc_a = run_experiment(
        "Test A: 15 base features + 2 market-relative features (no move threshold)",
        BASE_FEATURES + RELATIVE_FEATURES,
        use_threshold=False,
    )
    acc_b = run_experiment(
        "Test B: same 17 features + bigger-move label threshold (0.5%)",
        BASE_FEATURES + RELATIVE_FEATURES,
        use_threshold=True,
    )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("Previous baseline (15 features, no relative, no threshold): 0.5037")
    print(f"Test A (+ market-relative features):                         {acc_a:.4f}")
    print(f"Test B (+ market-relative + move threshold):                 {acc_b:.4f}")
