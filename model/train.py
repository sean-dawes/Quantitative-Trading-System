"""
Trains an XGBoost classifier to predict next-day up/down direction from
technical-indicator features, pooling data across the whole watchlist.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import config
from data_pipeline.market_data import get_ohlcv
from indicators.technical import add_all_indicators
from model.features import FEATURE_COLUMNS, prepare_dataset
from watchlist import WATCHLIST


def build_training_set(tickers=None, period: str = None):
    tickers = tickers or WATCHLIST
    period = period or config.LOOKBACK_PERIOD

    X_parts, y_parts = [], []
    for ticker in tickers:
        try:
            raw = get_ohlcv(ticker, period=period, interval=config.DATA_INTERVAL)
            if len(raw) < 60:
                print(f"  skipping {ticker}: not enough history")
                continue
            enriched = add_all_indicators(raw)
            X, y = prepare_dataset(enriched, horizon=config.PREDICTION_HORIZON_DAYS)
            X_parts.append(X)
            y_parts.append(y)
            print(f"  {ticker}: {len(X)} training rows")
        except Exception as exc:
            print(f"  skipping {ticker}: {exc}")

    if not X_parts:
        raise RuntimeError(
            "No training data could be built for any ticker. This usually means "
            "the data source (Yahoo Finance, via yfinance) couldn't be reached from "
            "this machine's network -- check your internet connection / firewall and try again."
        )

    X_all = pd.concat(X_parts, axis=0, ignore_index=True)
    y_all = pd.concat(y_parts, axis=0, ignore_index=True)
    return X_all, y_all


def train_model(tickers=None, period: str = None, save: bool = True) -> XGBClassifier:
    print("Fetching data and building features for the watchlist...")
    X, y = build_training_set(tickers, period)
    print(f"\nTotal training rows: {len(X)}  (positive class rate: {y.mean():.2%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # keep it time-ordered, no shuffling leakage
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nHold-out accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    if save:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        with open(config.FEATURE_COLUMNS_PATH, "w") as f:
            json.dump(FEATURE_COLUMNS, f)
        print(f"\nSaved model to {config.MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_model()
