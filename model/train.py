"""
Trains an XGBoost classifier to predict next-day up/down direction from
technical-indicator features, pooling data across the whole watchlist.

Evaluation methodology: each ticker's own history is split in time order
into fit / validation / test slices (no shuffling, no leakage across time).
A small hyperparameter grid is tried and scored on the validation slice;
the winner is retrained on fit+validation and reported once, honestly,
on the untouched test slice.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

import config
from data_pipeline.market_data import get_ohlcv
from indicators.technical import add_all_indicators
from model.features import FEATURE_COLUMNS, prepare_dataset
from watchlist import WATCHLIST

PARAM_GRID = [
    dict(max_depth=3, learning_rate=0.05, n_estimators=300, min_child_weight=5, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=4, learning_rate=0.05, n_estimators=300, min_child_weight=5, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=4, learning_rate=0.03, n_estimators=500, min_child_weight=10, subsample=0.7, colsample_bytree=0.7),
    dict(max_depth=5, learning_rate=0.02, n_estimators=500, min_child_weight=10, subsample=0.8, colsample_bytree=0.6),
    dict(max_depth=3, learning_rate=0.1, n_estimators=200, min_child_weight=3, subsample=0.9, colsample_bytree=0.9),
]


def build_dataset_per_ticker(tickers=None, period: str = None) -> dict:
    """Returns {ticker: (X, y)} -- kept per-ticker so splits can respect each stock's own timeline."""
    tickers = tickers or WATCHLIST
    period = period or config.LOOKBACK_PERIOD

    datasets = {}
    for ticker in tickers:
        try:
            raw = get_ohlcv(ticker, period=period, interval=config.DATA_INTERVAL)
            if len(raw) < 100:
                print(f"  skipping {ticker}: not enough history")
                continue
            enriched = add_all_indicators(raw)
            X, y = prepare_dataset(enriched, horizon=config.PREDICTION_HORIZON_DAYS)
            datasets[ticker] = (X, y)
            print(f"  {ticker}: {len(X)} usable rows")
        except Exception as exc:
            print(f"  skipping {ticker}: {exc}")

    if not datasets:
        raise RuntimeError(
            "No training data could be built for any ticker. This usually means "
            "the data source (Yahoo Finance, via yfinance) couldn't be reached from "
            "this machine's network -- check your internet connection / firewall and try again."
        )
    return datasets


def _three_way_split(datasets: dict, val_frac: float = 0.15, test_frac: float = 0.2):
    """
    Time-ordered per-ticker split: for every ticker, the earliest rows go to
    `fit`, the next slice to `val`, and the most recent slice to `test`.
    Doing this per ticker (instead of concatenating all tickers first) keeps
    every stock represented in every slice, instead of the test set being
    dominated by whichever tickers happen to fall at the end of the pooled list.
    """
    fit_X, fit_y, val_X, val_y, test_X, test_y = [], [], [], [], [], []

    for ticker, (X, y) in datasets.items():
        n = len(X)
        test_start = int(n * (1 - test_frac))
        val_start = int(test_start * (1 - val_frac))

        fit_X.append(X.iloc[:val_start]); fit_y.append(y.iloc[:val_start])
        val_X.append(X.iloc[val_start:test_start]); val_y.append(y.iloc[val_start:test_start])
        test_X.append(X.iloc[test_start:]); test_y.append(y.iloc[test_start:])

    cat = lambda parts: pd.concat(parts, ignore_index=True)
    return (
        cat(fit_X), cat(fit_y),
        cat(val_X), cat(val_y),
        cat(test_X), cat(test_y),
    )


def _fit_xgb(params: dict, X_train, y_train) -> XGBClassifier:
    model = XGBClassifier(**params, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    return model


def train_model(tickers=None, period: str = None, save: bool = True) -> XGBClassifier:
    print("Fetching data and building features for the watchlist...")
    datasets = build_dataset_per_ticker(tickers, period)

    total_rows = sum(len(X) for X, _ in datasets.values())
    overall_pos_rate = sum(y.sum() for _, y in datasets.values()) / total_rows
    print(f"\nTotal usable rows across watchlist: {total_rows}  (positive class rate: {overall_pos_rate:.2%})")

    fit_X, fit_y, val_X, val_y, test_X, test_y = _three_way_split(datasets)

    print(f"\nSearching {len(PARAM_GRID)} hyperparameter combinations on the validation slice...")
    best_params, best_val_acc = None, -1.0
    for params in PARAM_GRID:
        model = _fit_xgb(params, fit_X, fit_y)
        val_acc = accuracy_score(val_y, model.predict(val_X))
        print(f"  {params} -> val accuracy {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_params = val_acc, params

    print(f"\nBest params by validation accuracy: {best_params} ({best_val_acc:.4f})")

    # Retrain on fit + validation combined (everything except the untouched test slice),
    # then report accuracy on the test slice exactly once.
    train_X = pd.concat([fit_X, val_X], ignore_index=True)
    train_y = pd.concat([fit_y, val_y], ignore_index=True)
    model = _fit_xgb(best_params, train_X, train_y)

    preds = model.predict(test_X)
    print("\nHold-out accuracy (untouched test slice):", accuracy_score(test_y, preds))
    print(classification_report(test_y, preds))

    if save:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        with open(config.FEATURE_COLUMNS_PATH, "w") as f:
            json.dump(FEATURE_COLUMNS, f)
        print(f"\nSaved model to {config.MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_model()
