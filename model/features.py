"""Turns indicator-enriched OHLCV data into an ML-ready feature matrix + labels."""
import pandas as pd

FEATURE_COLUMNS = [
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width",
    "atr_pct",
    "returns_1d", "returns_5d",
    "volume_change",
]


def create_labels(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """1 if price is higher `horizon` days ahead (by more than `threshold`), else 0."""
    forward_return = df["close"].shift(-horizon) / df["close"] - 1
    return (forward_return > threshold).astype(int)


def prepare_dataset(df_with_indicators: pd.DataFrame, horizon: int = 1):
    """
    Build (X, y) from an indicator-enriched dataframe.
    Drops rows with NaNs from indicator warm-up or the label's forward-looking shift.
    """
    data = df_with_indicators.copy()
    data["label"] = create_labels(data, horizon=horizon)
    data = data.dropna(subset=FEATURE_COLUMNS + ["label"])
    X = data[FEATURE_COLUMNS]
    y = data["label"]
    return X, y


def latest_feature_row(df_with_indicators: pd.DataFrame) -> pd.DataFrame:
    """The single most recent row of features, used at prediction time."""
    return df_with_indicators[FEATURE_COLUMNS].iloc[[-1]]
