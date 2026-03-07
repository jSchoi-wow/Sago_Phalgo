import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from analysis.technical import add_all_indicators
from config.settings import (
    LABEL_HORIZON, WEEKLY_LABEL_HORIZON,
    TRAIN_END_DATE, TEST_START_DATE,
)

FEATURE_COLS = [
    "MA5", "MA20", "MA60", "MA120",
    "GoldenCross", "DeadCross",
    "RSI",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Mid", "BB_Lower", "BB_PctB",
    "Volume_Ratio", "OBV",
    # price-derived
    "Close_MA5_ratio", "Close_MA20_ratio", "Close_MA60_ratio",
    "High_Low_ratio", "Open_Close_ratio",
]


def add_price_derived(df: pd.DataFrame) -> pd.DataFrame:
    df["Close_MA5_ratio"] = df["Close"] / df["MA5"].replace(0, np.nan)
    df["Close_MA20_ratio"] = df["Close"] / df["MA20"].replace(0, np.nan)
    df["Close_MA60_ratio"] = df["Close"] / df["MA60"].replace(0, np.nan)
    df["High_Low_ratio"] = df["High"] / df["Low"].replace(0, np.nan)
    df["Open_Close_ratio"] = df["Open"] / df["Close"].replace(0, np.nan)
    return df


def add_labels(df: pd.DataFrame, horizon: int = LABEL_HORIZON) -> pd.DataFrame:
    future_close = df["Close"].shift(-horizon)
    df["Label"] = (future_close > df["Close"]).astype(int)
    df["FutureReturn"] = (future_close - df["Close"]) / df["Close"]
    return df


def build_features(
    daily: pd.DataFrame,
    weekly: pd.DataFrame | None = None,
    horizon: int = LABEL_HORIZON,
) -> pd.DataFrame:
    df = add_all_indicators(daily)
    df = add_price_derived(df)
    df = add_labels(df, horizon)

    if weekly is not None:
        weekly_ind = add_all_indicators(weekly)
        weekly_ind = weekly_ind[["MA5", "MA20"]].rename(
            columns={"MA5": "W_MA5", "MA20": "W_MA20"}
        )
        weekly_ind["W_Trend"] = (weekly_ind["W_MA5"] > weekly_ind["W_MA20"]).astype(int)
        weekly_ind.index = weekly_ind.index.normalize()
        df = df.join(weekly_ind["W_Trend"], how="left")
        df["W_Trend"] = df["W_Trend"].ffill().fillna(0)

    df.dropna(subset=FEATURE_COLS + ["Label"], inplace=True)
    return df


def split_train_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df.index <= TRAIN_END_DATE]
    test = df[df.index >= TEST_START_DATE]
    return train, test


def get_scaled_xy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple:
    """
    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in train.columns]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    y_train = train["Label"].values

    X_test = scaler.transform(test[feature_cols])
    y_test = test["Label"].values

    return X_train, y_train, X_test, y_test, scaler, feature_cols


def build_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    LSTM용 (samples, seq_len, features) 시퀀스 생성.
    Returns X, y, scaler
    """
    data = df[feature_cols].copy()
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
    else:
        scaled = scaler.transform(data)

    labels = df["FutureReturn"].values

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        label = labels[i]
        if np.isnan(label):
            continue
        X.append(scaled[i - seq_len : i])
        y.append(label)

    return np.array(X), np.array(y), scaler
