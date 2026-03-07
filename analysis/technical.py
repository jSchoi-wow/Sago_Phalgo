import pandas as pd
import numpy as np

from config.settings import (
    MA_WINDOWS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, VOLUME_MA_PERIOD,
)


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    for w in MA_WINDOWS:
        df[f"MA{w}"] = df["Close"].rolling(w).mean()
    return df


def add_golden_dead_cross(df: pd.DataFrame) -> pd.DataFrame:
    df["GoldenCross"] = (
        (df["MA5"] > df["MA20"]) & (df["MA5"].shift(1) <= df["MA20"].shift(1))
    ).astype(int)
    df["DeadCross"] = (
        (df["MA5"] < df["MA20"]) & (df["MA5"].shift(1) >= df["MA20"].shift(1))
    ).astype(int)
    return df


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame, period: int = BB_PERIOD, std_mult: float = BB_STD
) -> pd.DataFrame:
    ma = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    df["BB_Upper"] = ma + std_mult * std
    df["BB_Mid"] = ma
    df["BB_Lower"] = ma - std_mult * std
    df["BB_PctB"] = (df["Close"] - df["BB_Lower"]) / (
        df["BB_Upper"] - df["BB_Lower"]
    ).replace(0, np.nan)
    return df


def add_volume_indicators(df: pd.DataFrame, period: int = VOLUME_MA_PERIOD) -> pd.DataFrame:
    df["Volume_MA"] = df["Volume"].rolling(period).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"].replace(0, np.nan)

    # OBV
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (direction * df["Volume"]).cumsum()
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_moving_averages(df)
    df = add_golden_dead_cross(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_indicators(df)
    return df
