import pandas as pd

from analysis.technical import add_all_indicators
from config.settings import (
    RSI_OVERSOLD, RSI_OVERBOUGHT, RSI_BUY_MAX, VOLUME_RATIO_THRESHOLD,
)

BUY = 1
SELL = -1
HOLD = 0


def _weekly_trend(weekly: pd.DataFrame | None) -> pd.Series | None:
    if weekly is None:
        return None
    w = add_all_indicators(weekly)
    trend = (w["MA5"] > w["MA20"]).astype(int)
    trend.index = trend.index.normalize()
    return trend


def generate_signals(
    daily: pd.DataFrame,
    weekly: pd.DataFrame | None = None,
) -> pd.Series:
    """
    일봉 기반 규칙 신호 생성.
    주봉 MA5 > MA20 일 때만 매수 신호 활성화.

    Returns:
        pd.Series of {BUY=1, SELL=-1, HOLD=0}
    """
    df = add_all_indicators(daily).copy()
    w_trend = _weekly_trend(weekly)

    signals = pd.Series(HOLD, index=df.index, name="RuleSignal")

    for date, row in df.iterrows():
        if pd.isna(row.get("MA5")) or pd.isna(row.get("RSI")):
            continue

        # --- 매수 조건 ---
        ma_aligned = (
            row["MA5"] > row["MA20"]
            and row["MA20"] > row["MA60"]
        )
        rsi_ok = RSI_OVERSOLD <= row["RSI"] <= RSI_BUY_MAX
        volume_ok = row.get("Volume_Ratio", 0) >= VOLUME_RATIO_THRESHOLD
        above_bb_mid = row["Close"] >= row["BB_Mid"]

        # 주봉 추세 확인
        if w_trend is not None:
            norm_date = date.normalize() if hasattr(date, "normalize") else date
            # 가장 가까운 과거 주봉 날짜 찾기
            past = w_trend[w_trend.index <= norm_date]
            weekly_up = bool(past.iloc[-1]) if not past.empty else True
        else:
            weekly_up = True

        buy = ma_aligned and rsi_ok and volume_ok and above_bb_mid and weekly_up

        # --- 매도 조건 ---
        ma_bearish = row["MA5"] < row["MA20"]
        rsi_overbought = row["RSI"] > RSI_OVERBOUGHT
        bb_upper_reversal = (
            row["Close"] < row["BB_Upper"]
            and df["Close"].shift(1).loc[date] >= df["BB_Upper"].shift(1).loc[date]
            if date in df.index else False
        )

        sell = ma_bearish or rsi_overbought or bb_upper_reversal

        if buy:
            signals.loc[date] = BUY
        elif sell:
            signals.loc[date] = SELL

    return signals
