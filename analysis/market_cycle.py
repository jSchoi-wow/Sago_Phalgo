import pandas as pd

from data.collector import get_kospi_index
from config.settings import START_DATE, END_DATE


CYCLE_UP = "상승장"
CYCLE_DOWN = "하락장"
CYCLE_SIDEWAYS = "횡보"


def compute_market_cycle(
    kospi_df: pd.DataFrame | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.Series:
    """
    코스피 지수 기반 시장 사이클 판단.
    Returns:
        pd.Series (index=date, values=CYCLE_UP/DOWN/SIDEWAYS)
    """
    if kospi_df is None:
        kospi_df = get_kospi_index(start, end)

    df = kospi_df[["Close"]].copy()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    def _label(row):
        if pd.isna(row["MA60"]) or pd.isna(row["MA120"]):
            return CYCLE_SIDEWAYS
        if row["Close"] > row["MA60"] and row["MA60"] > row["MA120"]:
            return CYCLE_UP
        if row["Close"] < row["MA60"] and row["MA60"] < row["MA120"]:
            return CYCLE_DOWN
        return CYCLE_SIDEWAYS

    cycle = df.apply(_label, axis=1)
    cycle.name = "MarketCycle"
    return cycle


def get_cycle_multiplier(cycle: str) -> float:
    """시장 사이클에 따른 매수 신호 강도 조절 계수."""
    return {CYCLE_UP: 1.0, CYCLE_SIDEWAYS: 0.7, CYCLE_DOWN: 0.4}.get(cycle, 0.7)
