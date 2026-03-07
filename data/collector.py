import pandas as pd
import FinanceDataReader as fdr
from pathlib import Path
from tqdm import tqdm

from config.settings import (
    CACHE_DIR, START_DATE, END_DATE,
    KOSPI200_STOCKS, KOSPI_INDEX_CODE, TOP_N_STOCKS,
)


def _cache_path(code: str, freq: str) -> Path:
    return CACHE_DIR / f"{code}_{freq}.csv"


def _load_or_fetch(code: str, freq: str, start: str, end: str) -> pd.DataFrame:
    path = _cache_path(code, freq)
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    df = fdr.DataReader(code, start, end)
    if df.empty:
        return df

    if freq == "W":
        df = _to_weekly(df)

    df.to_csv(path)
    return df


def _to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    existing = {k: v for k, v in agg.items() if k in daily.columns}
    weekly = daily.resample("W").agg(existing).dropna()
    return weekly


def get_kospi_index(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    path = _cache_path(KOSPI_INDEX_CODE, "D")
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)

    df = fdr.DataReader(KOSPI_INDEX_CODE, start, end)
    df.to_csv(path)
    return df


def get_stock_daily(code: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    return _load_or_fetch(code, "D", start, end)


def get_stock_weekly(code: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    path = _cache_path(code, "W")
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)

    daily = get_stock_daily(code, start, end)
    if daily.empty:
        return daily
    weekly = _to_weekly(daily)
    weekly.to_csv(path)
    return weekly


def get_stock_list_from_market() -> list[str]:
    """FinanceDataReader로 KOSPI 종목 목록 가져와 시총 상위 필터링."""
    try:
        listing = fdr.StockListing("KOSPI")
        if "Marcap" in listing.columns:
            listing = listing.sort_values("Marcap", ascending=False)
        codes = listing["Code"].astype(str).str.zfill(6).tolist()
        return codes[:TOP_N_STOCKS]
    except Exception as e:
        print(f"[Warning] 종목 목록 조회 실패, 기본 목록 사용: {e}")
        return KOSPI200_STOCKS


def collect_all(
    codes: list[str] | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
    verbose: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    여러 종목의 일봉/주봉 데이터를 수집해 딕셔너리로 반환.
    Returns:
        {code: {"daily": df, "weekly": df}}
    """
    if codes is None:
        codes = KOSPI200_STOCKS

    result: dict[str, dict[str, pd.DataFrame]] = {}
    iterator = tqdm(codes, desc="Collecting") if verbose else codes

    for code in iterator:
        try:
            daily = get_stock_daily(code, start, end)
            weekly = get_stock_weekly(code, start, end)
            if not daily.empty:
                result[code] = {"daily": daily, "weekly": weekly}
        except Exception as e:
            print(f"[Warning] {code} 수집 실패: {e}")

    # KOSPI 지수도 수집
    try:
        get_kospi_index(start, end)
    except Exception as e:
        print(f"[Warning] KOSPI 지수 수집 실패: {e}")

    if verbose:
        print(f"수집 완료: {len(result)}개 종목")
    return result


def refresh_cache(codes: list[str] | None = None) -> None:
    """캐시 파일 삭제 후 재수집."""
    if codes is None:
        codes = KOSPI200_STOCKS
    for code in codes:
        for freq in ("D", "W"):
            p = _cache_path(code, freq)
            if p.exists():
                p.unlink()
    get_kospi_index.__wrapped__ = None  # type: ignore[attr-defined]
    p = _cache_path(KOSPI_INDEX_CODE, "D")
    if p.exists():
        p.unlink()
    collect_all(codes)
