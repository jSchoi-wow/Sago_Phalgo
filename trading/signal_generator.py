import pandas as pd
import numpy as np

from config.settings import SIGNAL_WEIGHTS, LSTM_BUY_THRESHOLD
from analysis.market_cycle import get_cycle_multiplier


def combine_signals(
    rule_signal: float,          # -1, 0, 1
    xgb_prob: float,             # 0~1 (상승 확률)
    lstm_return: float,          # 예측 수익률
    market_cycle: str = "횡보",
    weights: tuple = SIGNAL_WEIGHTS,
) -> dict:
    """
    3개 모델 신호를 가중 합산해 최종 신호 및 신뢰도 점수 반환.

    Returns:
        {
            "signal": 1(매수) / -1(매도) / 0(홀딩),
            "confidence": 0.0 ~ 1.0,
            "detail": {...}
        }
    """
    w_rule, w_xgb, w_lstm = weights

    # 각 신호를 -1~1 범위로 정규화
    rule_score = float(rule_signal)                           # 이미 -1~1
    xgb_score = (xgb_prob - 0.5) * 2                         # 0~1 → -1~1
    lstm_score = float(np.clip(lstm_return / 0.05, -1, 1))   # ±5% 기준 정규화

    # 가중 합산
    raw_score = w_rule * rule_score + w_xgb * xgb_score + w_lstm * lstm_score

    # 시장 사이클 조절
    cycle_mult = get_cycle_multiplier(market_cycle)
    if raw_score > 0:
        raw_score *= cycle_mult  # 하락장에서 매수 신호 약화

    # 최종 신호 결정
    if raw_score >= 0.3:
        signal = 1
    elif raw_score <= -0.3:
        signal = -1
    else:
        signal = 0

    confidence = min(abs(raw_score), 1.0)

    return {
        "signal": signal,
        "confidence": confidence,
        "raw_score": raw_score,
        "detail": {
            "rule_score": rule_score,
            "xgb_score": xgb_score,
            "lstm_score": lstm_score,
            "cycle_mult": cycle_mult,
        },
    }


def generate_signals_for_df(
    dates: pd.DatetimeIndex,
    rule_signals: pd.Series,
    xgb_probs: np.ndarray,
    lstm_returns: np.ndarray,
    market_cycles: pd.Series | None = None,
    weights: tuple = SIGNAL_WEIGHTS,
) -> pd.DataFrame:
    """
    날짜별로 통합 신호를 생성해 DataFrame으로 반환.
    """
    records = []
    for i, date in enumerate(dates):
        cycle = "횡보"
        if market_cycles is not None and date in market_cycles.index:
            cycle = market_cycles.loc[date]

        rule = rule_signals.iloc[i] if i < len(rule_signals) else 0
        xgb_p = float(xgb_probs[i]) if i < len(xgb_probs) else 0.5
        lstm_r = float(lstm_returns[i]) if i < len(lstm_returns) else 0.0

        result = combine_signals(rule, xgb_p, lstm_r, cycle, weights)
        result["date"] = date
        records.append(result)

    df = pd.DataFrame(records).set_index("date")
    return df
