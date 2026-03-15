"""
신호 스케줄러
- 장중 N분마다 ML 모델 신호를 생성해 AutoTrader로 전달
- FinanceDataReader 과거 데이터 + KIS API 현재가를 합쳐 실시간 피처 구성
"""

import time
import threading
import traceback
from datetime import datetime, date

import numpy as np
import pandas as pd


class SignalScheduler:
    """
    사용법:
        sched = SignalScheduler(trader, kis_api, log_fn=...)
        sched.start(watch_codes=["005930", "000660"], interval_min=5)
        sched.stop()
    """

    def __init__(self, trader, kis_api, log_fn=None, signal_cb=None):
        """
        trader     : AutoTrader 인스턴스
        kis_api    : KISApi 인스턴스
        log_fn     : 로그 콜백 fn(msg, level)
        signal_cb  : 신호 테이블 업데이트 콜백 fn(code, row_dict)
        """
        self.trader    = trader
        self.api       = kis_api
        self.running   = False
        self._log      = log_fn    or (lambda msg, lv="info": print(msg))
        self._sig_cb   = signal_cb or (lambda code, row: None)
        self._thread   = None

        # 모델 캐시 (최초 1회 로드)
        self._xgb_model  = None
        self._xgb_scaler = None
        self._lstm_model = None
        self._models_ok  = False

    # ── 시작 / 중지 ────────────────────────────────────────────────────
    def start(self, watch_codes: list, interval_min: int = 5):
        if self.running:
            self._log("[Scheduler] 이미 실행 중", "warn")
            return
        self.watch_codes  = watch_codes
        self.interval_sec = interval_min * 60
        self.running      = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._log(f"[Scheduler] 시작 | {interval_min}분 간격 | 종목: {watch_codes}", "ok")

    def stop(self):
        self.running = False
        self._log("[Scheduler] 중지", "info")

    # ── 메인 루프 ──────────────────────────────────────────────────────
    def _loop(self):
        self._load_models()
        while self.running:
            now = datetime.now()
            hhmm = int(now.strftime("%H%M"))

            if 900 <= hhmm <= 1520:
                self._log(f"[Scheduler] 신호 생성 시작 ({now.strftime('%H:%M')})", "info")
                for code in self.watch_codes:
                    if not self.running:
                        break
                    try:
                        self._process_code(code)
                    except Exception as e:
                        self._log(f"[Scheduler] {code} 처리 오류: {e}", "error")
                        self._log(traceback.format_exc(), "error")
            else:
                self._log(f"[Scheduler] 장 외 시간 대기 중... ({now.strftime('%H:%M')})", "info")

            time.sleep(self.interval_sec)

    # ── 종목별 신호 생성 ────────────────────────────────────────────────
    def _process_code(self, code: str):
        from data.collector import get_stock_daily, get_stock_weekly
        from data.preprocessor import build_features, FEATURE_COLS
        from config.settings import LSTM_FEATURES, LSTM_SEQ_LEN
        from analysis.market_cycle import compute_market_cycle, get_cycle_multiplier
        from models.rule_based import generate_signals
        from trading.signal_generator import combine_signals
        import models.ml_model  as ml
        import models.lstm_model as lstm

        # 1. 과거 데이터 로드
        daily  = get_stock_daily(code)
        weekly = get_stock_weekly(code)

        # 2. 오늘 현재가로 마지막 행 갱신
        try:
            info = self.api.get_price(code)
            today = pd.Timestamp(date.today())
            if today not in daily.index:
                new_row = pd.DataFrame([{
                    "Open":   info["open"],
                    "High":   info["high"],
                    "Low":    info["low"],
                    "Close":  info["price"],
                    "Volume": info["volume"],
                }], index=[today])
                daily = pd.concat([daily, new_row])
            else:
                daily.loc[today, "Close"]  = info["price"]
                daily.loc[today, "Volume"] = info["volume"]
        except Exception as e:
            self._log(f"[Scheduler] {code} 현재가 갱신 실패 (과거 데이터로 진행): {e}", "warn")

        # 3. 피처 구성
        df = build_features(daily, weekly)
        if df.empty or len(df) < 60:
            self._log(f"[Scheduler] {code} 데이터 부족 건너뜀", "warn")
            return

        # 4. 규칙 기반 신호
        rule_sig = generate_signals(daily, weekly)
        r_val = float(rule_sig.iloc[-1]) if len(rule_sig) > 0 else 0.0

        # 5. XGBoost
        xgb_prob = 0.5
        if self._models_ok and self._xgb_model is not None:
            feat_cols = [c for c in FEATURE_COLS if c in df.columns]
            X = self._xgb_scaler.transform(df[feat_cols].iloc[[-1]])
            xgb_prob = float(ml.predict_proba(self._xgb_model, X)[-1])

        # 6. LSTM
        lstm_ret = 0.0
        if self._models_ok and self._lstm_model is not None:
            try:
                from data.preprocessor import build_lstm_sequences
                lstm_feats = [f for f in LSTM_FEATURES if f in df.columns]
                X_seq, _, _ = build_lstm_sequences(df, lstm_feats, LSTM_SEQ_LEN)
                if len(X_seq) > 0:
                    lstm_ret = float(lstm.predict_return(self._lstm_model, X_seq)[-1])
            except Exception as e:
                self._log(f"[Scheduler] {code} LSTM 예측 실패: {e}", "warn")

        # 7. 시장 사이클
        cycle = "횡보"
        try:
            from data.collector import get_kospi_index
            kospi = get_kospi_index()
            cycles = compute_market_cycle(kospi)
            today_ts = pd.Timestamp(date.today())
            cycle = cycles.iloc[-1] if len(cycles) > 0 else "횡보"
        except Exception:
            pass

        # 8. 통합 신호
        result = combine_signals(r_val, xgb_prob, lstm_ret, cycle)
        signal     = result["signal"]
        confidence = result["confidence"]

        # 9. 현재가 정보
        price_str = "-"
        change_str = "-"
        try:
            info      = self.api.get_price(code)
            price_str  = f"{info['price']:,}원"
            change_str = f"{info['change_pct']:+.2f}%"
        except Exception:
            pass

        signal_map = {1: "▲ 매수", -1: "▼ 매도", 0: "─ 홀딩"}
        row = {
            "price":      price_str,
            "change":     change_str,
            "rule":       "▲" if r_val > 0 else ("▼" if r_val < 0 else "─"),
            "xgb":        f"{xgb_prob:.2f}",
            "lstm":       f"{lstm_ret:+.3f}",
            "signal":     signal_map.get(signal, "─"),
            "confidence": f"{confidence:.2f}",
            "cycle":      cycle,
        }
        self._sig_cb(code, row)
        self._log(
            f"[Scheduler] {code} | {price_str} {change_str} | "
            f"신호={signal_map.get(signal)} | 신뢰도={confidence:.2f} | 사이클={cycle}",
            "ok" if signal == 1 else ("warn" if signal == -1 else "info"),
        )

        # 10. AutoTrader에 신호 전달
        self.trader.on_signal(code, signal, confidence)

    # ── 모델 로드 ──────────────────────────────────────────────────────
    def _load_models(self):
        try:
            import models.ml_model  as ml
            import models.lstm_model as lstm
            self._xgb_model, self._xgb_scaler = ml.load()
            self._lstm_model = lstm.load()
            self._models_ok  = True
            self._log("[Scheduler] 모델 로드 완료 (XGBoost + LSTM)", "ok")
        except Exception as e:
            self._models_ok = False
            self._log(f"[Scheduler] 모델 로드 실패 (규칙 기반만 사용): {e}", "warn")
