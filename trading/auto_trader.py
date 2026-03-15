"""
자동매매 엔진
- 기존 분석 시스템(규칙+XGBoost+LSTM) 신호를 받아 KIS API로 실제 주문 실행
- 손절 / 익절 자동 처리
- 스레드 안전 (QThread에서 호출 가능)
"""

import time
import threading
from datetime import datetime

import config.kis_config as cfg
from trading.kis_api import KISApi


class AutoTrader:
    """
    사용법:
        trader = AutoTrader(dry_run=True)   # 신호 감시만 (주문 없음)
        trader = AutoTrader(dry_run=False)  # 실제 주문
        trader.start(watch_codes=["005930", "000660"])
        trader.on_signal("005930", signal=1, confidence=0.8)
        trader.stop()
    """

    def __init__(self, log_fn=None, dry_run: bool = False, api: KISApi = None):
        self.api       = api or KISApi()   # 외부에서 주입 가능 (토큰 재사용)
        self.running   = False
        self.dry_run   = dry_run          # True: 신호 감시만, 실제 주문 없음
        self._lock     = threading.Lock()
        self._log      = log_fn or print

        # 보유 포지션: {code: {"qty": int, "avg_price": int, "entered_at": str}}
        self.positions: dict = {}

        # 주문 내역
        self.order_history: list = []

    # ── 시작 / 중지 ────────────────────────────────────────────────────
    def start(self, watch_codes: list[str]):
        if self.running:
            self._log("[AutoTrader] 이미 실행 중", "warn")
            return
        ok = self.api.check_connection()
        if not ok:
            self._log("[AutoTrader] KIS API 연결 실패. APP_KEY/SECRET을 확인하세요.", "error")
            return

        self.watch_codes = watch_codes
        self.running     = True

        mode = "[신호 감시 모드 - 주문 없음]" if self.dry_run else "[실제 매매 모드]"
        self._log(f"[AutoTrader] {mode} 시작 | 감시 종목: {watch_codes}", "ok")

        if not self.dry_run:
            self._sync_positions()

        # 손절/익절 감시 스레드 (실제 매매 모드만)
        if not self.dry_run:
            t = threading.Thread(target=self._monitor_loop, daemon=True)
            t.start()

    def stop(self):
        self.running = False
        self._log("[AutoTrader] 자동매매 중지")

    # ── 신호 수신 ─────────────────────────────────────────────────────
    def on_signal(self, code: str, signal: int, confidence: float):
        """
        signal:  1=매수, -1=매도, 0=홀딩
        confidence: 0~1
        """
        if not self.running:
            return

        signal_map = {1: "▲ 매수", -1: "▼ 매도", 0: "─ 홀딩"}

        # 신호 감시 모드: 주문 없이 로그만
        if self.dry_run:
            if signal != 0:
                self._log(
                    f"[DRY-RUN] {code} | {signal_map[signal]} 신호 | 신뢰도 {confidence:.2f} "
                    f"(실제 주문 없음)", "info"
                )
            return

        now = datetime.now()
        # 장 시간 체크 (09:00 ~ 15:20)
        if not (900 <= int(now.strftime("%H%M")) <= 1520):
            self._log(f"[AutoTrader] 장 외 시간 – 신호 무시 ({code}, {signal_map.get(signal)})", "info")
            return

        with self._lock:
            if signal == 1 and code not in self.positions:
                self._try_buy(code, confidence)
            elif signal == -1 and code in self.positions:
                self._try_sell(code, reason="신호 매도")

    # ── 매수 시도 ─────────────────────────────────────────────────────
    def _try_buy(self, code: str, confidence: float):
        try:
            # 최대 포지션 수 확인
            if len(self.positions) >= cfg.MAX_POSITIONS:
                self._log(f"[AutoTrader] 최대 포지션 수 도달 ({cfg.MAX_POSITIONS}), {code} 매수 건너뜀")
                return

            # 잔고 확인
            bal  = self.api.get_balance()
            cash = bal["cash"]
            if cash < cfg.ORDER_AMOUNT:
                self._log(f"[AutoTrader] 예수금 부족 ({cash:,}원 < {cfg.ORDER_AMOUNT:,}원)")
                return

            # 현재가로 수량 계산
            info  = self.api.get_price(code)
            price = info["price"]
            qty   = max(1, cfg.ORDER_AMOUNT // price)

            result = self.api.buy_market(code, qty)
            self.positions[code] = {
                "qty":        qty,
                "avg_price":  price,
                "entered_at": datetime.now().strftime("%H:%M:%S"),
                "confidence": confidence,
            }
            self.order_history.append(result)
            self._log(f"[AutoTrader] 매수 체결 | {code} {qty}주 @ {price:,}원 (신뢰도 {confidence:.2f})")

        except Exception as e:
            self._log(f"[AutoTrader] 매수 오류 ({code}): {e}")

    # ── 매도 시도 ─────────────────────────────────────────────────────
    def _try_sell(self, code: str, reason: str = ""):
        try:
            pos   = self.positions.get(code)
            if not pos:
                return

            result = self.api.sell_market(code, pos["qty"])
            result["reason"] = reason
            self.order_history.append(result)

            info  = self.api.get_price(code)
            pnl   = (info["price"] - pos["avg_price"]) / pos["avg_price"] * 100
            self._log(
                f"[AutoTrader] 매도 체결 | {code} {pos['qty']}주 | "
                f"수익률 {pnl:+.2f}% | 사유: {reason}"
            )
            del self.positions[code]

        except Exception as e:
            self._log(f"[AutoTrader] 매도 오류 ({code}): {e}")

    # ── 손절/익절 감시 루프 ────────────────────────────────────────────
    def _monitor_loop(self):
        while self.running:
            try:
                with self._lock:
                    codes = list(self.positions.keys())
                for code in codes:
                    self._check_stop_take(code)
            except Exception as e:
                self._log(f"[AutoTrader] 감시 루프 오류: {e}")
            time.sleep(30)   # 30초마다 체크

    def _check_stop_take(self, code: str):
        pos = self.positions.get(code)
        if not pos:
            return
        try:
            info  = self.api.get_price(code)
            price = info["price"]
            pnl   = (price - pos["avg_price"]) / pos["avg_price"]

            if pnl <= cfg.STOP_LOSS_PCT:
                self._log(f"[AutoTrader] 손절 발동 | {code} | 수익률 {pnl*100:.2f}%")
                with self._lock:
                    self._try_sell(code, reason=f"손절 ({pnl*100:.2f}%)")

            elif pnl >= cfg.TAKE_PROFIT_PCT:
                self._log(f"[AutoTrader] 익절 발동 | {code} | 수익률 {pnl*100:.2f}%")
                with self._lock:
                    self._try_sell(code, reason=f"익절 ({pnl*100:.2f}%)")

        except Exception as e:
            self._log(f"[AutoTrader] 손익 체크 오류 ({code}): {e}")

    # ── 실시간 잔고 동기화 ──────────────────────────────────────────────
    def _sync_positions(self):
        try:
            bal = self.api.get_balance()
            self.positions = {}
            for p in bal["positions"]:
                self.positions[p["code"]] = {
                    "qty":        p["qty"],
                    "avg_price":  p["avg_price"],
                    "entered_at": "기보유",
                    "confidence": 0.0,
                }
            self._log(f"[AutoTrader] 기존 포지션 동기화: {list(self.positions.keys())}")
        except Exception as e:
            self._log(f"[AutoTrader] 포지션 동기화 실패: {e}")
