"""
한국투자증권 KIS Developers REST API 래퍼
- 토큰 발급 / 자동 갱신
- 현재가 조회
- 주식 매수 / 매도 주문
- 잔고 조회
"""

import time
import requests
from datetime import datetime, timedelta

import config.kis_config as cfg


class KISApi:
    def __init__(self):
        self._token: str = ""
        self._token_expired: datetime = datetime.min
        self.session = requests.Session()
        self.session.headers.update({
            "content-type": "application/json; charset=utf-8",
            "appkey":    cfg.APP_KEY,
            "appsecret": cfg.APP_SECRET,
        })

    # ── 토큰 ──────────────────────────────────────────────────────────
    def _ensure_token(self):
        if datetime.now() < self._token_expired:
            return
        self._issue_token()

    def _issue_token(self):
        url  = f"{cfg.BASE_URL}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey":     cfg.APP_KEY,
            "appsecret":  cfg.APP_SECRET,
        }
        resp = self.session.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self._token         = data["access_token"]
        expires_in          = int(data.get("expires_in", 86400))
        self._token_expired = datetime.now() + timedelta(seconds=expires_in - 60)
        self.session.headers.update({"authorization": f"Bearer {self._token}"})
        print(f"[KIS] 토큰 발급 완료 (만료: {self._token_expired.strftime('%H:%M:%S')})")

    # ── 현재가 조회 ────────────────────────────────────────────────────
    def get_price(self, code: str) -> dict:
        """
        Returns:
            {
                "code": str,
                "name": str,
                "price": int,       # 현재가
                "change_pct": float,# 등락률 (%)
                "volume": int,
            }
        """
        self._ensure_token()
        url = f"{cfg.BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = {"tr_id": cfg.TR_PRICE}
        params  = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
        }
        resp = self.session.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        d = resp.json()
        if d.get("rt_cd") != "0":
            raise RuntimeError(f"현재가 조회 실패: {d.get('msg1', '')}")
        o = d["output"]
        return {
            "code":       code,
            "name":       o.get("hts_kor_isnm", code),
            "price":      int(o.get("stck_prpr", 0)),
            "change_pct": float(o.get("prdy_ctrt", 0)),
            "volume":     int(o.get("acml_vol", 0)),
            "open":       int(o.get("stck_oprc", 0)),
            "high":       int(o.get("stck_hgpr", 0)),
            "low":        int(o.get("stck_lwpr", 0)),
        }

    # ── 잔고 조회 ──────────────────────────────────────────────────────
    def get_balance(self) -> dict:
        """
        Returns:
            {
                "cash": int,                  # 예수금
                "positions": [                # 보유 종목
                    {"code", "name", "qty", "avg_price", "eval_value", "pnl_pct"}
                ],
                "total_eval": int,            # 평가 총액
            }
        """
        self._ensure_token()
        url = f"{cfg.BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = {"tr_id": cfg.TR_BAL}
        params = {
            "CANO":               cfg.ACCOUNT_NO,
            "ACNT_PRDT_CD":       cfg.ACCOUNT_PROD,
            "AFHR_FLPR_YN":       "N",
            "OFL_YN":             "",
            "INQR_DVSN":          "02",
            "UNPR_DVSN":          "01",
            "FUND_STTL_ICLD_YN":  "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN":          "01",
            "CTX_AREA_FK100":     "",
            "CTX_AREA_NK100":     "",
        }
        resp = self.session.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        d = resp.json()
        if d.get("rt_cd") != "0":
            raise RuntimeError(f"잔고 조회 실패: {d.get('msg1', '')}")

        positions = []
        for item in d.get("output1", []):
            qty = int(item.get("hldg_qty", 0))
            if qty <= 0:
                continue
            positions.append({
                "code":       item.get("pdno", ""),
                "name":       item.get("prdt_name", ""),
                "qty":        qty,
                "avg_price":  int(item.get("pchs_avg_pric", 0)),
                "eval_value": int(item.get("evlu_amt", 0)),
                "pnl_pct":    float(item.get("evlu_pfls_rt", 0)),
            })

        summary = d.get("output2", [{}])[0]
        return {
            "cash":       int(summary.get("dnca_tot_amt", 0)),
            "total_eval": int(summary.get("tot_evlu_amt", 0)),
            "positions":  positions,
        }

    # ── 매수 주문 ──────────────────────────────────────────────────────
    def buy_market(self, code: str, qty: int) -> dict:
        """시장가 매수"""
        return self._order(code, qty, is_buy=True, price=0, order_type="01")

    def buy_limit(self, code: str, qty: int, price: int) -> dict:
        """지정가 매수"""
        return self._order(code, qty, is_buy=True, price=price, order_type="00")

    # ── 매도 주문 ──────────────────────────────────────────────────────
    def sell_market(self, code: str, qty: int) -> dict:
        """시장가 매도"""
        return self._order(code, qty, is_buy=False, price=0, order_type="01")

    def sell_limit(self, code: str, qty: int, price: int) -> dict:
        """지정가 매도"""
        return self._order(code, qty, is_buy=False, price=price, order_type="00")

    # ── 내부 주문 로직 ─────────────────────────────────────────────────
    def _order(self, code: str, qty: int, is_buy: bool,
               price: int, order_type: str) -> dict:
        self._ensure_token()
        tr_id = cfg.TR_BUY if is_buy else cfg.TR_SELL
        url   = f"{cfg.BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
        headers = {"tr_id": tr_id}
        body = {
            "CANO":         cfg.ACCOUNT_NO,
            "ACNT_PRDT_CD": cfg.ACCOUNT_PROD,
            "PDNO":         code,
            "ORD_DVSN":     order_type,   # 00=지정가, 01=시장가
            "ORD_QTY":      str(qty),
            "ORD_UNPR":     str(price),   # 시장가일 때 0
        }
        resp = self.session.post(url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        d = resp.json()
        if d.get("rt_cd") != "0":
            raise RuntimeError(f"주문 실패: {d.get('msg1', '')}")
        side = "매수" if is_buy else "매도"
        out  = d.get("output", {})
        print(f"[KIS] {side} 주문 완료 | {code} {qty}주 | 주문번호: {out.get('ODNO', '-')}")
        return {
            "order_no": out.get("ODNO", ""),
            "code":     code,
            "qty":      qty,
            "side":     side,
            "time":     datetime.now().strftime("%H:%M:%S"),
        }

    # ── 계좌 유효성 확인 ───────────────────────────────────────────────
    def check_connection(self) -> bool:
        try:
            self._issue_token()
            return True
        except Exception as e:
            print(f"[KIS] 연결 실패: {e}")
            return False
