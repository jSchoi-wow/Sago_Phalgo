# ── 한국투자증권 KIS Developers API 설정 ──────────────────────────
# https://apiportal.koreainvestment.com 에서 앱 등록 후 발급
#
# 실전계좌 / 모의계좌 중 사용할 것에 맞게 설정하세요.

# ── 앱 키 (apiportal.koreainvestment.com → 내 앱 → 앱 상세) ─────
APP_KEY    = "YOUR_APP_KEY"       # 발급받은 APP KEY
APP_SECRET = "YOUR_APP_SECRET"    # 발급받은 APP SECRET

# ── 계좌번호 ────────────────────────────────────────────────────────
# 형식: 앞 8자리-뒤 2자리  예) "12345678-01"
ACCOUNT_NO    = "YOUR_ACCOUNT_NO"   # 전체 계좌번호 (앞 8자리)
ACCOUNT_PROD  = "01"                # 계좌상품코드 (보통 01)

# ── 실전 / 모의 선택 ────────────────────────────────────────────────
IS_REAL = True   # True=실전계좌, False=모의계좌

# ── URL 자동 결정 ────────────────────────────────────────────────────
if IS_REAL:
    BASE_URL = "https://openapi.koreainvestment.com:9443"
    TR_BUY   = "TTTC0802U"   # 주식 현금 매수
    TR_SELL  = "TTTC0801U"   # 주식 현금 매도
    TR_BAL   = "TTTC8434R"   # 주식 잔고 조회
    TR_PRICE = "FHKST01010100"  # 현재가 조회
else:
    BASE_URL = "https://openapivts.koreainvestment.com:29443"
    TR_BUY   = "VTTC0802U"
    TR_SELL  = "VTTC0801U"
    TR_BAL   = "VTTC8434R"
    TR_PRICE = "FHKST01010100"

# ── 자동매매 파라미터 ────────────────────────────────────────────────
ORDER_AMOUNT   = 500_000   # 1회 매수 금액 (원)
MAX_POSITIONS  = 5         # 최대 보유 종목 수
STOP_LOSS_PCT  = -0.05     # 손절 -5%
TAKE_PROFIT_PCT = 0.15     # 익절 +15%
